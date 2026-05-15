"""Core training and evaluation utilities for DINOv2-UNet."""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image

from .data import DATASET_SPECS, DatasetSpec
from .losses import ComboLoss, DeepSupervisionLoss
from .metrics import compute_segmentation_metrics, dice_iou_from_logits
from .models import DinoV2UNet
from .transforms import denorm
from .utils import set_seed


class EarlyStopping:
    """Save the best checkpoint and stop after repeated non-improvements."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = "best.pt",
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = -math.inf

    def __call__(self, val_score: float, model: torch.nn.Module) -> None:
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(
                f"Validation score improved ({self.val_score_min:.6f} --> "
                f"{val_score:.6f}). Saving model ..."
            )
        torch.save({"model": model.state_dict()}, self.path)
        self.val_score_min = val_score


@dataclass
class TrainConfig:
    """Configuration for the core train/eval loop."""

    dataset: str
    data_dir: str
    img_size: int = 448
    batch_size: int = 8
    epochs: int = 80
    backbone: str = "vit_base_patch14_dinov2"
    out_indices: Tuple[int, ...] = (2, 5, 8, 11)
    lr: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    num_workers: int = 4
    no_amp: bool = False
    freeze_blocks_until: int = 6
    patience: int = 10
    save_dir: str = ""
    seed: int = 42
    aug_mode: str = "strong"
    decoder_dropout: float = 0.2
    use_tta: bool = True
    pretrained_type: str = "dinov2"
    decoder_type: str = "simple"
    optimizer_strategy: str = "partial_finetune"
    deep_supervision: bool = True
    grad_clip: float = 1.0
    track_metrics: bool = True
    aux_weight_scale: float = 1.0
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    fold: int = 0
    num_folds: int = 1
    joint_train_specs: Optional[list] = None


@dataclass
class DatasetBundle:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_model(
    cfg: TrainConfig,
    pretrained: bool = True,
    deep_supervision: Optional[bool] = None,
) -> DinoV2UNet:
    """Build the segmentation model from training configuration."""
    return DinoV2UNet(
        backbone=cfg.backbone,
        out_indices=cfg.out_indices,
        pretrained=pretrained,
        freeze_blocks_until=cfg.freeze_blocks_until,
        num_classes=1,
        decoder_dropout=cfg.decoder_dropout,
        pretrained_type=cfg.pretrained_type,
        decoder_type=cfg.decoder_type,
        deep_supervision=cfg.deep_supervision if deep_supervision is None else deep_supervision,
    )


def align_img_size_to_patch(cfg: TrainConfig, model: DinoV2UNet) -> None:
    """Adjust image size to be divisible by the encoder patch size."""
    patch_size = getattr(model.encoder, "patch_size", 1)
    if patch_size > 1 and (cfg.img_size % patch_size) != 0:
        new_size = int(math.ceil(cfg.img_size / patch_size) * patch_size)
        print(
            f"Requested img_size {cfg.img_size} is not divisible by encoder "
            f"patch size {patch_size}. Resizing to {new_size}."
        )
        cfg.img_size = new_size


def get_optimizer(model: DinoV2UNet, strategy: str, cfg: TrainConfig):
    """Create AdamW optimizer with optional encoder fine-tuning strategies."""
    if strategy == "frozen_encoder":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == "full_finetune":
        for param in model.encoder.parameters():
            param.requires_grad = True
    elif strategy == "partial_finetune":
        if hasattr(model.encoder, "model") and hasattr(model.encoder.model, "blocks"):
            for i, block in enumerate(model.encoder.model.blocks):
                requires_grad = i >= cfg.freeze_blocks_until
                for param in block.parameters():
                    param.requires_grad = requires_grad
    else:
        raise ValueError(f"Unknown optimizer strategy: {strategy}")

    for param in model.decoder.parameters():
        param.requires_grad = True
    if hasattr(model, "aux_heads"):
        for param in model.aux_heads.parameters():
            param.requires_grad = True

    enc_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith("encoder") and param.requires_grad
    ]
    dec_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("encoder") and param.requires_grad
    ]

    print(f"Optimizer strategy: {strategy}")
    print(f"Trainable encoder params: {len(enc_params)}")
    print(f"Trainable decoder params: {len(dec_params)}")

    return torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_backbone, "name": "enc"},
            {"params": dec_params, "lr": cfg.lr, "name": "dec"},
        ],
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
):
    """Build cosine learning-rate decay with optional linear warm-up."""
    total_iters = max(1, epochs * steps_per_epoch)
    warmup_iters = warmup_epochs * steps_per_epoch

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = (current_iter - warmup_iters) / float(
            max(1, total_iters - warmup_iters)
        )
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_datasets(cfg: TrainConfig, spec: DatasetSpec) -> DatasetBundle:
    """Build train/val/test datasets, including optional joint training."""
    dataset_cls = spec.cls

    if cfg.joint_train_specs:
        train_datasets, val_datasets, test_datasets = [], [], []
        mean, std = None, None
        for data_cls, data_dir in cfg.joint_train_specs:
            train_ds = data_cls(
                data_dir,
                "train",
                cfg.img_size,
                seed=cfg.seed,
                aug_mode=cfg.aug_mode,
                fold_idx=cfg.fold,
                num_folds=cfg.num_folds,
            )
            val_ds = data_cls(
                data_dir,
                "val",
                cfg.img_size,
                seed=cfg.seed,
                aug_mode="none",
                fold_idx=cfg.fold,
                num_folds=cfg.num_folds,
            )
            test_ds = data_cls(
                data_dir,
                "test",
                cfg.img_size,
                seed=cfg.seed,
                aug_mode="none",
                fold_idx=cfg.fold,
                num_folds=cfg.num_folds,
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            if mean is None:
                mean, std = train_ds.mean, train_ds.std

        return DatasetBundle(
            train=ConcatDataset(train_datasets),
            val=ConcatDataset(val_datasets),
            test=ConcatDataset(test_datasets),
            mean=mean,
            std=std,
        )

    train_ds = dataset_cls(
        cfg.data_dir,
        "train",
        cfg.img_size,
        seed=cfg.seed,
        aug_mode=cfg.aug_mode,
        fold_idx=cfg.fold,
        num_folds=cfg.num_folds,
    )
    val_ds = dataset_cls(
        cfg.data_dir,
        "val",
        cfg.img_size,
        seed=cfg.seed,
        aug_mode="none",
        fold_idx=cfg.fold,
        num_folds=cfg.num_folds,
    )
    test_ds = dataset_cls(
        cfg.data_dir,
        "test",
        cfg.img_size,
        seed=cfg.seed,
        aug_mode="none",
        fold_idx=cfg.fold,
        num_folds=cfg.num_folds,
    )
    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        mean=train_ds.mean,
        std=train_ds.std,
    )


def build_loaders(cfg: TrainConfig, datasets: DatasetBundle) -> LoaderBundle:
    """Build DataLoaders for all splits."""
    drop_last = len(datasets.train) >= cfg.batch_size
    train_loader = DataLoader(
        datasets.train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    eval_batch_size = cfg.batch_size * 2
    val_loader = DataLoader(
        datasets.val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return LoaderBundle(train=train_loader, val=val_loader, test=test_loader)


def train_one_epoch(
    model,
    loader,
    optim,
    scheduler,
    scaler,
    loss_fn,
    device,
    grad_clip: float = 1.0,
    max_batches: Optional[int] = None,
):
    """Run one training epoch."""
    model.train()
    running_loss, running_dice, running_iou, iters = 0.0, 0.0, 0.0, 0

    for batch_idx, (imgs, msks, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)
            if isinstance(outputs, dict):
                loss = loss_fn(outputs, msks)
                main_logits = outputs["main"]
            else:
                loss = loss_fn({"main": outputs}, msks)
                main_logits = outputs

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        scheduler.step()
        dice, iou = dice_iou_from_logits(main_logits, msks)
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        iters += 1

    denom = max(iters, 1)
    return running_loss / denom, running_dice / denom, running_iou / denom


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_tta: bool = False,
    max_batches: Optional[int] = None,
):
    """Evaluate the model on a dataset split."""
    model.eval()
    combo_loss = ComboLoss(0.5, 0.5)
    metrics_sum = {}
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (imgs, msks, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs, msks = imgs.to(device), msks.to(device)

        outputs = model(imgs)
        logits = outputs["main"] if isinstance(outputs, dict) else outputs

        if use_tta:
            outputs_hf = model(TF.hflip(imgs))
            logits_hf = outputs_hf["main"] if isinstance(outputs_hf, dict) else outputs_hf
            logits = (logits + TF.hflip(logits_hf)) / 2.0

        loss = combo_loss(logits, msks)
        batch_metrics = compute_segmentation_metrics(logits, msks)
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        for key, value in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + value * batch_size
        total_samples += batch_size

    denom = max(total_samples, 1)
    averaged = {key: metrics_sum[key] / denom for key in metrics_sum}
    averaged["loss"] = total_loss / denom
    return averaged


@torch.no_grad()
def save_visuals(model, loader, device, save_dir: str, mean, std, max_batches: int = 2):
    """Save a small qualitative sample of images, masks, and predictions."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for batch_idx, (imgs, msks, names) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        imgs_gpu = imgs.to(device)
        outputs = model(imgs_gpu)
        logits = outputs["main"] if isinstance(outputs, dict) else outputs
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        imgs_denorm = denorm(imgs, mean, std)
        for b in range(imgs.size(0)):
            base_name = os.path.splitext(os.path.basename(names[b]))[0]
            save_image(imgs_denorm[b], os.path.join(save_dir, f"{base_name}_img.png"))
            save_image(msks[b], os.path.join(save_dir, f"{base_name}_gt.png"))
            save_image(preds[b], os.path.join(save_dir, f"{base_name}_pred.png"))


def _format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} | loss {metrics['loss']:.4f} mdice {metrics['mDice']:.4f} "
        f"miou {metrics['mIoU']:.4f} mae {metrics['mae']:.4f} "
        f"Fw {metrics['Fbeta_w']:.4f} Salpha {metrics['s_alpha']:.4f} "
        f"mE {metrics['mE']:.4f}/{metrics['mE_max']:.4f} "
        f"prec {metrics['precision']:.4f} rec {metrics['recall']:.4f} "
        f"acc {metrics['accuracy']:.4f}"
    )


def run_training(
    cfg: TrainConfig,
    dataset_key: str,
    spec: Optional[DatasetSpec] = None,
):
    """Execute model training, validation, checkpointing, and final testing."""
    if spec is None:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key '{dataset_key}'.")
        spec = DATASET_SPECS[dataset_key]

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = build_model(cfg, pretrained=True).to(device)
    align_img_size_to_patch(cfg, model)
    datasets = build_datasets(cfg, spec)
    loaders = build_loaders(cfg, datasets)

    optim = get_optimizer(model, cfg.optimizer_strategy, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(not cfg.no_amp and device == "cuda"))
    scheduler = build_scheduler(
        optim,
        cfg.epochs,
        max(1, len(loaders.train)),
        cfg.warmup_epochs,
    )

    aux_weights = [0.4 * cfg.aux_weight_scale, 0.3 * cfg.aux_weight_scale, 0.2 * cfg.aux_weight_scale]
    loss_fn = DeepSupervisionLoss(base_loss=ComboLoss(0.5, 0.5), aux_weights=aux_weights)

    print(f"Using dataset '{dataset_key}' from {cfg.data_dir}")
    print(f"Saving outputs to {cfg.save_dir}")
    print(f"Device: {device}")
    print(f"Deep supervision: {cfg.deep_supervision}")

    metrics_history = None
    if cfg.track_metrics:
        from .metrics_tracking import MetricsHistory

        metrics_history = MetricsHistory(cfg.save_dir)

    early_stopper = EarlyStopping(
        patience=cfg.patience,
        verbose=True,
        path=os.path.join(cfg.save_dir, "best.pt"),
    )
    best_val_metrics: Optional[Dict[str, float]] = None
    best_score = -float("inf")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model,
            loaders.train,
            optim,
            scheduler,
            scaler,
            loss_fn,
            device,
            grad_clip=cfg.grad_clip,
            max_batches=cfg.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            loaders.val,
            device,
            max_batches=cfg.max_eval_batches,
        )
        dt = time.time() - t0

        if metrics_history:
            metrics_history.record_epoch(
                epoch,
                "train",
                {"loss": tr_loss, "dice": tr_dice, "iou": tr_iou},
            )
            metrics_history.record_epoch(epoch, "val", val_metrics)
            metrics_history.save_json(os.path.join(cfg.save_dir, "metrics_history.json"))

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | time {dt:.1f}s | "
            f"train: loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} | "
            + _format_metrics("val", val_metrics)
        )

        score = (val_metrics["mDice"] + val_metrics["mIoU"]) / 2.0
        if score > best_score:
            best_score = score
            best_val_metrics = val_metrics
        early_stopper(score, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(os.path.join(cfg.save_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(
        model,
        loaders.test,
        device,
        use_tta=cfg.use_tta,
        max_batches=cfg.max_eval_batches,
    )
    print(_format_metrics(f"Test with {'TTA' if cfg.use_tta else 'no TTA'}", test_metrics))

    save_visuals(
        model,
        loaders.test,
        device,
        os.path.join(cfg.save_dir, "vis_test"),
        datasets.mean,
        datasets.std,
        max_batches=4,
    )

    return {
        "best_val": best_val_metrics or {},
        "test": test_metrics,
    }
