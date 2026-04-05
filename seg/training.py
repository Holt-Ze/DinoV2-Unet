"""Training loop and evaluation utilities for DINOv2-UNet.

This module implements the transfer-oriented training strategy described in
the paper (Section 3.6), including:
- Partial fine-tuning with differential learning rates
- Cosine annealing with linear warm-up
- Gradient clipping
- Deep supervision
- Early stopping based on composite validation score
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import DATASET_SPECS, DatasetSpec
from .losses import ComboLoss, DeepSupervisionLoss
from .metrics import compute_segmentation_metrics, dice_iou_from_logits
from .models import DinoV2UNet
from .transforms import denorm
from .utils import set_seed
from .profiling import describe_profile


class EarlyStopping:
    """Early stopping based on a composite validation score.

    Monitors the validation score and saves the best model checkpoint.
    Training is stopped if no improvement is observed for `patience` epochs.

    Args:
        patience: Number of epochs to wait before stopping.
        verbose: Whether to print status messages.
        delta: Minimum improvement to qualify as progress.
        path: File path for saving the best model checkpoint.
    """

    def __init__(self, patience: int = 7, verbose: bool = False,
                 delta: float = 0, path: str = "best.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = -math.inf

    def __call__(self, val_score: float, model: torch.nn.Module) -> None:
        """Check if the validation score improved and save checkpoint."""
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        """Save model state dict when validation score improves."""
        if self.verbose:
            print(f"Validation score improved ({self.val_score_min:.6f} --> "
                  f"{val_score:.6f}). Saving model ...")
        torch.save({"model": model.state_dict()}, self.path)
        self.val_score_min = val_score


@dataclass
class TrainConfig:
    """Training configuration dataclass.

    All hyperparameters are exposed via command-line arguments in train.py.
    Default values match the paper (Section 3.6, Section 4.1).

    Attributes:
        dataset: Dataset key (e.g., 'kvasir', 'clinicdb').
        data_dir: Path to the dataset directory.
        img_size: Input image resolution (default: 448).
        batch_size: Training batch size (default: 8).
        epochs: Maximum training epochs (default: 80).
        backbone: ViT backbone model name.
        out_indices: Transformer block indices for feature extraction.
        lr: Decoder learning rate (default: 1e-3).
        lr_backbone: Encoder learning rate (default: 1e-5).
        weight_decay: AdamW weight decay (default: 0.01).
        warmup_epochs: Linear warm-up duration (default: 5).
        num_workers: DataLoader worker processes.
        no_amp: Disable automatic mixed precision.
        freeze_blocks_until: Freeze encoder blocks with index < this value.
        patience: Early stopping patience (default: 10).
        save_dir: Directory for checkpoints and outputs.
        seed: Random seed for reproducibility (default: 42).
        aug_mode: Data augmentation strength ('strong', 'weak', 'none').
        decoder_dropout: Dropout rate in decoder ConvBlocks.
        use_tta: Enable test-time augmentation (horizontal flip).
        profile: Enable model profiling (Params/FLOPs/FPS).
        pretrained_type: Pretrained weight type ('dinov2' or 'imagenet_supervised').
        decoder_type: Decoder variant ('simple' or 'complex').
        optimizer_strategy: Fine-tuning strategy ('partial_finetune', 'frozen_encoder', 'full_finetune').
        deep_supervision: Enable deep supervision with auxiliary heads.
        grad_clip: Maximum gradient norm for clipping.
    """

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
    profile: bool = True
    pretrained_type: str = "dinov2"
    decoder_type: str = "simple"
    optimizer_strategy: str = "partial_finetune"
    deep_supervision: bool = True
    grad_clip: float = 1.0
    track_metrics: bool = True
    track_gradients: bool = False
    track_activations: bool = False
    save_failure_analysis: bool = True
    fold: int = 0
    num_folds: int = 1
    joint_train_specs: Optional[list] = None


def get_optimizer(model: DinoV2UNet, strategy: str, cfg: TrainConfig):
    """Create AdamW optimizer with differential learning rates.

    Implements the paper's training strategy (Section 3.6, Eq. 13):
    - Encoder trainable params: lr = 1e-5
    - Decoder params: lr = 1e-3

    Three fine-tuning strategies are supported (ablation study, Section 4.9.2):
    - 'frozen_encoder': Freeze all encoder params (linear probing).
    - 'full_finetune': Train all encoder params.
    - 'partial_finetune': Freeze blocks 0-(N-1), train blocks N-11 (default).

    Args:
        model: The DinoV2UNet model.
        strategy: One of 'frozen_encoder', 'full_finetune', 'partial_finetune'.
        cfg: Training configuration.

    Returns:
        AdamW optimizer with parameter groups.
    """
    if strategy == "frozen_encoder":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "full_finetune":
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif strategy == "partial_finetune":
        if hasattr(model.encoder, "model") and hasattr(model.encoder.model, "blocks"):
            for i, blk in enumerate(model.encoder.model.blocks):
                requires = i >= cfg.freeze_blocks_until
                for p in blk.parameters():
                    p.requires_grad = requires
    else:
        raise ValueError(f"Unknown optimizer strategy: {strategy}")

    # Ensure decoder and auxiliary heads are always trainable
    for p in model.decoder.parameters():
        p.requires_grad = True
    if hasattr(model, "aux_heads"):
        for p in model.aux_heads.parameters():
            p.requires_grad = True

    enc_params = [p for n, p in model.named_parameters()
                  if n.startswith("encoder") and p.requires_grad]
    dec_params = [p for n, p in model.named_parameters()
                  if not n.startswith("encoder") and p.requires_grad]

    print(f"Optimizer Strategy: {strategy}")
    print(f"Trainable Encoder Params: {len(enc_params)}")
    print(f"Trainable Decoder Params: {len(dec_params)}")

    return torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_backbone, "name": "enc"},
            {"params": dec_params, "lr": cfg.lr, "name": "dec"},
        ],
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int,
                    warmup_epochs: int = 0):
    """Build cosine annealing learning rate scheduler with linear warm-up.

    Implements paper Eq. 14:
    - Linear warm-up from 0 to base LR over warmup_epochs.
    - Cosine decay from base LR to lambda_min * base LR.

    Args:
        optimizer: The optimizer to schedule.
        epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.
        warmup_epochs: Number of warm-up epochs.

    Returns:
        LambdaLR scheduler.
    """
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


def train_one_epoch(model, loader, optim, scheduler, scaler, loss_fn, device,
                    grad_clip: float = 1.0):
    """Run one training epoch.

    Args:
        model: The segmentation model.
        loader: Training DataLoader.
        optim: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: AMP GradScaler.
        loss_fn: Loss function (DeepSupervisionLoss or ComboLoss).
        device: Target device ('cuda' or 'cpu').
        grad_clip: Maximum gradient norm for clipping (paper Section 3.6).

    Returns:
        Tuple of (mean_loss, mean_dice, mean_iou) for the epoch.
    """
    model.train()
    running_loss, running_dice, running_iou, iters = 0.0, 0.0, 0.0, 0

    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(imgs)
            # Support both deep supervision (dict) and plain tensor outputs
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
        d, i = dice_iou_from_logits(main_logits, msks)
        running_loss += loss.item()
        running_dice += d
        running_iou += i
        iters += 1

    denom = max(iters, 1)
    return running_loss / denom, running_dice / denom, running_iou / denom


@torch.no_grad()
def evaluate(model, loader, device, use_tta: bool = False):
    """Evaluate the model on a dataset split.

    Computes all segmentation metrics defined in the paper (Section 4.2):
    mDice, mIoU, MAE, Fw_beta, S_alpha, mE_xi, precision, recall, accuracy.

    Args:
        model: The segmentation model.
        loader: Evaluation DataLoader.
        device: Target device.
        use_tta: Enable test-time augmentation (horizontal flip).

    Returns:
        Dict of averaged metric values including 'loss'.
    """
    model.eval()
    combo_loss = ComboLoss(0.5, 0.5)
    metrics_sum = {}
    total_loss = 0.0
    total_samples = 0

    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)

        # Get main logits (inference mode: no aux heads)
        outputs = model(imgs)
        if isinstance(outputs, dict):
            logits = outputs["main"]
        else:
            logits = outputs

        if use_tta:
            outputs_hf = model(TF.hflip(imgs))
            if isinstance(outputs_hf, dict):
                logits_hf = outputs_hf["main"]
            else:
                logits_hf = outputs_hf
            logits = (logits + TF.hflip(logits_hf)) / 2.0

        loss = combo_loss(logits, msks)
        batch_metrics = compute_segmentation_metrics(logits, msks)
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        for key, value in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + value * batch_size
        total_samples += batch_size

    denom = max(total_samples, 1)
    averaged = {k: metrics_sum[k] / denom for k in metrics_sum}
    averaged["loss"] = total_loss / denom
    return averaged


@torch.no_grad()
def save_visuals(model, loader, device, save_dir: str, mean, std,
                 max_batches: int = 2):
    """Save visual predictions for qualitative analysis.

    Exports input images, ground-truth masks, and predicted masks as PNG files.

    Args:
        model: The segmentation model.
        loader: Evaluation DataLoader.
        device: Target device.
        save_dir: Output directory for saved images.
        mean: Normalization mean for de-normalization.
        std: Normalization std for de-normalization.
        max_batches: Maximum number of batches to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    cnt = 0
    for imgs, msks, names in loader:
        imgs_gpu = imgs.to(device)
        outputs = model(imgs_gpu)
        if isinstance(outputs, dict):
            logits = outputs["main"]
        else:
            logits = outputs
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        imgs_denorm = denorm(imgs, mean, std)
        for b in range(imgs.size(0)):
            base_name = os.path.splitext(os.path.basename(names[b]))[0]
            save_image(imgs_denorm[b], os.path.join(save_dir, f"{base_name}_img.png"))
            save_image(msks[b], os.path.join(save_dir, f"{base_name}_gt.png"))
            save_image(preds[b], os.path.join(save_dir, f"{base_name}_pred.png"))
        cnt += 1
        if cnt >= max_batches:
            break


def run_training(cfg: TrainConfig, dataset_key: str,
                 spec: Optional[DatasetSpec] = None):
    """Execute the full training pipeline.

    Implements the complete training procedure described in the paper:
    1. Model initialization with pretrained encoder
    2. Optimizer setup with differential learning rates
    3. Training loop with deep supervision, gradient clipping, AMP
    4. Early stopping based on composite validation score (Eq. 16)
    5. Final evaluation on test set with optional TTA

    Args:
        cfg: Training configuration.
        dataset_key: Dataset identifier string.
        spec: Dataset specification (auto-resolved if None).

    Returns:
        Dict containing 'best_val' and 'test' metric dictionaries.
    """
    if spec is None:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key '{dataset_key}'.")
        spec = DATASET_SPECS[dataset_key]

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.save_dir, exist_ok=True)

    dataset_cls = spec.cls

    model = DinoV2UNet(
        cfg.backbone, cfg.out_indices, True, cfg.freeze_blocks_until, 1,
        decoder_dropout=cfg.decoder_dropout,
        pretrained_type=cfg.pretrained_type,
        decoder_type=cfg.decoder_type,
        deep_supervision=cfg.deep_supervision,
    ).to(device)

    patch_size = getattr(model.encoder, "patch_size", 1)
    if patch_size > 1 and (cfg.img_size % patch_size) != 0:
        new_size = int(math.ceil(cfg.img_size / patch_size) * patch_size)
        print(f"Requested img_size {cfg.img_size} is not divisible by encoder "
              f"patch size {patch_size}. Resizing to {new_size}.")
        cfg.img_size = new_size

    if cfg.profile:
        use_amp = not cfg.no_amp and device == "cuda"
        profile_msg = describe_profile(
            model, (cfg.img_size, cfg.img_size), device, use_amp=use_amp
        )
        print(profile_msg)

    if cfg.joint_train_specs:
        train_dss, val_dss, test_dss = [], [], []
        import torch.utils.data as data_utils
        mean, std = None, None
        for (d_cls, d_dir) in cfg.joint_train_specs:
            t = d_cls(d_dir, "train", cfg.img_size, seed=cfg.seed, aug_mode=cfg.aug_mode, fold_idx=cfg.fold, num_folds=cfg.num_folds)
            v = d_cls(d_dir, "val", cfg.img_size, seed=cfg.seed, aug_mode="none", fold_idx=cfg.fold, num_folds=cfg.num_folds)
            te = d_cls(d_dir, "test", cfg.img_size, seed=cfg.seed, aug_mode="none", fold_idx=cfg.fold, num_folds=cfg.num_folds)
            train_dss.append(t)
            val_dss.append(v)
            test_dss.append(te)
            if mean is None: mean, std = t.mean, t.std
        train_ds = data_utils.ConcatDataset(train_dss)
        val_ds = data_utils.ConcatDataset(val_dss)
        test_ds = data_utils.ConcatDataset(test_dss)
        train_ds.mean, train_ds.std = mean, std
    else:
        train_ds = dataset_cls(
            cfg.data_dir, "train", cfg.img_size, seed=cfg.seed, aug_mode=cfg.aug_mode, fold_idx=cfg.fold, num_folds=cfg.num_folds
        )
        val_ds = dataset_cls(
            cfg.data_dir, "val", cfg.img_size, seed=cfg.seed, aug_mode="none", fold_idx=cfg.fold, num_folds=cfg.num_folds
        )
        test_ds = dataset_cls(
            cfg.data_dir, "test", cfg.img_size, seed=cfg.seed, aug_mode="none", fold_idx=cfg.fold, num_folds=cfg.num_folds
        )

    drop_last = len(train_ds) >= cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    optim = get_optimizer(model, cfg.optimizer_strategy, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(not cfg.no_amp and device == "cuda"))
    scheduler = build_scheduler(
        optim, cfg.epochs, max(1, len(train_loader)), cfg.warmup_epochs
    )

    # Deep supervision loss (paper Eq. 10)
    loss_fn = DeepSupervisionLoss(
        base_loss=ComboLoss(0.5, 0.5),
        aux_weights=[0.4, 0.3, 0.2],
    )

    print(f"Using dataset '{dataset_key}' from {cfg.data_dir}")
    print(f"Saving outputs to {cfg.save_dir}")
    print(f"Deep supervision: {cfg.deep_supervision}")

    # Initialize metrics tracking if enabled
    metrics_history = None
    if cfg.track_metrics:
        from .metrics_tracking import MetricsHistory
        metrics_history = MetricsHistory(cfg.save_dir)

    early_stopper = EarlyStopping(
        patience=cfg.patience, verbose=True,
        path=os.path.join(cfg.save_dir, "best.pt"),
    )
    best_val_metrics: Optional[Dict[str, float]] = None
    best_score = -float("inf")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model, train_loader, optim, scheduler, scaler, loss_fn, device,
            grad_clip=cfg.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        # Record metrics if tracking enabled
        if metrics_history:
            train_metrics = {
                "loss": tr_loss,
                "dice": tr_dice,
                "iou": tr_iou,
            }
            metrics_history.record_epoch(epoch, "train", train_metrics)
            metrics_history.record_epoch(epoch, "val", val_metrics)
            metrics_history.save_json(os.path.join(cfg.save_dir, "metrics_history.json"))

        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | time {dt:.1f}s | "
            f"train: loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} | "
            f"val: loss {val_metrics['loss']:.4f} mdice {val_metrics['mDice']:.4f} "
            f"miou {val_metrics['mIoU']:.4f} mae {val_metrics['mae']:.4f} "
            f"Fw {val_metrics['Fbeta_w']:.4f} Salpha {val_metrics['s_alpha']:.4f} "
            f"mE {val_metrics['mE']:.4f}/{val_metrics['mE_max']:.4f} "
            f"prec {val_metrics['precision']:.4f} rec {val_metrics['recall']:.4f} "
            f"acc {val_metrics['accuracy']:.4f}"
        )

        # Composite validation score (paper Eq. 16)
        score = (val_metrics["mDice"] + val_metrics["mIoU"]) / 2.0
        if score > best_score:
            best_score = score
            best_val_metrics = val_metrics
        early_stopper(score, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for evaluation...")
    ckpt = torch.load(os.path.join(cfg.save_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, device, use_tta=cfg.use_tta)
    print(
        f"Test with {'TTA' if cfg.use_tta else 'no TTA'} | "
        f"loss {test_metrics['loss']:.4f} mdice {test_metrics['mDice']:.4f} "
        f"miou {test_metrics['mIoU']:.4f} mae {test_metrics['mae']:.4f} "
        f"Fw {test_metrics['Fbeta_w']:.4f} Salpha {test_metrics['s_alpha']:.4f} "
        f"mE {test_metrics['mE']:.4f}/{test_metrics['mE_max']:.4f} "
        f"prec {test_metrics['precision']:.4f} rec {test_metrics['recall']:.4f} "
        f"acc {test_metrics['accuracy']:.4f}"
    )

    save_visuals(
        model, test_loader, device, os.path.join(cfg.save_dir, "vis_test"),
        train_ds.mean, train_ds.std, max_batches=4,
    )

    return {
        "best_val": best_val_metrics or {},
        "test": test_metrics,
    }
