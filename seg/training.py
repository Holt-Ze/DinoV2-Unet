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
from .losses import ComboLoss
from .metrics import compute_segmentation_metrics, dice_iou_from_logits
from .models import DinoV2UNet
from .transforms import denorm
from .utils import set_seed
from .profiling import describe_profile


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = -math.inf

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f}). Saving model ...')
        torch.save({'model': model.state_dict()}, self.path)
        self.val_score_min = val_score


@dataclass
class TrainConfig:
    dataset: str
    data_dir: str
    img_size: int = 448
    batch_size: int = 8
    epochs: int = 80
    backbone: str = 'vit_base_patch14_dinov2'
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
    subset_ratio: float = 1.0
    pretrained_type: str = 'dinov2'
    decoder_type: str = 'simple'
    optimizer_strategy: str = 'partial_finetune'


def get_optimizer(model: DinoV2UNet, strategy: str, cfg: TrainConfig):
    # Adjust requires_grad based on strategy
    if strategy == 'frozen_encoder':
        # Linear Probing: Freeze ALL encoder parameters
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == 'full_finetune':
        # Unfreeze ALL encoder parameters
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif strategy == 'partial_finetune':
        # Default: relying on model.__init__ logic which freezes blocks < freeze_blocks_until
        # But to be safe, we re-apply or just trust the init. 
        # The user said: "Default: freeze Block 0-5, train 6-11".
        # VitDinoV2Encoder.__init__ handles the blocks. 
        # But we should ensure consistency if someone changed it manually or if we want to be explicit.
        # Let's re-enforce it.
        # Note: We should handle non-block params? 
        # Implementation in models.py only touched blocks. 
        # If we follow "partial_finetune" strictly as "Freeze 0-5", we keep 6-11 open. 
        # We assume patch_embed/pos_embed are kept as initialized (likely True).
        # To strictly follow "Freeze 0-5, tune 6-11", we might want to ensure 0-5 are really frozen.
        # Let's iterate blocks again to be sure.
        if hasattr(model.encoder, 'model') and hasattr(model.encoder.model, 'blocks'):
            for i, blk in enumerate(model.encoder.model.blocks):
                requires = i >= cfg.freeze_blocks_until
                for p in blk.parameters():
                    p.requires_grad = requires
        # For non-block parameters (patch_embed, etc), we leave them as is (usually True) or freeze?
        # Standard partial finetuning usually freezes the "stem" too if blocks are frozen.
        # But let's respect the user's specific instruction "Freeze Block 0-5, train 6-11" 
        # and assume other parts stay as default.
        pass
    else:
        raise ValueError(f"Unknown optimizer strategy: {strategy}")

    # Always ensure decoder is trainable
    for p in model.decoder.parameters():
        p.requires_grad = True

    # Filter parameters
    enc_params = [p for n, p in model.named_parameters() if n.startswith('encoder') and p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if not n.startswith('encoder') and p.requires_grad]

    print(f"Optimizer Strategy: {strategy}")
    print(f"Trainable Encoder Params: {len(enc_params)}")
    print(f"Trainable Decoder Params: {len(dec_params)}")

    return torch.optim.AdamW([
        {'params': enc_params, 'lr': cfg.lr_backbone, 'name': 'enc'},
        {'params': dec_params, 'lr': cfg.lr, 'name': 'dec'},
    ], weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0):
    total_iters = max(1, epochs * steps_per_epoch)
    warmup_iters = warmup_epochs * steps_per_epoch

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        progress = (current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optim, scheduler, scaler, loss_fn, device):
    model.train()
    running_loss, running_dice, running_iou, iters = 0.0, 0.0, 0.0, 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device == 'cuda'
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits, msks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        scheduler.step()
        d, i = dice_iou_from_logits(logits, msks)
        running_loss += loss.item()
        running_dice += d
        running_iou += i
        iters += 1
    denom = max(iters, 1)
    return running_loss / denom, running_dice / denom, running_iou / denom


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    loss_fn = ComboLoss(0.5, 0.5)
    metrics_sum = {}
    total_loss = 0.0
    total_samples = 0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        if use_tta:
            logits_orig = model(imgs)
            logits_hf = model(TF.hflip(imgs))
            logits = (logits_orig + TF.hflip(logits_hf)) / 2.0
        else:
            logits = model(imgs)
        loss = loss_fn(logits, msks)
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
def save_visuals(model, loader, device, save_dir, mean, std, max_batches=2):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    cnt = 0
    for imgs, msks, names in loader:
        imgs_gpu = imgs.to(device)
        preds = (torch.sigmoid(model(imgs_gpu)) > 0.5).float().cpu()
        imgs_denorm = denorm(imgs, mean, std)
        for b in range(imgs.size(0)):
            base_name = os.path.splitext(os.path.basename(names[b]))[0]
            save_image(imgs_denorm[b], os.path.join(save_dir, f"{base_name}_img.png"))
            save_image(msks[b], os.path.join(save_dir, f"{base_name}_gt.png"))
            save_image(preds[b], os.path.join(save_dir, f"{base_name}_pred.png"))
        cnt += 1
        if cnt >= max_batches:
            break


def run_training(cfg: TrainConfig, dataset_key: str, spec: Optional[DatasetSpec] = None):
    if spec is None:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key '{dataset_key}'.")
        spec = DATASET_SPECS[dataset_key]

    set_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(cfg.save_dir, exist_ok=True)

    dataset_cls = spec.cls

    model = DinoV2UNet(cfg.backbone, cfg.out_indices, True, cfg.freeze_blocks_until, 1,
                       decoder_dropout=cfg.decoder_dropout,
                       pretrained_type=cfg.pretrained_type,
                       decoder_type=cfg.decoder_type).to(device)
    patch_size = getattr(model.encoder, 'patch_size', 1)
    if patch_size > 1 and (cfg.img_size % patch_size) != 0:
        new_size = int(math.ceil(cfg.img_size / patch_size) * patch_size)
        print(f"Requested img_size {cfg.img_size} is not divisible by encoder patch size {patch_size}. "
              f"Resizing to {new_size}.")
        cfg.img_size = new_size

    if cfg.profile:
        use_amp = (not cfg.no_amp and device == "cuda")
        profile_msg = describe_profile(model, (cfg.img_size, cfg.img_size), device, use_amp=use_amp)
        print(profile_msg)

    train_ds = dataset_cls(cfg.data_dir, 'train', cfg.img_size, seed=cfg.seed, aug_mode=cfg.aug_mode, subset_ratio=cfg.subset_ratio)
    val_ds = dataset_cls(cfg.data_dir, 'val', cfg.img_size, seed=cfg.seed, aug_mode='none')
    test_ds = dataset_cls(cfg.data_dir, 'test', cfg.img_size, seed=cfg.seed, aug_mode='none')

    drop_last = len(train_ds) >= cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    optim = get_optimizer(model, cfg.optimizer_strategy, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=(not cfg.no_amp and device == 'cuda'))
    scheduler = build_scheduler(optim, cfg.epochs, max(1, len(train_loader)), cfg.warmup_epochs)
    loss_fn = ComboLoss(0.5, 0.5)

    print(f"Using dataset '{dataset_key}' from {cfg.data_dir}")
    print(f"Saving outputs to {cfg.save_dir}")

    early_stopper = EarlyStopping(patience=cfg.patience, verbose=True, path=os.path.join(cfg.save_dir, 'best.pt'))
    best_val_metrics: Optional[Dict[str, float]] = None
    best_score = -float("inf")
    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = train_one_epoch(model, train_loader, optim, scheduler, scaler, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | time {dt:.1f}s | "
              f"train: loss {tr_loss:.4f} dice {tr_dice:.4f} iou {tr_iou:.4f} | "
              f"val: loss {val_metrics['loss']:.4f} mdice {val_metrics['mDice']:.4f} miou {val_metrics['mIoU']:.4f} "
              f"mae {val_metrics['mae']:.4f} Fw {val_metrics['Fbeta_w']:.4f} "
              f"Salpha {val_metrics['s_alpha']:.4f} mE {val_metrics['mE']:.4f}/{val_metrics['mE_max']:.4f} "
              f"prec {val_metrics['precision']:.4f} rec {val_metrics['recall']:.4f} acc {val_metrics['accuracy']:.4f}")

        score = (val_metrics['mDice'] + val_metrics['mIoU']) / 2.0
        if score > best_score:
            best_score = score
            best_val_metrics = val_metrics
        early_stopper(score, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for evaluation...")
    ckpt = torch.load(os.path.join(cfg.save_dir, 'best.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model'])

    test_metrics = evaluate(model, test_loader, device, use_tta=cfg.use_tta)
    print(f"Test with {'TTA' if cfg.use_tta else 'no TTA'} | loss {test_metrics['loss']:.4f} "
          f"mdice {test_metrics['mDice']:.4f} "
          f"miou {test_metrics['mIoU']:.4f} mae {test_metrics['mae']:.4f} Fw {test_metrics['Fbeta_w']:.4f} "
          f"Salpha {test_metrics['s_alpha']:.4f} mE {test_metrics['mE']:.4f}/{test_metrics['mE_max']:.4f} "
          f"prec {test_metrics['precision']:.4f} rec {test_metrics['recall']:.4f} acc {test_metrics['accuracy']:.4f}")

    save_visuals(model, test_loader, device, os.path.join(cfg.save_dir, 'vis_test'),
                 train_ds.mean, train_ds.std, max_batches=4)
    return {
        "best_val": best_val_metrics or {},
        "test": test_metrics,
    }

