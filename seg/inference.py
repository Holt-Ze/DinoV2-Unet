"""Inference utilities for exporting predicted segmentation masks.

Provides functionality to load a trained DINOv2-UNet checkpoint and
generate binary prediction masks for specified dataset splits.
"""

import os
from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import DATASET_SPECS, DatasetSpec
from .models import DinoV2UNet


@torch.no_grad()
def export_dataset_masks(
    dataset_key: str,
    data_dir: str,
    save_dir: str,
    backbone: str,
    out_indices: Tuple[int, ...],
    img_size: int,
    freeze_blocks_until: int,
    decoder_dropout: float,
    num_workers: int,
    splits: Iterable[str] = ("test",),
    checkpoint_path: Optional[str] = None,
    threshold: float = 0.5,
    device: Optional[str] = None,
    seed: int = 42,
    spec: Optional[DatasetSpec] = None,
):
    """Export predicted segmentation masks for a dataset.

    Loads a trained model checkpoint and generates binary prediction masks
    for the specified dataset splits, saving them as PNG images.

    Args:
        dataset_key: Dataset identifier (e.g., 'kvasir').
        data_dir: Path to the dataset directory.
        save_dir: Output directory for predicted masks.
        backbone: ViT backbone model name.
        out_indices: Transformer block indices for feature extraction.
        img_size: Input image resolution.
        freeze_blocks_until: Number of frozen encoder blocks.
        decoder_dropout: Decoder dropout probability.
        num_workers: DataLoader worker processes.
        splits: Dataset splits to export ('train', 'val', 'test').
        checkpoint_path: Path to the model checkpoint file.
        threshold: Binarization threshold for predictions.
        device: Target device (auto-detected if None).
        seed: Random seed for reproducibility.
        spec: Dataset specification (auto-resolved if None).
    """
    if spec is None:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset key '{dataset_key}'.")
        spec = DATASET_SPECS[dataset_key]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DinoV2UNet(
        backbone=backbone,
        out_indices=out_indices,
        pretrained=True,
        freeze_blocks_until=freeze_blocks_until,
        num_classes=1,
        decoder_dropout=decoder_dropout,
        deep_supervision=False,  # No aux heads needed for inference
    ).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if isinstance(state, dict):
            state = {
                k: v for k, v in state.items()
                if not (k.endswith("total_ops") or k.endswith("total_params"))
            }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[warn] Unexpected keys in checkpoint: {unexpected}")
        if missing:
            print(f"[warn] Missing keys in checkpoint: {missing}")

    model.eval()
    for split in splits:
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        dataset = spec.cls(data_dir, split, img_size, seed=seed, aug_mode="none")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        for imgs, _, names in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if isinstance(outputs, dict):
                logits = outputs["main"]
            else:
                logits = outputs
            preds = (torch.sigmoid(logits) > threshold).float().cpu()
            for b in range(preds.size(0)):
                base_name = os.path.splitext(os.path.basename(names[b]))[0]
                save_image(preds[b], os.path.join(split_dir, f"{base_name}.png"))
