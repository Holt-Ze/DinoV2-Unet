"""Data augmentation and preprocessing transforms.

Implements the augmentation strategy described in the paper (Section 3.6):
- Strong: flips, rotations, elastic/grid/optical distortions, color jitter.
- Weak: flips and rotations only.
- None: resize and normalize only (for validation/testing).

All images are normalized to ImageNet statistics (Eq. 12).
"""

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

if hasattr(Image, "Resampling"):
    PIL_BICUBIC = Image.Resampling.BICUBIC
else:
    PIL_BICUBIC = Image.BICUBIC

VALID_AUG_MODES = ("strong", "weak", "none")


def _normalize_aug_mode(mode: Optional[str]) -> str:
    """Validate and normalize augmentation mode string.

    Args:
        mode: Augmentation mode ('strong', 'weak', 'none').

    Returns:
        Normalized lowercase mode string.

    Raises:
        ValueError: If mode is not one of the valid options.
    """
    value = (mode or "strong").lower()
    if value not in VALID_AUG_MODES:
        raise ValueError(
            f"Unsupported aug_mode '{mode}'. Choose from "
            f"{', '.join(VALID_AUG_MODES)}."
        )
    return value


def build_polyp_transform(split: str, img_size: int, mean, std,
                           aug_mode: str = "strong"):
    """Build albumentations transform pipeline for polyp segmentation.

    Args:
        split: Dataset split ('train', 'val', 'test').
        img_size: Target image resolution.
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).
        aug_mode: Augmentation strength ('strong', 'weak', 'none').

    Returns:
        albumentations.Compose transform pipeline.
    """
    aug_mode = _normalize_aug_mode(aug_mode)
    resize = A.Resize(img_size, img_size, interpolation=PIL_BICUBIC)
    normalize = A.Normalize(mean=mean, std=std)

    if split != "train" or aug_mode == "none":
        return A.Compose([resize, normalize, ToTensorV2()])

    transforms = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    if aug_mode == "strong":
        transforms.append(
            A.OneOf(
                [
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, p=1),
                ],
                p=0.8,
            )
        )
        transforms.append(
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
            )
        )

    transforms.extend([normalize, ToTensorV2()])
    return A.Compose(transforms)


def denorm(x, mean, std):
    """De-normalize a tensor for visualization.

    Args:
        x: Normalized tensor of shape (B, C, H, W) or (C, H, W).
        mean: Normalization mean used during preprocessing.
        std: Normalization std used during preprocessing.

    Returns:
        De-normalized tensor with the same shape.
    """
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return x * std + mean
