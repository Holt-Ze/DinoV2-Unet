"""Loss functions for polyp segmentation.

This module implements the hybrid loss described in the paper (Section 3.5):
- **DiceLoss**: Region-based loss for spatial overlap optimization.
- **BCEWithLogitsLoss2D**: Pixel-wise Binary Cross-Entropy loss.
- **ComboLoss**: Hybrid BCE + Dice loss (Eq. 11).
- **DeepSupervisionLoss**: Weighted sum of main and auxiliary losses (Eq. 10).
"""

from typing import Dict

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Computes 1 - Dice coefficient using soft predictions (sigmoid of logits)
    to maintain differentiability. Operates on the spatial dimensions.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw model output of shape (B, 1, H, W).
            targets: Binary ground-truth masks of shape (B, 1, H, W).

        Returns:
            Scalar Dice loss value.
        """
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        dice = (num / den).mean()
        return 1 - dice


class BCEWithLogitsLoss2D(nn.Module):
    """Pixel-wise Binary Cross-Entropy loss with logits.

    Wraps PyTorch's BCEWithLogitsLoss for 2D segmentation masks,
    enforcing accurate per-pixel classification.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Raw model output of shape (B, 1, H, W).
            targets: Binary ground-truth masks of shape (B, 1, H, W).

        Returns:
            Scalar BCE loss value.
        """
        return self.loss(logits, targets)


class ComboLoss(nn.Module):
    """Hybrid BCE + Dice loss for polyp segmentation (paper Eq. 11).

    Combines pixel-wise BCE for accurate classification with region-based
    Dice loss for spatial overlap, addressing foreground-background imbalance
    inherent in colonoscopy images.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = BCEWithLogitsLoss2D()
        self.dice = DiceLoss()
        self.wb = bce_weight
        self.wd = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE + Dice loss.

        Args:
            logits: Raw model output of shape (B, 1, H, W).
            targets: Binary ground-truth masks of shape (B, 1, H, W).

        Returns:
            Scalar combined loss value.
        """
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)


class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss with weighted auxiliary terms (paper Eq. 10).

    Computes the total loss as:
        L_total = L_main + sum(lambda_i * L_aux_i)

    where lambda_i decays with depth to prioritize the final prediction.

    Args:
        base_loss: The loss function to apply (typically ComboLoss).
        aux_weights: List of weights for each auxiliary head loss.
            Defaults to [0.4, 0.3, 0.2] (decaying with depth).
    """

    def __init__(
        self,
        base_loss: nn.Module = None,
        aux_weights: list = None,
    ):
        super().__init__()
        self.base_loss = base_loss or ComboLoss(0.5, 0.5)
        self.aux_weights = aux_weights or [0.4, 0.3, 0.2]

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute deep supervision loss.

        Args:
            outputs: Dict from DinoV2UNet.forward() containing 'main' and
                optional 'aux_0', 'aux_1', 'aux_2' logits.
            targets: Binary ground-truth masks of shape (B, 1, H, W).

        Returns:
            Scalar total loss value.
        """
        # Main head loss
        total_loss = self.base_loss(outputs["main"], targets)

        # Auxiliary head losses (if present during training)
        for i, weight in enumerate(self.aux_weights):
            aux_key = f"aux_{i}"
            if aux_key in outputs:
                total_loss = total_loss + weight * self.base_loss(outputs[aux_key], targets)

        return total_loss
