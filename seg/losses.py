import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        dice = (num / den).mean()
        return 1 - dice


class BCEWithLogitsLoss2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.loss(logits, targets)


class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = BCEWithLogitsLoss2D()
        self.dice = DiceLoss()
        self.wb = bce_weight
        self.wd = dice_weight

    def forward(self, logits, targets):
        return self.wb * self.bce(logits, targets) + self.wd * self.dice(logits, targets)

