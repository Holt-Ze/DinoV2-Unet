"""Segmentation metrics for polyp evaluation.

Implements the evaluation metrics described in the paper (Section 4.2):
- Dice coefficient and IoU (Eq. 17)
- Mean Absolute Error (MAE, Eq. 18)
- Weighted F-measure (Fw_beta)
- Structure measure (S_alpha, Eq. 19)
- Enhanced alignment measure (E_xi, Eq. 20)
- Precision, Recall, Accuracy
"""

from typing import Dict, Tuple

import torch


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                         thr: float = 0.5, eps: float = 1e-6):
    """Compute batch-averaged Dice and IoU from raw logits.

    Args:
        logits: Raw model output of shape (B, 1, H, W).
        targets: Binary ground-truth masks of shape (B, 1, H, W).
        thr: Binarization threshold for predictions.
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (mean_dice, mean_iou) as float values.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    dice = (2 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def _object_score(values: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute object-level similarity score for S_alpha."""
    if values.numel() == 0:
        return 0.0
    mean_val = values.mean()
    std_val = values.std(unbiased=False)
    return ((2 * mean_val + eps) / (mean_val.pow(2) + 1 + std_val + eps)).item()


def _s_object(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute object-level structure similarity (S_o)."""
    fg_mask = gt >= 0.5
    w_fg = fg_mask.float().mean().item()
    w_bg = 1.0 - w_fg
    fg_score = _object_score(pred[fg_mask], eps)
    bg_score = _object_score(1.0 - pred[~fg_mask], eps)
    return w_fg * fg_score + w_bg * bg_score


def _centroid(gt: torch.Tensor) -> Tuple[int, int]:
    """Compute the centroid of the ground-truth mask."""
    h, w = gt.shape
    total = gt.sum()
    if total <= 0:
        return h // 2, w // 2
    rows = torch.arange(h, device=gt.device, dtype=gt.dtype)
    cols = torch.arange(w, device=gt.device, dtype=gt.dtype)
    cy = torch.round((gt.sum(dim=1) * rows).sum() / (total + 1e-6)).item()
    cx = torch.round((gt.sum(dim=0) * cols).sum() / (total + 1e-6)).item()
    cy = int(max(min(cy, h - 1), 0))
    cx = int(max(min(cx, w - 1), 0))
    return cy, cx


def _ssim(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute structural similarity for a single region."""
    if pred.numel() == 0 or gt.numel() == 0:
        return 0.0
    mean_x = pred.mean()
    mean_y = gt.mean()
    var_x = pred.var(unbiased=False)
    var_y = gt.var(unbiased=False)
    cov_xy = ((pred - mean_x) * (gt - mean_y)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / (
        (mean_x.pow(2) + mean_y.pow(2) + c1) * (var_x + var_y + c2) + eps
    )
    return score.clamp(0.0, 1.0).item()


def _s_region(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute region-level structure similarity (S_r)."""
    h, w = gt.shape
    if h == 0 or w == 0:
        return 0.0
    cy, cx = _centroid(gt)
    if h > 1:
        cy = min(max(cy, 1), h - 1)
    else:
        cy = 0
    if w > 1:
        cx = min(max(cx, 1), w - 1)
    else:
        cx = 0
    regions = [
        (pred[:cy, :cx], gt[:cy, :cx], float(cy * cx)),
        (pred[:cy, cx:], gt[:cy, cx:], float(cy * (w - cx))),
        (pred[cy:, :cx], gt[cy:, :cx], float((h - cy) * cx)),
        (pred[cy:, cx:], gt[cy:, cx:], float((h - cy) * (w - cx))),
    ]
    area = float(h * w) + eps
    score = 0.0
    for pred_reg, gt_reg, weight in regions:
        if weight <= 0:
            continue
        score += (weight / area) * _ssim(pred_reg, gt_reg, eps)
    return score


def structure_measure_map(pred: torch.Tensor, gt: torch.Tensor,
                          eps: float = 1e-6) -> float:
    """Compute structure measure S_alpha for a single prediction-GT pair (Eq. 19).

    Args:
        pred: Predicted probability map (H, W).
        gt: Ground-truth binary mask (H, W).
        eps: Numerical stability constant.

    Returns:
        S_alpha value in [0, 1].
    """
    total = gt.sum()
    numel = gt.numel()
    if total <= 0:
        return float(1.0 - pred.mean().item())
    if total >= numel:
        return float(pred.mean().item())
    s_obj = _s_object(pred, gt, eps)
    s_reg = _s_region(pred, gt, eps)
    alpha = 0.5
    return alpha * s_obj + (1 - alpha) * s_reg


def _enhanced_alignment(bin_pred: torch.Tensor, gt: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """Compute enhanced alignment score for a single threshold."""
    mean_pred = bin_pred.mean()
    mean_gt = gt.mean()
    align = (2 * (bin_pred - mean_pred) * (gt - mean_gt) + eps) / (
        (bin_pred - mean_pred).pow(2) + (gt - mean_gt).pow(2) + eps
    )
    enhanced = ((align + 1) ** 2) / 4.0
    return enhanced.mean()


def enhanced_measure_map(pred: torch.Tensor, gt: torch.Tensor,
                          num_steps: int = 41,
                          eps: float = 1e-6) -> Tuple[float, float]:
    """Compute enhanced alignment measure E_xi (Eq. 20).

    Returns both mean and max E_xi across multiple thresholds.

    Args:
        pred: Predicted probability map (H, W).
        gt: Ground-truth binary mask (H, W).
        num_steps: Number of threshold steps.
        eps: Numerical stability constant.

    Returns:
        Tuple of (mean_E_xi, max_E_xi).
    """
    total = gt.sum()
    numel = gt.numel()
    if total <= 0:
        score = float(1.0 - pred.mean().item())
        return score, score
    if total >= numel:
        score = float(pred.mean().item())
        return score, score
    thresholds = torch.linspace(0.0, 1.0, steps=max(num_steps, 2), device=pred.device)
    scores = []
    for thr in thresholds:
        bin_pred = (pred >= thr).float()
        scores.append(_enhanced_alignment(bin_pred, gt, eps))
    stacked = torch.stack(scores)
    return stacked.mean().item(), stacked.max().item()


def compute_segmentation_metrics(
    logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5,
    beta_sq: float = 0.3 ** 2, e_steps: int = 41, eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute all segmentation evaluation metrics.

    Implements the full evaluation protocol from Section 4.2 of the paper.

    Args:
        logits: Raw model output of shape (B, C, H, W).
        targets: Binary ground-truth masks of shape (B, C, H, W).
        thr: Binarization threshold.
        beta_sq: Squared beta for weighted F-measure.
        e_steps: Number of threshold steps for E_xi computation.
        eps: Numerical stability constant.

    Returns:
        Dict containing: mDice, mIoU, mae, Fbeta_w, s_alpha, mE, mE_max,
        precision, recall, accuracy.
    """
    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    preds = (probs > thr).float()
    dims = tuple(range(2, targets.ndim))

    inter = (preds * targets).sum(dim=dims)
    preds_sum = preds.sum(dim=dims)
    targets_sum = targets.sum(dim=dims)
    union = preds_sum + targets_sum - inter

    dice = (2 * inter + eps) / (preds_sum + targets_sum + eps)
    iou = (inter + eps) / (union + eps)

    tp = inter
    fp = (preds * (1 - targets)).sum(dim=dims)
    fn = ((1 - preds) * targets).sum(dim=dims)
    tn = ((1 - preds) * (1 - targets)).sum(dim=dims)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    mae = torch.abs(probs - targets).mean(dim=dims)

    # Weighted F-measure
    tp_w = (probs * targets).sum(dim=dims)
    fp_w = (probs * (1 - targets)).sum(dim=dims)
    fn_w = ((1 - probs) * targets).sum(dim=dims)
    precision_w = (tp_w + eps) / (tp_w + fp_w + eps)
    recall_w = (tp_w + eps) / (tp_w + fn_w + eps)
    fbeta_w = (1 + beta_sq) * precision_w * recall_w / (
        beta_sq * precision_w + recall_w + eps
    )

    # Per-sample structure and alignment measures
    s_scores, e_means, e_maxes = [], [], []
    bsz = probs.size(0)
    for b in range(bsz):
        pred_map = probs[b, 0] if probs.size(1) == 1 else probs[b].mean(dim=0)
        gt_map = targets[b, 0] if targets.size(1) == 1 else targets[b].mean(dim=0)
        s_scores.append(structure_measure_map(pred_map, gt_map, eps))
        e_mean, e_max = enhanced_measure_map(pred_map, gt_map,
                                              num_steps=e_steps, eps=eps)
        e_means.append(e_mean)
        e_maxes.append(e_max)

    denom = max(len(s_scores), 1)
    metrics = {
        "mDice": dice.mean().item(),
        "mIoU": iou.mean().item(),
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "mae": mae.mean().item(),
        "Fbeta_w": fbeta_w.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "accuracy": accuracy.mean().item(),
        "s_alpha": float(sum(s_scores) / denom),
        "mE": float(sum(e_means) / denom),
        "mE_max": float(sum(e_maxes) / denom),
    }
    return metrics
