"""Failure analysis tools for understanding model errors.

Identifies hard examples, categorizes failures, and analyzes
confidence/calibration issues.
"""

from typing import Dict, List, Tuple, Optional
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from PIL import Image


class FailureAnalyzer:
    """Mine hard examples and categorize failure modes."""

    def __init__(self, model: torch.nn.Module, dataset, device: str = "cuda"):
        self.model = model
        self.dataset = dataset
        self.device = device

    def identify_hard_examples(self, metric: str = "mDice", percentile: int = 10,
                              batch_size: int = 8) -> List[Dict]:
        """Identify worst-performing examples.

        Args:
            metric: Metric to use for ranking ('mDice', 'mIoU', 'mae').
            percentile: Return bottom-P% worst examples.
            batch_size: Evaluation batch size.

        Returns:
            List of dicts with keys: idx, image_name, metric_value, pred, gt, confidence.
        """
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0)

        all_metrics = {}
        confidences = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                imgs, msks, names = batch
                imgs, msks = imgs.to(self.device), msks.to(self.device)

                outputs = self.model(imgs)
                if isinstance(outputs, dict):
                    logits = outputs["main"]
                else:
                    logits = outputs

                # Compute metrics per sample
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                for b in range(len(imgs)):
                    sample_idx = batch_idx * batch_size + b
                    pred = preds[b].squeeze().cpu().numpy()
                    gt = msks[b].squeeze().cpu().numpy()
                    conf = probs[b].max().item()

                    # Compute metric
                    if metric == "mDice":
                        m = self._dice_score(pred, gt)
                    elif metric == "mIoU":
                        m = self._iou_score(pred, gt)
                    elif metric == "mae":
                        m = np.abs(pred - gt).mean()
                        m = 1 - m  # Higher is better for MAE
                    else:
                        m = 0.0

                    all_metrics[sample_idx] = m
                    confidences[sample_idx] = conf

        # Find worst percentile
        threshold_idx = int(len(all_metrics) * percentile / 100)
        worst_indices = sorted(all_metrics.keys(),
                              key=lambda k: all_metrics[k])[:threshold_idx]

        # Retrieve details for worst examples
        hard_examples = []
        for idx in worst_indices:
            imgs_test, msks_test, names_test = self.dataset[idx]
            hard_examples.append({
                "idx": idx,
                "image_name": names_test if isinstance(names_test, str)
                            else f"sample_{idx:05d}",
                metric: all_metrics[idx],
                "confidence": confidences[idx],
            })

        return hard_examples

    def categorize_failures(self, hard_examples: List[Dict],
                           batch_size: int = 8) -> Dict[str, List]:
        """Categorize failures into semantic categories.

        Categories:
        - small_polyp: area < 5%
        - large_polyp: area > 50%
        - shadow: low contrast
        - bleeding: high saturation
        - unclear_boundary: high gradient

        Args:
            hard_examples: List of hard examples from identify_hard_examples.
            batch_size: Batch size for processing.

        Returns:
            Dict mapping category -> list of examples in that category.
        """
        categorized = {
            "small_polyp": [],
            "large_polyp": [],
            "shadow": [],
            "bleeding": [],
            "unclear_boundary": [],
            "other": [],
        }

        for example in hard_examples:
            idx = example["idx"]
            imgs, msks, _ = self.dataset[idx]

            # Convert to numpy
            if isinstance(msks, torch.Tensor):
                gt = msks.squeeze().numpy()
            else:
                gt = np.array(msks)

            img_np = imgs.numpy().transpose(1, 2, 0)  # CHW -> HWC (normalized)
            mean = np.array(
                getattr(self.dataset, "mean", (0.485, 0.456, 0.406)), dtype=np.float32
            ).reshape(1, 1, 3)
            std = np.array(
                getattr(self.dataset, "std", (0.229, 0.224, 0.225)), dtype=np.float32
            ).reshape(1, 1, 3)
            img_denorm = (img_np * std + mean) * 255.0
            img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)

            # Compute features
            area_ratio = gt.mean()
            im_hsv = cv2.cvtColor(img_denorm, cv2.COLOR_RGB2HSV)
            saturation = im_hsv[:, :, 1].astype(float).mean() / 255.0
            laplacian = cv2.Laplacian((gt * 255).astype(np.uint8), cv2.CV_64F)
            gradient = np.abs(laplacian).mean()
            contrast = img_denorm.std()

            # Categorize
            category = "other"
            if area_ratio < 0.05:
                category = "small_polyp"
            elif area_ratio > 0.50:
                category = "large_polyp"
            elif contrast < 50:
                category = "shadow"
            elif saturation > 0.4:
                category = "bleeding"
            elif gradient > 5.0:
                category = "unclear_boundary"

            categorized[category].append(example)

        return categorized

    def confidence_analysis(self, batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze confidence vs accuracy (calibration).

        Returns:
            Tuple of (confidences, accuracies).
        """
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0)

        confidences = []
        accuracies = []

        with torch.no_grad():
            for imgs, msks, _ in loader:
                imgs, msks = imgs.to(self.device), msks.to(self.device)

                outputs = self.model(imgs)
                if isinstance(outputs, dict):
                    logits = outputs["main"]
                else:
                    logits = outputs

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                for b in range(len(imgs)):
                    conf = probs[b].max().item()
                    acc = self._dice_score(
                        preds[b].squeeze().cpu().numpy(),
                        msks[b].squeeze().cpu().numpy()
                    )

                    confidences.append(conf)
                    accuracies.append(acc)

        return np.array(confidences), np.array(accuracies)

    @staticmethod
    def _dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient."""
        intersection = np.sum(pred * gt)
        return 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-6)

    @staticmethod
    def _iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute IoU."""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection
        return intersection / (union + 1e-6)


class FailureVisualizer:
    """Visualize failure cases."""

    @staticmethod
    def save_failure_montages(dataset, hard_examples: List[Dict],
                             category_dict: Dict[str, List],
                             save_dir: str, max_per_category: int = 10) -> None:
        """Save montages of failure cases organized by category.

        Args:
            dataset: The dataset to retrieve images from.
            hard_examples: List of hard examples.
            category_dict: Categorized examples from categorize_failures.
            save_dir: Output directory.
            max_per_category: Maximum examples per category to save.
        """
        os.makedirs(save_dir, exist_ok=True)

        for category, examples in category_dict.items():
            if not examples:
                continue

            category_dir = os.path.join(save_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            for cidx, example in enumerate(examples[:max_per_category]):
                idx = example["idx"]
                imgs, msks, name = dataset[idx]

                # Denormalize image
                if isinstance(imgs, torch.Tensor):
                    mean = np.array(
                        getattr(dataset, "mean", (0.485, 0.456, 0.406))
                    )
                    std = np.array(
                        getattr(dataset, "std", (0.229, 0.224, 0.225))
                    )
                    img_denorm = imgs.numpy().transpose(1, 2, 0)
                    img_denorm = (img_denorm * std + mean) * 255
                    img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
                else:
                    img_denorm = imgs

                # Get prediction
                model = None  # Would need to pass model
                if isinstance(msks, torch.Tensor):
                    gt = (msks.squeeze().numpy() * 255).astype(np.uint8)
                else:
                    gt = (np.array(msks) * 255).astype(np.uint8)

                # Save component images
                base_name = f"{cidx:03d}"
                cv2.imwrite(
                    os.path.join(category_dir, f"{base_name}_input.png"),
                    cv2.cvtColor(img_denorm, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(category_dir, f"{base_name}_gt.png"),
                    gt
                )

    @staticmethod
    def plot_confidence_distribution(confidences: np.ndarray,
                                    accuracies: np.ndarray,
                                    save_path: str) -> None:
        """Plot confidence vs accuracy calibration curve.

        Args:
            confidences: Array of model confidences.
            accuracies: Array of actual accuracies (Dice scores).
            save_path: Output file path.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for confidence plotting.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(confidences, accuracies, alpha=0.5, s=20, c="steelblue")

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Calibration")

        ax.set_xlabel("Model Confidence", fontsize=11)
        ax.set_ylabel("Accuracy (Dice Score)", fontsize=11)
        ax.set_title("Confidence Calibration", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
