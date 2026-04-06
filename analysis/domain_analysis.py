"""Cross-domain analysis for understanding transfer success.

Analyzes feature distributions, computes domain gaps, and predicts
zero-shot transfer success.
"""

from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance


class DomainAnalyzer:
    """Analyze feature distributions and domain gaps."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture intermediate features."""
        # Hook into encoder blocks.
        # DinoV2UNet uses `model.encoder.model.blocks`; keep a fallback for other layouts.
        if not hasattr(self.model, "encoder"):
            return

        encoder = self.model.encoder
        blocks = None
        if hasattr(encoder, "model") and hasattr(encoder.model, "blocks"):
            blocks = encoder.model.blocks
        elif hasattr(encoder, "blocks"):
            blocks = encoder.blocks

        if not blocks:
            return

        def hook_fn(module, input, output):
            # output is expected to be (B, N, D)
            self.activations["features"] = output.detach()

        blocks[-1].register_forward_hook(hook_fn)

    def extract_intermediate_features(self, loader: DataLoader,
                                     layer_idx: int = 11) -> np.ndarray:
        """Extract intermediate features from model.

        Args:
            loader: Data loader.
            layer_idx: Encoder block index to extract from (11 = last block).

        Returns:
            Feature array of shape (N, D) or (N, C*H*W) if spatial dims exist.
        """
        self.model.eval()
        features_list = []

        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(self.device)
                _ = self.model(imgs)  # Forward pass triggers hooks

                if 'features' in self.activations:
                    feats = self.activations['features'].cpu().numpy()
                    # Shape: (B, N, D) -> take mean across spatial (N) -> (B, D)
                    if len(feats.shape) == 3 and feats.shape[1] > 1:
                        feats = feats.mean(axis=1)
                    features_list.append(feats)

        if not features_list:
            raise RuntimeError("No features captured from model hooks.")

        return np.vstack(features_list)

    def compute_domain_gap(self, features_dict: Dict[str, np.ndarray]) \
            -> pd.DataFrame:
        """Compute pairwise domain gaps using Wasserstein distance.

        Args:
            features_dict: Dict mapping dataset_name -> feature_array (N, D).

        Returns:
            DataFrame with columns: dataset1, dataset2, wasserstein, mmd.
        """
        dataset_names = list(features_dict.keys())
        gaps = []

        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i >= j:
                    continue

                feats1 = features_dict[name1].reshape(len(features_dict[name1]), -1)
                feats2 = features_dict[name2].reshape(len(features_dict[name2]), -1)

                try:
                    # Use first dimension for efficiency
                    wasserstein = wasserstein_distance(feats1[:, 0], feats2[:, 0])
                except Exception:
                    wasserstein = np.nan

                # Compute MMD (simplified)
                mmd = self._compute_mmd(feats1[:100], feats2[:100])

                gaps.append({
                    "dataset1": name1,
                    "dataset2": name2,
                    "wasserstein": wasserstein,
                    "mmd": mmd,
                })

        return pd.DataFrame(gaps)

    @staticmethod
    def _compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy (simplified).

        Args:
            X: Features from domain X, shape (N, D).
            Y: Features from domain Y, shape (M, D).
            sigma: Kernel bandwidth.

        Returns:
            MMD^2 value.
        """
        def gaussian_kernel(z: float) -> float:
            return np.exp(-z / (2 * sigma ** 2))

        # Subsample for efficiency
        n = min(50, len(X))
        m = min(50, len(Y))
        X = X[:n]
        Y = Y[:m]

        # Compute kernel matrix elements
        XX = np.mean([gaussian_kernel(np.linalg.norm(X[i] - X[j]))
                     for i in range(len(X)) for j in range(len(X))])
        YY = np.mean([gaussian_kernel(np.linalg.norm(Y[i] - Y[j]))
                     for i in range(len(Y)) for j in range(len(Y))])
        XY = np.mean([gaussian_kernel(np.linalg.norm(X[i] - Y[j]))
                     for i in range(len(X)) for j in range(len(Y))])

        mmd_sq = XX + YY - 2 * XY
        return max(0, mmd_sq)

    def predict_zero_shot_success(self,
                                 source_features: Dict[str, np.ndarray],
                                 target_features: Dict[str, np.ndarray],
                                 source_metrics: Dict[str, float]) -> Dict:
        """Predict zero-shot transfer success based on domain gap.

        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            source_metrics: Source domain test metrics (e.g., mDice).

        Returns:
            Dict with predictions and insights.
        """
        predictions = {}

        for target_name, target_feats in target_features.items():
            min_gap = float('inf')
            closest_source = None

            for source_name, source_feats in source_features.items():
                # Compute gap
                source_flat = source_feats.reshape(len(source_feats), -1)
                target_flat = target_feats.reshape(len(target_feats), -1)

                gap = wasserstein_distance(source_flat[:, 0], target_flat[:, 0])

                if gap < min_gap:
                    min_gap = gap
                    closest_source = source_name

            # Predict performance based on gap
            source_perf = source_metrics.get(closest_source, 0.8)
            # Simple linear model: performance decreases with gap
            predicted_perf = max(0.3, source_perf - 0.2 * min_gap)

            # Difficulty prediction
            if min_gap < 0.3:
                difficulty = "easy"
            elif min_gap < 0.6:
                difficulty = "medium"
            else:
                difficulty = "hard"

            predictions[target_name] = {
                "source_model": closest_source,
                "domain_gap": min_gap,
                "expected_mdice": predicted_perf,
                "transfer_difficulty": difficulty,
            }

        return predictions


class DomainVisualizer:
    """Visualize cross-domain analysis results."""

    @staticmethod
    def save_feature_distributions(features_dict: Dict[str, np.ndarray],
                                  save_dir: str) -> None:
        """Save histograms of feature distributions per dataset.

        Args:
            features_dict: Dict mapping dataset_name -> features.
            save_dir: Output directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        for dataset_name, features in features_dict.items():
            features_flat = features.reshape(len(features), -1)
            mean_feat = features_flat.mean(axis=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(mean_feat, bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Feature Value", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"Feature Distribution: {dataset_name}",
                        fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            save_path = os.path.join(save_dir, f"feature_dist_{dataset_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {save_path}")

    @staticmethod
    def save_transfer_summary(predictions: Dict, save_path: str) -> None:
        """Save transfer success predictions as a summary table.

        Args:
            predictions: Predictions from predict_zero_shot_success.
            save_path: Output file path.
        """
        rows = []
        for target, pred in predictions.items():
            rows.append({
                "target_dataset": target,
                "source_model": pred["source_model"],
                "domain_gap": pred["domain_gap"],
                "predicted_mDice": pred["expected_mdice"],
                "difficulty": pred["transfer_difficulty"],
            })

        df = pd.DataFrame(rows)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

        # Print summary
        print("\nZero-Shot Transfer Predictions:")
        print(df.to_string(index=False))
