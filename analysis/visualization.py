"""Visualization utilities for experiment analysis.

Provides classes for plotting training curves, ablation studies,
and cross-domain analysis results.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE


class TrainingVisualizer:
    """Generate visualizations for training progress and dynamics."""

    @staticmethod
    def plot_loss_curves(metrics_dict: Dict, save_path: str,
                        title: str = "Training Progress") -> None:
        """Create a two-panel plot of training and validation losses.

        Args:
            metrics_dict: Dictionary with structure {'metrics': {epoch: {phase: {...}}}}
                         or direct {'0': {'train': {...}, 'val': {...}}, ...}
            save_path: Output file path.
            title: Plot title.
        """
        # Handle both dict formats
        metrics = metrics_dict
        if "metrics" in metrics_dict:
            metrics = metrics_dict["metrics"]

        # Extract loss values
        epochs = sorted([int(e) for e in metrics.keys()])
        train_loss = []
        val_loss = []
        train_score = []
        val_score = []

        for e in epochs:
            if "train" in metrics[str(e)] and "loss" in metrics[str(e)]["train"]:
                train_loss.append(metrics[str(e)]["train"]["loss"])
            if "val" in metrics[str(e)] and "loss" in metrics[str(e)]["val"]:
                val_loss.append(metrics[str(e)]["val"]["loss"])

            # Composite score (mDice + mIoU) / 2
            if "val" in metrics[str(e)]:
                val_m = metrics[str(e)]["val"]
                if "mDice" in val_m and "mIoU" in val_m:
                    val_score.append((val_m["mDice"] + val_m["mIoU"]) / 2.0)

            if "train" in metrics[str(e)]:
                train_m = metrics[str(e)]["train"]
                if "dice" in train_m and "iou" in train_m:
                    train_score.append((train_m["dice"] + train_m["iou"]) / 2.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(epochs[:len(train_loss)], train_loss, "o-", label="Train",
                    linewidth=2, markersize=4)
        axes[0].plot(epochs[:len(val_loss)], val_loss, "s-", label="Val",
                    linewidth=2, markersize=4)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].set_title("Loss Evolution", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Composite score plot
        if train_score and val_score:
            axes[1].plot(epochs[:len(train_score)], train_score, "o-", label="Train",
                        linewidth=2, markersize=4)
            axes[1].plot(epochs[:len(val_score)], val_score, "s-", label="Val",
                        linewidth=2, markersize=4)
            axes[1].set_xlabel("Epoch", fontsize=11)
            axes[1].set_ylabel("Composite Score (mDice + mIoU)/2", fontsize=11)
            axes[1].set_title("Validation Score Evolution", fontsize=12, fontweight="bold")
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    @staticmethod
    def plot_metric_evolution(metrics_dict: Dict, metric_names: List[str],
                             save_path: str) -> None:
        """Create subplots for individual metric evolution.

        Args:
            metrics_dict: Metrics dictionary.
            metric_names: List of metric names to plot.
            save_path: Output file path.
        """
        # Handle both dict formats
        metrics = metrics_dict
        if "metrics" in metrics_dict:
            metrics = metrics_dict["metrics"]

        epochs = sorted([int(e) for e in metrics.keys()])
        num_metrics = len(metric_names)
        fig, axes = plt.subplots((num_metrics + 1) // 2, min(2, num_metrics),
                                figsize=(14, 5 * ((num_metrics + 1) // 2)))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            val_values = []
            val_epochs = []

            for e in epochs:
                if "val" in metrics[str(e)]:
                    val_m = metrics[str(e)]["val"]
                    # Look for both exact match and case variations
                    key = None
                    for k in val_m.keys():
                        if k.lower() == metric_name.lower():
                            key = k
                            break
                    if key:
                        val_values.append(val_m[key])
                        val_epochs.append(e)

            if val_values:
                ax.plot(val_epochs, val_values, "s-", linewidth=2, markersize=5,
                       color="steelblue")
                ax.fill_between(val_epochs, val_values, alpha=0.2, color="steelblue")
                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel(metric_name, fontsize=10)
                ax.set_title(f"{metric_name} Evolution", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Metric Evolution (Validation)", fontsize=13, fontweight="bold",
                    y=0.995)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    @staticmethod
    def plot_lr_schedule(warmup_epochs: int, max_epochs: int,
                        save_path: str, lr_base: float = 1.0) -> None:
        """Visualize cosine annealing with linear warm-up schedule.

        Args:
            warmup_epochs: Number of warm-up epochs.
            max_epochs: Total number of epochs.
            save_path: Output file path.
            lr_base: Base learning rate (for visualization, use normalized 1.0).
        """
        steps_per_epoch = 100  # Arbitrary for visualization
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        lrs = []
        for step in range(total_steps):
            if step < warmup_steps:
                # Linear warmup
                lr = lr_base * (step / warmup_steps)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = lr_base * 0.5 * (1 + np.cos(np.pi * progress))
                # Add minimum floor
                lr = max(lr, lr_base * 0.01)
            lrs.append(lr)

        epochs_axis = np.linspace(0, max_epochs, len(lrs))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(epochs_axis, lrs, linewidth=2.5, color="darkblue")
        ax.axvline(warmup_epochs, color="red", linestyle="--", linewidth=2,
                  label=f"Warmup End (epoch {warmup_epochs})")
        ax.fill_between(epochs_axis[:warmup_steps], 0, max(lrs) * 1.1, alpha=0.1,
                       color="orange", label="Warmup Phase")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Learning Rate (normalized)", fontsize=11)
        ax.set_title("Learning Rate Schedule: Linear Warmup + Cosine Annealing",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    @staticmethod
    def plot_gradient_flow(grad_history: List[Dict], save_path: str) -> None:
        """Visualize gradient norm evolution as a heatmap.

        Args:
            grad_history: List of gradient snapshots from training.
            save_path: Output file path.
        """
        if not grad_history or not grad_history[0]:
            print("No gradient history to plot.")
            return

        # Extract layer names and steps
        steps = [g.get("step", i) for i, g in enumerate(grad_history)]
        layer_names = [k for k in grad_history[0].keys() if k != "step"]

        # Build heatmap data
        data = np.zeros((len(layer_names), len(grad_history)))
        for step_idx, grad_snap in enumerate(grad_history):
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name in grad_snap and isinstance(grad_snap[layer_name], dict):
                    data[layer_idx, step_idx] = grad_snap[layer_name].get("norm", 0.0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")

        # Labels
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title("Gradient Flow: Norm Evolution Across Layers",
                    fontsize=12, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label="Gradient Norm")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


class AblationVisualizer:
    """Generate visualizations for ablation study results."""

    @staticmethod
    def plot_heatmap_2d(results_df: pd.DataFrame, param1: str, param2: str,
                       metric: str, save_path: str) -> None:
        """Create a 2D heatmap for ablation results.

        Args:
            results_df: DataFrame of ablation results with columns:
                       param1, param2, metric, seed (optional).
            param1: X-axis parameter name.
            param2: Y-axis parameter name.
            metric: Metric to visualize (heatmap color).
            save_path: Output file path.
        """
        # Aggregate by averaging over seeds
        grouped = results_df.groupby([param1, param2])[metric].mean().unstack()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grouped.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(grouped.columns)))
        ax.set_yticks(range(len(grouped.index)))
        ax.set_xticklabels(grouped.columns, fontsize=10)
        ax.set_yticklabels(grouped.index, fontsize=10)
        ax.set_xlabel(param1, fontsize=11)
        ax.set_ylabel(param2, fontsize=11)
        ax.set_title(f"Ablation: {param1} × {param2} → {metric}",
                    fontsize=12, fontweight="bold")

        # Add value annotations
        for i in range(len(grouped.index)):
            for j in range(len(grouped.columns)):
                value = grouped.values[i, j]
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                       color=text_color, fontsize=9)

        cbar = plt.colorbar(im, ax=ax, label=metric)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    @staticmethod
    def plot_ablation_curves(results_df: pd.DataFrame, vary_param: str,
                            fixed_params: Dict, metric: str,
                            save_path: str) -> None:
        """Create line plot with error bars for ablation curves.

        Args:
            results_df: Ablation results DataFrame.
            vary_param: Parameter being varied (X-axis).
            fixed_params: Dictionary of fixed parameter values for filtering.
            metric: Metric to plot (Y-axis).
            save_path: Output file path.
        """
        # Filter by fixed parameters
        filtered_df = results_df.copy()
        for param, value in fixed_params.items():
            filtered_df = filtered_df[filtered_df[param] == value]

        if filtered_df.empty:
            print(f"No data matches fixed parameters: {fixed_params}")
            return

        # Group by vary_param and compute mean/std
        grouped = filtered_df.groupby(vary_param)[metric].agg(["mean", "std", "count"])

        fig, ax = plt.subplots(figsize=(12, 6))
        x_vals = sorted(grouped.index.astype(float))
        y_vals = grouped.loc[x_vals, "mean"].values
        y_std = grouped.loc[x_vals, "std"].values

        ax.errorbar(x_vals, y_vals, yerr=y_std, fmt="o-", linewidth=2.5,
                   markersize=8, capsize=5, capthick=2, label=metric)
        ax.fill_between(x_vals, y_vals - y_std, y_vals + y_std, alpha=0.15)

        ax.set_xlabel(vary_param, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Ablation Study: {vary_param} Effect on {metric}",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for x, y in zip(x_vals, y_vals):
            ax.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


class DomainVisualizer:
    """Generate visualizations for cross-domain analysis."""

    @staticmethod
    def plot_tsne_features(features_dict: Dict[str, np.ndarray],
                          dataset_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          layer_name: str = "features") -> None:
        """Create t-SNE visualization of feature distributions.

        Args:
            features_dict: Dictionary mapping dataset_name -> features (N, D).
            dataset_names: List of dataset names. If None, uses keys from dict.
            save_path: Output file path. If None, displays but doesn't save.
            layer_name: Name of the layer being visualized.
        """
        if not features_dict:
            print("No features to visualize.")
            return

        if dataset_names is None:
            dataset_names = list(features_dict.keys())

        # Concatenate all features
        all_features = []
        labels = []

        for idx, dataset_name in enumerate(dataset_names):
            if dataset_name not in features_dict:
                continue

            feats = features_dict[dataset_name]
            if feats.ndim > 2:
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C*H*W)
                feats = feats.reshape(len(feats), -1)

            all_features.append(feats)
            labels.extend([idx] * len(feats))

        if not all_features:
            print("No features found in provided dictionary.")
            return

        X = np.vstack(all_features)
        print(f"Computing t-SNE on {X.shape[0]} samples with {X.shape[1]} features...")

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_2d = tsne.fit_transform(X)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
        for idx, dataset_name in enumerate(dataset_names):
            mask = np.array(labels) == idx
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[idx]], s=30,
                      label=dataset_name, alpha=0.6, edgecolors="black", linewidth=0.5)

        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.set_title(f"Feature Distribution (t-SNE): {layer_name}",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, markerscale=1.5)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_domain_gap_matrix(domain_gap_df: pd.DataFrame, save_path: str) -> None:
        """Visualize pairwise domain gaps as a heatmap.

        Args:
            domain_gap_df: DataFrame with dataset pairs and gap metrics.
            save_path: Output file path.
        """
        # Assume DataFrame has columns: dataset1, dataset2, wasserstein, mmd
        if domain_gap_df.empty:
            print("Empty domain gap DataFrame.")
            return

        datasets = sorted(set(domain_gap_df["dataset1"].tolist() +
                            domain_gap_df["dataset2"].tolist()))

        # Build symmetric matrix
        n = len(datasets)
        gap_matrix = np.zeros((n, n))

        for _, row in domain_gap_df.iterrows():
            i = datasets.index(row["dataset1"])
            j = datasets.index(row["dataset2"])
            gap = row.get("wasserstein", 0.0)
            gap_matrix[i, j] = gap
            gap_matrix[j, i] = gap

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(gap_matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(datasets, fontsize=10, rotation=45, ha="right")
        ax.set_yticklabels(datasets, fontsize=10)
        ax.set_title("Domain Gap Matrix (Wasserstein Distance)",
                    fontsize=12, fontweight="bold")

        # Annotations
        for i in range(n):
            for j in range(n):
                text_color = "white" if gap_matrix[i, j] > gap_matrix.max() / 2 else "black"
                ax.text(j, i, f"{gap_matrix[i, j]:.3f}", ha="center", va="center",
                       color=text_color, fontsize=9)

        cbar = plt.colorbar(im, ax=ax, label="Wasserstein Distance")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    @staticmethod
    def plot_transfer_prediction(domain_gaps: List[float],
                                test_metrics: List[float],
                                save_path: str) -> None:
        """Plot transfer success prediction (domain gap vs test mDice).

        Args:
            domain_gaps: List of domain gap values.
            test_metrics: List of corresponding test mDice values.
            save_path: Output file path.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Scatter plot
        ax.scatter(domain_gaps, test_metrics, s=100, alpha=0.6, c="steelblue",
                  edgecolors="black", linewidth=1)

        # Fit line (if enough points)
        if len(domain_gaps) >= 2:
            z = np.polyfit(domain_gaps, test_metrics, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(domain_gaps), max(domain_gaps), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7,
                   label=f"Linear fit: y={z[0]:.3f}x+{z[1]:.3f}")

        ax.set_xlabel("Domain Gap (Wasserstein)", fontsize=11)
        ax.set_ylabel("Zero-Shot Test mDice", fontsize=11)
        ax.set_title("Transfer Success Prediction: Domain Gap vs Performance",
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
