#!/usr/bin/env python3
"""Orchestrate systematic ablation studies across multiple hyperparameters.

Usage:
    # Focused ablation: freeze_blocks vs lr_ratio
    python run_ablation_studies.py --dataset kvasir \
        --axes freeze_blocks_until lr_ratio \
        --freeze-blocks-until 0 3 6 9 12 \
        --lr-ratio 0.001 0.01 0.1 1.0 \
        --num-seeds 3 \
        --save-results ablation_results/

    # Single axis: vary freeze_blocks only
    python run_ablation_studies.py --dataset kvasir \
        --axes freeze_blocks_until \
        --freeze-blocks-until 0 3 6 9 12 \
        --num-seeds 2
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product

import pandas as pd
import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


class AblationScheduler:
    """Generate hyperparameter configurations for ablation studies."""

    # Default ranges for each axis
    DEFAULT_RANGES = {
        "freeze_blocks_until": [0, 3, 6, 9, 12],
        "lr_ratio": [0.001, 0.01, 0.1, 1.0],           # lr_backbone / lr_decoder
        "aux_weight": [0.0, 0.3, 0.5, 0.7],            # deep supervision weight scaling
        "img_size": [256, 384, 448, 512],
        "warmup_epochs": [0, 3, 5, 10],
        "grad_clip": [0.5, 1.0, 2.0, 5.0],
    }

    def __init__(self):
        self.ranges = self.DEFAULT_RANGES.copy()

    def generate_combinations(self, axes: List[str],
                            axis_values: Dict[str, List],
                            sample_size: int = None) -> List[Dict[str, Any]]:
        """Generate hyperparameter combinations.

        Args:
            axes: List of axes to vary.
            axis_values: Dict mapping axis_name -> list of values to test.
            sample_size: If specified, randomly sample this many combinations.

        Returns:
            List of configuration dictionaries.
        """
        # Use provided values or defaults
        ranges_to_use = {}
        for axis in axes:
            if axis in axis_values:
                ranges_to_use[axis] = axis_values[axis]
            else:
                ranges_to_use[axis] = self.DEFAULT_RANGES.get(axis, [])

        # Generate cartesian product
        keys = list(ranges_to_use.keys())
        values = [ranges_to_use[k] for k in keys]

        configs = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)

        # Optionally sample
        if sample_size and len(configs) > sample_size:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(configs), size=sample_size, replace=False)
            configs = [configs[i] for i in sorted(indices)]

        return configs

    def _config_to_args(
        self,
        config: Dict[str, Any],
        base_args: argparse.Namespace,
        save_dir_override: str = None,
    ) -> List[str]:
        """Convert a config dict to train.py command-line arguments.

        Args:
            config: Configuration dictionary.
            base_args: Base arguments (dataset, data_dir, etc.).

        Returns:
            List of command-line argument strings.
        """
        save_dir = save_dir_override or base_args.save_dir
        args = [
            sys.executable, "train.py",
            "--dataset", base_args.dataset,
            "--data-dir", base_args.data_dir,
            "--save-dir", save_dir,
            "--batch-size", str(base_args.batch_size),
            "--epochs", str(base_args.epochs),
            "--num-workers", str(base_args.num_workers),
        ]

        if getattr(base_args, "max_train_batches", None) is not None:
            args.extend(["--max-train-batches", str(base_args.max_train_batches)])
        if getattr(base_args, "max_eval_batches", None) is not None:
            args.extend(["--max-eval-batches", str(base_args.max_eval_batches)])

        # Add ablation-specific arguments
        if "freeze_blocks_until" in config:
            args.extend(["--freeze-blocks-until", str(config["freeze_blocks_until"])])

        if "img_size" in config:
            args.extend(["--img-size", str(config["img_size"])])

        if "warmup_epochs" in config:
            args.extend(["--warmup-epochs", str(config["warmup_epochs"])])

        if "grad_clip" in config:
            args.extend(["--grad-clip", str(config["grad_clip"])])

        if "aux_weight" in config:
            args.extend(["--aux-weight-scale", str(config["aux_weight"])])

        # Handle lr_ratio: compute lr and lr_backbone
        if "lr_ratio" in config:
            lr_ratio = config["lr_ratio"]
            # Default lr_decoder = 1e-3
            lr_decoder = 1e-3
            lr_backbone = lr_decoder * lr_ratio
            args.extend(["--lr", str(lr_decoder)])
            args.extend(["--lr-backbone", str(lr_backbone)])

        # Add tracking flags
        if base_args.track_metrics:
            args.append("--track-metrics")

        return args


def run_ablation_studies(args: argparse.Namespace):
    """Execute ablation studies."""

    # Validate inputs
    if not args.axes:
        print("Error: No axes specified. Use --axes to specify parameters to vary.")
        sys.exit(1)

    scheduler = AblationScheduler()

    # Parse axis values from command line
    axis_values = {}
    for axis in args.axes:
        arg_name = axis.replace("-", "_")
        if hasattr(args, arg_name):
            values = getattr(args, arg_name)
            if values:
                axis_values[axis] = values
                print(f"  {axis}: {values}")
        else:
            print(f"Warning: No values found for axis '{axis}'. Using defaults.")

    # Generate configurations
    if args.sample_size:
        print(f"\nGenerating {args.sample_size} random combinations...")
        configs = scheduler.generate_combinations(
            args.axes, axis_values, sample_size=args.sample_size
        )
    else:
        print(f"\nGenerating all combinations...")
        configs = scheduler.generate_combinations(args.axes, axis_values)

    print(f"Total configurations: {len(configs)}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Total training runs: {len(configs) * args.num_seeds}\n")

    # Setup results directory
    os.makedirs(args.save_results, exist_ok=True)

    results = []
    failed_runs = []

    print(f"Starting ablation study (saving results to {args.save_results})...\n")

    # Execute training runs
    for run_idx, config in enumerate(configs):
        for seed_idx in range(args.num_seeds):
            run_num = run_idx * args.num_seeds + seed_idx + 1
            total_runs = len(configs) * args.num_seeds

            print(f"\n[{run_num}/{total_runs}] Running with config: {config}, seed={seed_idx}")

            # Prepare arguments

            base_args = args
            run_save_dir = os.path.join(
                args.save_dir, "ablation_runs", f"run_{run_num:04d}"
            )
            cmd = scheduler._config_to_args(
                config, base_args, save_dir_override=run_save_dir
            )

            # Add seed
            cmd.extend(["--seed", str(42 + seed_idx)])

            print(f"Command: {' '.join(cmd[:10])}...\n")

            # Execute training
            try:
                result = subprocess.run(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=False,
                    timeout=3600,  # 1 hour timeout
                )

                if result.returncode != 0:
                    print(f"Warning: Training exited with code {result.returncode}")
                    failed_runs.append((config, seed_idx))
                    continue

                # Parse metrics from metrics_history.json
                save_dir = os.path.join(
                    run_save_dir, f"dinov2_unet_{args.dataset}"
                )
                metrics_path = os.path.join(save_dir, "metrics_history.json")

                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics_data = json.load(f)

                    # Get best validation metrics
                    best_mdice = -1
                    best_miou = -1
                    for epoch_str, epoch_data in metrics_data.get("metrics", {}).items():
                        if "val" in epoch_data:
                            mdice = epoch_data["val"].get("mDice", 0)
                            miou = epoch_data["val"].get("mIoU", 0)
                            if mdice > best_mdice:
                                best_mdice = mdice
                                best_miou = miou

                    result_entry = {
                        "seed": seed_idx,
                        "mDice": best_mdice,
                        "mIoU": best_miou,
                    }
                    result_entry.update(config)
                    results.append(result_entry)

                    print(f"[ok] Completed. mDice={best_mdice:.4f}, mIoU={best_miou:.4f}")
                else:
                    print(f"Warning: metrics_history.json not found at {metrics_path}")
                    failed_runs.append((config, seed_idx))

            except subprocess.TimeoutExpired:
                print(f"Error: Training timeout (>1 hour)")
                failed_runs.append((config, seed_idx))
            except Exception as e:
                print(f"Error: {e}")
                failed_runs.append((config, seed_idx))

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(args.save_results, "summary.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"\n[ok] Saved results to {results_csv}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("ABLATION STUDY SUMMARY")
        print("=" * 60)
        print(f"Completed runs: {len(results)}/{len(configs) * args.num_seeds}")
        if failed_runs:
            print(f"Failed runs: {len(failed_runs)}")
        print("\nBest configurations (by mDice):")
        top_k = min(5, len(results_df))
        best = results_df.nlargest(top_k, "mDice")
        print(best[["mDice", "mIoU"] + args.axes].to_string(index=False))

        # Trigger visualization
        if len(args.axes) == 2:
            try:
                from analysis.visualization import AblationVisualizer
                ax1, ax2 = args.axes[0], args.axes[1]
                AblationVisualizer.plot_heatmap_2d(
                    results_df, ax1, ax2, "mDice",
                    os.path.join(args.save_results,
                                f"heatmap_{ax1}_x_{ax2}_mdice.png")
                )
                AblationVisualizer.plot_heatmap_2d(
                    results_df, ax1, ax2, "mIoU",
                    os.path.join(args.save_results,
                                f"heatmap_{ax1}_x_{ax2}_miou.png")
                )
            except ImportError:
                print("Note: Install matplotlib + sklearn to generate heatmap plots.")

    else:
        print("\n[warn] No results to save. Check logs above for errors.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate systematic ablation studies."
    )

    # Required arguments
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset key (e.g., 'kvasir').",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to data directory.",
    )

    # Ablation configuration
    parser.add_argument(
        "--axes", nargs="+", default=[],
        help="Axes to vary in ablation (e.g., freeze_blocks_until lr_ratio).",
    )
    parser.add_argument(
        "--freeze-blocks-until", type=int, nargs="+", default=None,
        help="Values for freeze_blocks_until axis.",
    )
    parser.add_argument(
        "--lr-ratio", type=float, nargs="+", default=None,
        help="Values for lr_ratio axis (lr_backbone / lr_decoder).",
    )
    parser.add_argument(
        "--img-size", type=int, nargs="+", default=None,
        help="Values for img_size axis.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, nargs="+", default=None,
        help="Values for warmup_epochs axis.",
    )
    parser.add_argument(
        "--grad-clip", type=float, nargs="+", default=None,
        help="Values for grad_clip axis.",
    )

    # Sampling options
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="If specified, randomly sample this many configurations.",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of random seeds to test per configuration.",
    )

    # Training defaults
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--track-metrics", action="store_true", default=True)

    # Output
    parser.add_argument(
        "--save-results", default="ablation_results",
        help="Directory to save ablation results and plots.",
    )

    args = parser.parse_args()
    run_ablation_studies(args)


if __name__ == "__main__":
    main()
