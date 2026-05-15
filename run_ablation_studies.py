#!/usr/bin/env python3
"""Run lightweight hyperparameter ablation studies for DINOv2-UNet."""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from itertools import product
from typing import Any, Dict, List, Optional


class AblationScheduler:
    """Generate train.py commands for ablation configurations."""

    DEFAULT_RANGES = {
        "freeze_blocks_until": [0, 3, 6, 9, 12],
        "lr_ratio": [0.001, 0.01, 0.1, 1.0],
        "aux_weight": [0.0, 0.3, 0.5, 0.7],
        "img_size": [256, 384, 448, 512],
        "warmup_epochs": [0, 3, 5, 10],
        "grad_clip": [0.5, 1.0, 2.0, 5.0],
        "optimizer_strategy": ["partial_finetune", "frozen_encoder", "full_finetune"],
        "decoder_type": ["simple", "complex"],
        "pretrained_type": ["dinov2", "imagenet_supervised"],
    }

    def generate_combinations(
        self,
        axes: List[str],
        axis_values: Dict[str, List[Any]],
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ranges_to_use = {}
        for axis in axes:
            values = axis_values.get(axis, self.DEFAULT_RANGES.get(axis, []))
            if not values:
                raise ValueError(f"No values configured for ablation axis '{axis}'.")
            ranges_to_use[axis] = values

        keys = list(ranges_to_use.keys())
        configs = [dict(zip(keys, values)) for values in product(*(ranges_to_use[k] for k in keys))]
        if sample_size and len(configs) > sample_size:
            indices = sorted(random.Random(42).sample(range(len(configs)), sample_size))
            configs = [configs[i] for i in indices]
        return configs

    def config_to_args(
        self,
        config: Dict[str, Any],
        base_args: argparse.Namespace,
        save_dir_override: Optional[str] = None,
    ) -> List[str]:
        save_dir = save_dir_override or base_args.save_dir
        args = [
            sys.executable,
            "train.py",
            "--dataset",
            base_args.dataset,
            "--data-dir",
            base_args.data_dir,
            "--save-dir",
            save_dir,
            "--batch-size",
            str(base_args.batch_size),
            "--epochs",
            str(base_args.epochs),
            "--num-workers",
            str(base_args.num_workers),
        ]

        optional_scalar_args = {
            "freeze_blocks_until": "--freeze-blocks-until",
            "img_size": "--img-size",
            "warmup_epochs": "--warmup-epochs",
            "grad_clip": "--grad-clip",
            "optimizer_strategy": "--optimizer-strategy",
            "decoder_type": "--decoder-type",
            "pretrained_type": "--pretrained-type",
        }
        for key, flag in optional_scalar_args.items():
            if key in config:
                args.extend([flag, str(config[key])])

        if "aux_weight" in config:
            args.extend(["--aux-weight-scale", str(config["aux_weight"])])

        if "lr_ratio" in config:
            lr_decoder = base_args.lr
            lr_backbone = lr_decoder * float(config["lr_ratio"])
            args.extend(["--lr", str(lr_decoder), "--lr-backbone", str(lr_backbone)])
        else:
            args.extend(["--lr", str(base_args.lr), "--lr-backbone", str(base_args.lr_backbone)])

        if base_args.max_train_batches is not None:
            args.extend(["--max-train-batches", str(base_args.max_train_batches)])
        if base_args.max_eval_batches is not None:
            args.extend(["--max-eval-batches", str(base_args.max_eval_batches)])
        if base_args.no_amp:
            args.append("--no-amp")
        if not base_args.export_masks:
            args.append("--no-export")
        if not base_args.track_metrics:
            args.append("--no-track-metrics")

        return args


def _best_validation_metrics(metrics_path: str) -> Optional[Dict[str, float]]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    best = None
    for epoch_data in metrics_data.get("metrics", {}).values():
        val = epoch_data.get("val")
        if not val:
            continue
        if best is None or val.get("mDice", -1.0) > best.get("mDice", -1.0):
            best = {
                "mDice": float(val.get("mDice", 0.0)),
                "mIoU": float(val.get("mIoU", 0.0)),
            }
    return best


def _write_summary_csv(path: str, rows: List[Dict[str, Any]], axes: List[str]) -> None:
    fieldnames = ["seed", "mDice", "mIoU"] + axes
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _print_best(rows: List[Dict[str, Any]], axes: List[str], top_k: int = 5) -> None:
    best_rows = sorted(rows, key=lambda item: item["mDice"], reverse=True)[:top_k]
    headers = ["mDice", "mIoU"] + axes
    widths = {header: max(len(header), 8) for header in headers}
    for row in best_rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    print(" ".join(header.rjust(widths[header]) for header in headers))
    for row in best_rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).rjust(widths[header]))
        print(" ".join(values))


def _plot_heatmaps(rows: List[Dict[str, Any]], axes: List[str], output_dir: str) -> None:
    if len(axes) != 2:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: install matplotlib to generate heatmaps.")
        return

    x_axis, y_axis = axes
    x_values = sorted({row[x_axis] for row in rows}, key=str)
    y_values = sorted({row[y_axis] for row in rows}, key=str)
    for metric in ("mDice", "mIoU"):
        matrix = []
        for y_value in y_values:
            row_values = []
            for x_value in x_values:
                matches = [
                    row[metric]
                    for row in rows
                    if row[x_axis] == x_value and row[y_axis] == y_value
                ]
                row_values.append(sum(matches) / len(matches) if matches else 0.0)
            matrix.append(row_values)

        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(x_values)), labels=[str(v) for v in x_values], rotation=45)
        ax.set_yticks(range(len(y_values)), labels=[str(v) for v in y_values])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{metric}: {x_axis} x {y_axis}")
        fig.colorbar(image, ax=ax, label=metric)
        fig.tight_layout()
        save_path = os.path.join(output_dir, f"heatmap_{x_axis}_x_{y_axis}_{metric}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved heatmap: {save_path}")


def run_ablation_studies(args: argparse.Namespace) -> None:
    if not args.axes:
        raise SystemExit("Error: specify at least one axis with --axes.")

    scheduler = AblationScheduler()
    unsupported = [axis for axis in args.axes if axis not in scheduler.DEFAULT_RANGES]
    if unsupported:
        raise SystemExit(f"Unsupported ablation axis: {unsupported}")

    axis_values = {}
    for axis in args.axes:
        values = getattr(args, axis, None)
        if values:
            axis_values[axis] = values
        print(f"  {axis}: {axis_values.get(axis, scheduler.DEFAULT_RANGES[axis])}")

    configs = scheduler.generate_combinations(
        args.axes,
        axis_values,
        sample_size=args.sample_size,
    )
    total_runs = len(configs) * args.num_seeds
    print(f"\nTotal configurations: {len(configs)}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Total training runs: {total_runs}\n")

    os.makedirs(args.save_results, exist_ok=True)
    results = []
    failed_runs = []

    for run_idx, config in enumerate(configs):
        for seed_idx in range(args.num_seeds):
            run_num = run_idx * args.num_seeds + seed_idx + 1
            print(f"\n[{run_num}/{total_runs}] Running config={config}, seed={seed_idx}")

            run_save_dir = os.path.join(args.save_dir, "ablation_runs", f"run_{run_num:04d}")
            cmd = scheduler.config_to_args(config, args, save_dir_override=run_save_dir)
            cmd.extend(["--seed", str(42 + seed_idx)])
            print(f"Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=False,
                    timeout=args.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                print(f"Error: training timed out after {args.timeout_seconds}s")
                failed_runs.append((config, seed_idx))
                continue

            if result.returncode != 0:
                print(f"Warning: training exited with code {result.returncode}")
                failed_runs.append((config, seed_idx))
                continue

            metrics_path = os.path.join(
                run_save_dir,
                f"dinov2_unet_{args.dataset}",
                "metrics_history.json",
            )
            if not os.path.exists(metrics_path):
                print(f"Warning: metrics_history.json not found at {metrics_path}")
                failed_runs.append((config, seed_idx))
                continue

            best = _best_validation_metrics(metrics_path)
            if not best:
                print(f"Warning: no validation metrics found in {metrics_path}")
                failed_runs.append((config, seed_idx))
                continue

            result_entry = {"seed": seed_idx, **best, **config}
            results.append(result_entry)
            print(f"[ok] Completed. mDice={best['mDice']:.4f}, mIoU={best['mIoU']:.4f}")

    if not results:
        print("\n[warn] No results to save. Check logs above for errors.")
        return

    results_csv = os.path.join(args.save_results, "summary.csv")
    _write_summary_csv(results_csv, results, args.axes)
    print(f"\n[ok] Saved results to {results_csv}")

    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"Completed runs: {len(results)}/{total_runs}")
    if failed_runs:
        print(f"Failed runs: {len(failed_runs)}")
    print("\nBest configurations (by mDice):")
    _print_best(results, args.axes)

    if args.plot_heatmaps:
        _plot_heatmaps(results, args.axes, args.save_results)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run DINOv2-UNet ablation studies.")
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. kvasir.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory or data root.")
    parser.add_argument("--axes", nargs="+", default=[], help="Ablation axes to vary.")

    parser.add_argument("--freeze-blocks-until", type=int, nargs="+", default=None)
    parser.add_argument("--lr-ratio", type=float, nargs="+", default=None)
    parser.add_argument("--aux-weight", type=float, nargs="+", default=None)
    parser.add_argument("--img-size", type=int, nargs="+", default=None)
    parser.add_argument("--warmup-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--grad-clip", type=float, nargs="+", default=None)
    parser.add_argument(
        "--optimizer-strategy",
        nargs="+",
        choices=["partial_finetune", "frozen_encoder", "full_finetune"],
        default=None,
    )
    parser.add_argument(
        "--decoder-type",
        nargs="+",
        choices=["simple", "complex"],
        default=None,
    )
    parser.add_argument(
        "--pretrained-type",
        nargs="+",
        choices=["dinov2", "imagenet_supervised"],
        default=None,
    )

    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--save-results", default="ablation_results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--export-masks", action="store_true")
    parser.add_argument("--plot-heatmaps", action="store_true")
    parser.add_argument("--track-metrics", action="store_true", default=True)
    parser.add_argument("--no-track-metrics", action="store_false", dest="track_metrics")
    return parser.parse_args(argv)


def main():
    run_ablation_studies(parse_args())


if __name__ == "__main__":
    main()
