import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime

from seg.data import DATASET_SPECS, resolve_data_dir, resolve_dataset_key
from seg.training import TrainConfig, run_training
from seg.transforms import VALID_AUG_MODES

try:
    import tifffile  # noqa: F401
except ImportError:
    tifffile = None


DEFAULT_EXPERIMENTS = [
    {"name": "baseline_strong_aug", "aug_mode": "strong", "decoder_dropout": 0.2, "freeze_blocks_until": 6},
    {"name": "weak_aug", "aug_mode": "weak", "decoder_dropout": 0.2, "freeze_blocks_until": 6},
    {"name": "no_aug", "aug_mode": "none", "decoder_dropout": 0.2, "freeze_blocks_until": 6},
    {"name": "no_decoder_dropout", "aug_mode": "strong", "decoder_dropout": 0.0, "freeze_blocks_until": 6},
    {"name": "unfreeze_more", "aug_mode": "strong", "decoder_dropout": 0.2, "freeze_blocks_until": 4},
]


def parse_args():
    dataset_choices = sorted(set(list(DATASET_SPECS.keys())))
    parser = argparse.ArgumentParser(description="Run ablation experiments for DINOv2-UNet segmentation.")
    parser.add_argument("--dataset", required=True, choices=dataset_choices, help="Dataset key.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--save-root", type=str, default=None, help="Root folder to store ablation runs.")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-tta", action="store_true", help="Enable TTA during test (overrides per-experiment).")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help="Subset of experiment names to run (defaults to all).")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    return parser.parse_args()


def build_experiments(selected):
    if selected is None:
        return DEFAULT_EXPERIMENTS
    names = set(selected)
    return [exp for exp in DEFAULT_EXPERIMENTS if exp["name"] in names]


def main():
    args = parse_args()
    dataset_key = resolve_dataset_key(args.dataset)
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported: {', '.join(sorted(DATASET_SPECS.keys()))}")
    if args.list:
        print("Available experiments:")
        for exp in DEFAULT_EXPERIMENTS:
            print(f"- {exp['name']}: {exp}")
        return

    spec = DATASET_SPECS[dataset_key]
    if spec.requires_tifffile and tifffile is None:
        raise RuntimeError("tifffile is required for the selected dataset. Install with `pip install tifffile imagecodecs`.")

    data_dir = resolve_data_dir(spec, args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    save_root = os.path.abspath(args.save_root) if args.save_root else os.path.join(spec.default_save_dir, "ablation")
    os.makedirs(save_root, exist_ok=True)

    experiments = build_experiments(args.experiments)
    if not experiments:
        raise ValueError("No experiments to run (check --experiments names).")

    summary = []
    for exp in experiments:
        if exp.get("aug_mode") not in VALID_AUG_MODES:
            raise ValueError(f"Invalid aug_mode in experiment {exp['name']}: {exp.get('aug_mode')}")

        exp_save_dir = os.path.join(save_root, exp["name"])
        cfg = TrainConfig(
            dataset=dataset_key,
            data_dir=data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            num_workers=args.num_workers,
            no_amp=args.no_amp,
            patience=args.patience,
            save_dir=exp_save_dir,
            seed=args.seed,
            aug_mode=exp.get("aug_mode", "strong"),
            decoder_dropout=exp.get("decoder_dropout", 0.2),
            freeze_blocks_until=exp.get("freeze_blocks_until", 6),
            use_tta=args.use_tta if args.use_tta else exp.get("use_tta", True),
        )

        print(f"\n=== Running experiment: {exp['name']} ===")
        results = run_training(cfg, dataset_key, spec)
        summary.append({
            "experiment": exp["name"],
            "save_dir": exp_save_dir,
            "best_val": results.get("best_val", {}),
            "test": results.get("test", {}),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(save_root, f"summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also dump a flat CSV with top-level metrics (best_val and test mdice/miou if available)
    csv_path = os.path.join(save_root, f"summary_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "save_dir", "val_mDice", "val_mIoU", "test_mDice", "test_mIoU"])
        for row in summary:
            best_val = row.get("best_val", {}) or {}
            test = row.get("test", {}) or {}
            writer.writerow([
                row["experiment"],
                row["save_dir"],
                best_val.get("mDice"),
                best_val.get("mIoU"),
                test.get("mDice"),
                test.get("mIoU"),
            ])

    print(f"\nAblation finished. Summary written to:\n- {summary_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
