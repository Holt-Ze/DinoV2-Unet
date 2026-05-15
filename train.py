"""Train DINOv2-UNet on one or more polyp segmentation datasets."""

import argparse
import os

DATASET_KEYS = ("clinicdb", "colondb", "etis", "kvasir")
VALID_AUG_MODES = ("strong", "weak", "none")


def parse_args(args=None):
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train DINOv2-UNet for polyp segmentation."
    )

    parser.add_argument(
        "--data",
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        required=True,
        help=f"One or more dataset keys ({', '.join(DATASET_KEYS)}).",
    )
    parser.add_argument(
        "--data-dir",
        "--data-root",
        dest="data_dir",
        type=str,
        default=None,
        help="Dataset directory or root containing dataset subdirectories.",
    )
    parser.add_argument(
        "--save-dir",
        "--save-root",
        dest="save_dir",
        type=str,
        default=None,
        help="Root directory for checkpoints and outputs.",
    )

    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze-blocks-until", type=int, default=6)
    parser.add_argument("--decoder-dropout", type=float, default=0.2)
    parser.add_argument(
        "--pretrained-type",
        type=str,
        default="dinov2",
        choices=["dinov2", "imagenet_supervised"],
        help="Backbone pretraining family for ablation experiments.",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="simple",
        choices=["simple", "complex"],
        help="Decoder variant for ablation experiments.",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight-scale", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aug-mode",
        type=str,
        choices=VALID_AUG_MODES,
        default="strong",
        help="Training augmentation strength.",
    )
    parser.add_argument(
        "--optimizer-strategy",
        type=str,
        default="partial_finetune",
        choices=["frozen_encoder", "full_finetune", "partial_finetune"],
        help="Encoder fine-tuning strategy for ablation experiments.",
    )

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument(
        "--joint-train",
        action="store_true",
        help="Combine all specified datasets into one joint training run.",
    )

    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument(
        "--track-metrics",
        action="store_true",
        default=True,
        dest="track_metrics",
        help="Write metrics_history.json after each epoch (default: enabled).",
    )
    parser.add_argument(
        "--no-track-metrics",
        action="store_false",
        dest="track_metrics",
        help="Disable metrics_history.json output.",
    )

    parser.add_argument("--no-export", action="store_true", help="Skip mask export.")
    parser.add_argument(
        "--export-splits",
        nargs="*",
        default=["test"],
        help="Splits to export masks for: train val test or all.",
    )

    return parser.parse_args(args)


def main():
    from seg.pipeline import configure_runtime, run_training_jobs

    repo_root = os.path.dirname(os.path.abspath(__file__))
    configure_runtime(repo_root)
    run_training_jobs(parse_args(), repo_root)


if __name__ == "__main__":
    main()
