import os
import argparse
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_OFFLINE"] = "1"

warnings.filterwarnings("ignore", message="Error fetching version info", category=UserWarning)

from seg.data import (
    DATASET_ALIASES,
    DATASET_SPECS,
    resolve_data_dir,
    resolve_dataset_key,
)
from seg.training import TrainConfig, run_training
from seg.transforms import VALID_AUG_MODES

try:
    import tifffile  # noqa: F401
except ImportError:
    tifffile = None


def parse_args():
    dataset_choices = sorted(set(list(DATASET_SPECS.keys()) + list(DATASET_ALIASES.keys())))
    parser = argparse.ArgumentParser(description="Train the DINOv2-UNet polyp segmenter on supported datasets.")
    parser.add_argument("--data", "--dataset", dest="dataset", required=True, choices=dataset_choices,
                        help=f"Dataset key ({', '.join(sorted(DATASET_SPECS.keys()))}).")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=None,
                        help="Override dataset root directory.")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None,
                        help="Directory to store checkpoints and outputs.")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze-blocks-until", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-mode", type=str, choices=VALID_AUG_MODES, default="strong",
                        help="训练阶段的增强强度。strong=原默认增强，weak=仅翻转/旋转，none=无额外增强。")
    parser.add_argument("--decoder-dropout", type=float, default=0.2,
                        help="UNet 解码器内部的 Dropout 概率。")
    parser.add_argument("--no-tta", action="store_true",
                        help="禁用测试阶段的水平翻转 TTA。")
    parser.add_argument("--no-profile", action="store_true",
                        help="Disable model Params/FLOPs/FPS profiling.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_key = resolve_dataset_key(args.dataset)
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported keys: {', '.join(sorted(DATASET_SPECS.keys()))}")
    spec = DATASET_SPECS[dataset_key]
    if spec.requires_tifffile and tifffile is None:
        raise RuntimeError("tifffile is required for the selected dataset. Install with `pip install tifffile imagecodecs`.")

    data_dir = resolve_data_dir(spec, args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else spec.default_save_dir

    cfg = TrainConfig(
        dataset=dataset_key,
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        backbone=args.backbone,
        freeze_blocks_until=args.freeze_blocks_until,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        no_amp=args.no_amp,
        patience=args.patience,
        save_dir=save_dir,
        seed=args.seed,
        aug_mode=args.aug_mode,
        decoder_dropout=args.decoder_dropout,
        use_tta=not args.no_tta,
        profile=not args.no_profile,
    )

    run_training(cfg, dataset_key, spec)


if __name__ == '__main__':
    main()
