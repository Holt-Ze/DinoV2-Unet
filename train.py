import os
import sys
import argparse
import warnings
import datetime
from seg.data import (
    DATASET_ALIASES,
    DATASET_SPECS,
    resolve_data_dir,
    resolve_dataset_key,
)
from seg.training import TrainConfig, run_training
from seg.transforms import VALID_AUG_MODES
from seg.inference import export_dataset_masks

# Filter warnings
warnings.filterwarnings("ignore", message="Error fetching version info", category=UserWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")

try:
    import tifffile  # noqa: F401
except ImportError:
    tifffile = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_OFFLINE"] = "1"


class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    dataset_choices = sorted(set(list(DATASET_SPECS.keys()) + list(DATASET_ALIASES.keys())))
    parser = argparse.ArgumentParser(description="Train the DINOv2-UNet polyp segmenter on supported datasets.")
    parser.add_argument("--data", "--dataset", "--datasets", dest="datasets", nargs='+', required=True,
                        help=f"One or more dataset keys ({', '.join(sorted(DATASET_SPECS.keys()))}).")
    parser.add_argument("--data-dir", "--data-root", dest="data_dir", type=str, default=None,
                        help="Override dataset root directory.")
    parser.add_argument("--save-dir", "--save-root", dest="save_dir", type=str, default=None,
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
                        help="Training augmentation strength. strong=default, weak=flip/rotate only, none=no extra.")
    parser.add_argument("--decoder-dropout", type=float, default=0.2,
                        help="Dropout probability in UNet decoder.")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable Test Time Augmentation (horizontal flip).")
    parser.add_argument("--no-profile", action="store_true",
                        help="Disable model Params/FLOPs/FPS profiling.")
    
    # New argument
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of training data to use (0.0 to 1.0). Default is 1.0 (all data).")

    # Ablation Study arguments
    parser.add_argument("--optimizer-strategy", type=str, default="partial_finetune",
                        choices=["frozen_encoder", "full_finetune", "partial_finetune"],
                        help="Finetuning strategy: frozen_encoder, full_finetune, or partial_finetune (default).")
    parser.add_argument("--pretrained-type", type=str, default="dinov2",
                        choices=["dinov2", "imagenet_supervised"],
                        help="Pretrained weights type: dinov2 (default) or imagenet_supervised.")
    parser.add_argument("--decoder-type", type=str, default="simple",
                        choices=["simple", "complex"],
                        help="Decoder type: simple (Streamlined) or complex (Attention Gates).")
    
    # Export options
    parser.add_argument("--no-export", action="store_true", help="Skip mask export step.")
    parser.add_argument("--export-splits", nargs="*", default=["test"],
                        help="Splits to export masks for: train val test or all.")

    return parser.parse_args()


def resolve_split_list(values):
    splits = [v.lower() for v in values]
    if len(splits) == 1 and splits[0] == "all":
        return ["train", "val", "test"]
    return splits


def main():
    args = parse_args()
    datasets_keys = [resolve_dataset_key(d) for d in args.datasets]
    export_splits = resolve_split_list(args.export_splits)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_root, exist_ok=True)

    for dataset_key in datasets_keys:
        if dataset_key not in DATASET_SPECS:
            print(f"Skipping unsupported dataset: {dataset_key}")
            continue

        spec = DATASET_SPECS[dataset_key]
        if spec.requires_tifffile and tifffile is None:
            print(f"Skipping {dataset_key}: tifffile required. Install with `pip install tifffile imagecodecs`.")
            continue

        # Setup logging
        dataset_log_dir = os.path.join(log_root, dataset_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(dataset_log_dir, f"{timestamp}.log")

        logger = Logger(log_file, original_stdout)
        sys.stdout = logger
        sys.stderr = logger

        print(f"\n{'='*40}\nProcessing Dataset: {dataset_key}\n{'='*40}\n")

        # Resolve Data Directory
        if args.data_dir:
            if len(datasets_keys) > 1 and spec.default_subdir:
                 # If passing a root for multiple datasets, append the default subdir
                 # Assumes args.data_dir is a common root containing dataset subfolders
                 # UNLESS the user passed specific single path for a single dataset.
                 # Heuristic: if passed data_dir exists and has images, use it? 
                 # Simplest logic: if multiple datasets, assume data_dir is ROOT.
                 cur_data_dir = os.path.join(args.data_dir, spec.default_subdir)
            else:
                 # If single dataset, args.data_dir is likely the direct path
                 # OR if it's a root without subdirs needed.
                 # Let's stick to resolve_data_dir logic usually, but override.
                 # If user provided data_dir, we act like it's the root if we can find default_subdir inside.
                 # Otherwise we assume it's the dataset dir itself if single dataset.
                 if len(datasets_keys) == 1:
                     cur_data_dir = args.data_dir
                 else:
                     cur_data_dir = os.path.join(args.data_dir, spec.default_subdir) if spec.default_subdir else args.data_dir
        else:
            cur_data_dir = resolve_data_dir(spec, None)

        if not os.path.isdir(cur_data_dir):
            print(f"[Error] Dataset directory not found: {cur_data_dir}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            continue

        # Resolve Save Directory
        if args.save_dir:
             if len(datasets_keys) > 1:
                 cur_save_dir = os.path.join(args.save_dir, f"dinov2_unet_{dataset_key}")
             else:
                 cur_save_dir = args.save_dir
        else:
            cur_save_dir = spec.default_save_dir

        # Append ratio to save_dir to avoid overwriting default runs
        if args.ratio < 1.0:
            cur_save_dir = f"{cur_save_dir}_ratio{int(args.ratio*100)}"
        
        cfg = TrainConfig(
            dataset=dataset_key,
            data_dir=cur_data_dir,
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
            save_dir=cur_save_dir,
            seed=args.seed,
            aug_mode=args.aug_mode,
            decoder_dropout=args.decoder_dropout,
            use_tta=not args.no_tta,
            profile=not args.no_profile,
            subset_ratio=args.ratio,
            pretrained_type=args.pretrained_type,
            decoder_type=args.decoder_type,
            optimizer_strategy=args.optimizer_strategy,
        )

        try:
            run_training(cfg, dataset_key, spec)
        except Exception as e:
            print(f"Error training {dataset_key}: {e}")
            import traceback
            traceback.print_exc()

        if not args.no_export:
            export_dir = os.path.join(cur_save_dir, "pred_masks")
            try:
                export_dataset_masks(
                    dataset_key=dataset_key,
                    data_dir=cur_data_dir,
                    save_dir=export_dir,
                    backbone=args.backbone,
                    out_indices=cfg.out_indices,
                    img_size=cfg.img_size,
                    freeze_blocks_until=args.freeze_blocks_until,
                    decoder_dropout=args.decoder_dropout,
                    num_workers=args.num_workers,
                    splits=export_splits,
                    checkpoint_path=os.path.join(cur_save_dir, "best.pt"),
                    device=None,
                    seed=args.seed,
                )
                print(f"Saved masks to {export_dir}")
            except Exception as e:
                print(f"Error exporting masks for {dataset_key}: {e}")

        # Restore stdout/stderr for next iteration
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == '__main__':
    main()
