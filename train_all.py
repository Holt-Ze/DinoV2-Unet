import argparse
import os
import sys
import warnings
import datetime
from seg.data import DATASET_SPECS, resolve_data_dir, resolve_dataset_key
from seg.inference import export_dataset_masks
from seg.training import TrainConfig, run_training

# 过滤警告
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
warnings.filterwarnings("ignore", message=".*Error fetching version info.*")

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
    parser = argparse.ArgumentParser(description="Train DINOv2-UNet on all datasets and export masks.")
    parser.add_argument("--datasets", nargs="*", default=["kvasir", "clinicdb", "colondb", "etis"])
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override root data directory (subfolders follow dataset defaults).")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Override root output directory for checkpoints and masks.")
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
    parser.add_argument("--aug-mode", type=str, default="strong")
    parser.add_argument("--decoder-dropout", type=float, default=0.2)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
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
    datasets = [resolve_dataset_key(x) for x in args.datasets]
    export_splits = resolve_split_list(args.export_splits)

    # 保存原始的 stdout 和 stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_root, exist_ok=True)

    for dataset_key in datasets:
        # 设置每个数据集的日志
        dataset_log_dir = os.path.join(log_root, dataset_key)
        os.makedirs(dataset_log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(dataset_log_dir, f"{timestamp}.log")

        # 重定向输出
        logger = Logger(log_file, original_stdout)
        sys.stdout = logger
        sys.stderr = logger # 将错误也重定向到同一个日志文件

        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset '{dataset_key}'.")
        spec = DATASET_SPECS[dataset_key]

        if args.data_root:
            if spec.default_subdir:
                data_dir = os.path.abspath(os.path.join(args.data_root, spec.default_subdir))
            else:
                data_dir = os.path.abspath(args.data_root)
        else:
            data_dir = resolve_data_dir(spec, None)

        if args.save_root:
            save_dir = os.path.abspath(os.path.join(args.save_root, f"dinov2_unet_{dataset_key}"))
        else:
            save_dir = spec.default_save_dir

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

        try:
            run_training(cfg, dataset_key, spec)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[warn] Skip dataset '{dataset_key}': {exc}")
            continue

        if not args.no_export:
            export_dir = os.path.join(save_dir, "pred_masks")
            export_dataset_masks(
                dataset_key=dataset_key,
                data_dir=data_dir,
                save_dir=export_dir,
                backbone=args.backbone,
                out_indices=cfg.out_indices,
                img_size=cfg.img_size,
                freeze_blocks_until=args.freeze_blocks_until,
                decoder_dropout=args.decoder_dropout,
                num_workers=args.num_workers,
                splits=export_splits,
                checkpoint_path=os.path.join(save_dir, "best.pt"),
                device=None,
                seed=args.seed,
            )
            print(f"Saved masks to {export_dir}")
        
    # 恢复原始输出（虽然程序结束了，但这是一个好习惯）
    sys.stdout = original_stdout
    sys.stderr = original_stderr


if __name__ == "__main__":
    main()
