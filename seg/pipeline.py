"""CLI-facing orchestration for training jobs."""

import datetime
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .data import (
    DATASET_SPECS,
    DatasetSpec,
    resolve_data_dir_from_root,
    resolve_dataset_key,
)
from .inference import export_dataset_masks
from .training import TrainConfig, run_training

try:
    import tifffile  # noqa: F401
except ImportError:
    tifffile = None


class TeeLogger:
    """Write stdout/stderr to both the terminal and a log file."""

    def __init__(self, filename: str, stream):
        self.terminal = stream
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


@dataclass
class TrainingJob:
    dataset_key: str
    spec: DatasetSpec
    data_dir: str
    save_dir: str
    joint_train_specs: Optional[list] = None


def configure_runtime(repo_root: str) -> None:
    """Set quiet, local defaults for third-party runtime caches/checks."""
    warnings.filterwarnings("ignore", message="Error fetching version info", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["ALBUMENTATIONS_OFFLINE"] = "1"
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(repo_root, ".cache", "matplotlib"),
    )
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def resolve_split_list(values: Iterable[str]) -> List[str]:
    """Resolve export split names, expanding 'all' to all three splits."""
    splits = [value.lower() for value in values]
    if len(splits) == 1 and splits[0] == "all":
        return ["train", "val", "test"]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(
            f"Invalid export split(s): {invalid}. Allowed: {sorted(valid)} or ['all']."
        )
    return splits


def _resolve_save_dir(
    dataset_key: str,
    spec: DatasetSpec,
    save_root: Optional[str],
    fold: int,
    num_folds: int,
) -> str:
    if save_root:
        save_dir = os.path.join(save_root, f"dinov2_unet_{dataset_key}")
    else:
        save_dir = spec.default_save_dir
    if num_folds > 1:
        save_dir = os.path.join(save_dir, f"fold_{fold}")
    return save_dir


def collect_training_jobs(
    datasets: Iterable[str],
    data_dir: Optional[str],
    save_dir: Optional[str],
    joint_train: bool,
    fold: int,
    num_folds: int,
) -> List[TrainingJob]:
    """Resolve dataset names, paths, save directories, and joint jobs."""
    valid_datasets: List[Tuple[str, DatasetSpec, str]] = []
    for dataset in datasets:
        dataset_key = resolve_dataset_key(dataset)
        if dataset_key not in DATASET_SPECS:
            print(f"Skipping unsupported dataset: {dataset_key}")
            continue

        spec = DATASET_SPECS[dataset_key]
        if spec.requires_tifffile and tifffile is None:
            print(
                f"Skipping {dataset_key}: tifffile required. "
                "Install with `pip install tifffile imagecodecs`."
            )
            continue

        current_data_dir = resolve_data_dir_from_root(spec, data_dir)
        if not os.path.isdir(current_data_dir):
            print(f"[Error] Dataset directory not found: {current_data_dir}")
            continue
        valid_datasets.append((dataset_key, spec, current_data_dir))

    if not valid_datasets:
        return []

    if joint_train and len(valid_datasets) > 1:
        joint_key = "joint_" + "_".join(key for key, _, _ in valid_datasets)
        save_path = _resolve_save_dir(joint_key, valid_datasets[0][1], save_dir, fold, num_folds)
        joint_specs = [(spec.cls, path) for _, spec, path in valid_datasets]
        return [
            TrainingJob(
                dataset_key=joint_key,
                spec=valid_datasets[0][1],
                data_dir=valid_datasets[0][2],
                save_dir=save_path,
                joint_train_specs=joint_specs,
            )
        ]

    return [
        TrainingJob(
            dataset_key=key,
            spec=spec,
            data_dir=path,
            save_dir=_resolve_save_dir(key, spec, save_dir, fold, num_folds),
        )
        for key, spec, path in valid_datasets
    ]


def _build_config(args, job: TrainingJob) -> TrainConfig:
    return TrainConfig(
        dataset=job.dataset_key,
        data_dir=job.data_dir,
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
        save_dir=job.save_dir,
        seed=args.seed,
        aug_mode=args.aug_mode,
        decoder_dropout=args.decoder_dropout,
        use_tta=not args.no_tta,
        pretrained_type=args.pretrained_type,
        decoder_type=args.decoder_type,
        optimizer_strategy=args.optimizer_strategy,
        deep_supervision=not args.no_deep_supervision,
        grad_clip=args.grad_clip,
        track_metrics=args.track_metrics,
        aux_weight_scale=args.aux_weight_scale,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        fold=args.fold,
        num_folds=args.num_folds,
        joint_train_specs=job.joint_train_specs,
    )


def _export_job_masks(args, job: TrainingJob, cfg: TrainConfig, export_splits: List[str]) -> None:
    if job.dataset_key.startswith("joint_"):
        print(f"Skipping mask export for joint dataset {job.dataset_key}.")
        return

    export_dir = os.path.join(job.save_dir, "pred_masks")
    export_dataset_masks(
        dataset_key=job.dataset_key,
        data_dir=job.data_dir,
        save_dir=export_dir,
        backbone=args.backbone,
        out_indices=cfg.out_indices,
        img_size=cfg.img_size,
        freeze_blocks_until=args.freeze_blocks_until,
        decoder_dropout=args.decoder_dropout,
        num_workers=args.num_workers,
        pretrained_type=args.pretrained_type,
        decoder_type=args.decoder_type,
        splits=export_splits,
        checkpoint_path=os.path.join(job.save_dir, "best.pt"),
        device=None,
        seed=args.seed,
        fold_idx=cfg.fold,
        num_folds=cfg.num_folds,
        spec=job.spec,
    )
    print(f"Saved masks to {export_dir}")


def run_training_jobs(args, repo_root: str) -> None:
    """Run all training jobs described by parsed CLI arguments."""
    export_splits = resolve_split_list(args.export_splits)
    jobs = collect_training_jobs(
        datasets=args.datasets,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        joint_train=args.joint_train,
        fold=args.fold,
        num_folds=args.num_folds,
    )

    if not jobs:
        print("No valid datasets to process.")
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_root = os.path.join(repo_root, "log")
    os.makedirs(log_root, exist_ok=True)

    for job in jobs:
        dataset_log_dir = os.path.join(log_root, job.dataset_key)
        if args.num_folds > 1:
            dataset_log_dir = os.path.join(dataset_log_dir, f"fold_{args.fold}")
        os.makedirs(dataset_log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = TeeLogger(
            os.path.join(dataset_log_dir, f"{timestamp}.log"),
            original_stdout,
        )

        sys.stdout = logger
        sys.stderr = logger
        try:
            print(
                f"\n{'=' * 40}\n"
                f"Processing: {job.dataset_key}\n"
                f"Fold: {args.fold}/{args.num_folds}\n"
                f"{'=' * 40}\n"
            )
            cfg = _build_config(args, job)
            run_training(cfg, job.dataset_key, job.spec)

            if not args.no_export:
                try:
                    _export_job_masks(args, job, cfg, export_splits)
                except Exception as exc:
                    print(f"Error exporting masks for {job.dataset_key}: {exc}")
        except Exception as exc:
            print(f"Error training {job.dataset_key}: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            logger.close()
