import argparse
import os

from analyze import run_analysis
from seg.data import DATASET_SPECS, resolve_data_dir, resolve_dataset_key


def parse_args():
    parser = argparse.ArgumentParser(description="Run analysis on multiple datasets.")
    parser.add_argument("--datasets", nargs="*", default=["kvasir", "clinicdb", "colondb", "etis"])
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override root data directory (subfolders follow dataset defaults).")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Override root output directory for analysis outputs.")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt",
                        help="Checkpoint filename inside each dataset save dir.")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze-blocks-until", type=int, default=6)
    parser.add_argument("--attn-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--frozen-block", type=int, default=None)
    parser.add_argument("--trainable-block", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--skip-attn", action="store_true")
    parser.add_argument("--skip-feature-compare", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne-block", type=int, default=None)
    parser.add_argument("--tsne-samples-per-class", type=int, default=2000)
    parser.add_argument("--tsne-max-images", type=int, default=40)
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--compare-backbone", type=str, default=None)
    parser.add_argument("--compare-out-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_data_save_dirs(dataset_key: str, data_root: str, save_root: str):
    spec = DATASET_SPECS[dataset_key]
    if data_root:
        if spec.default_subdir:
            data_dir = os.path.abspath(os.path.join(data_root, spec.default_subdir))
        else:
            data_dir = os.path.abspath(data_root)
    else:
        data_dir = resolve_data_dir(spec, None)

    if save_root:
        save_dir = os.path.abspath(os.path.join(save_root, f"dinov2_unet_{dataset_key}"))
    else:
        save_dir = spec.default_save_dir
    return data_dir, save_dir


def main():
    args = parse_args()
    datasets = [resolve_dataset_key(x) for x in args.datasets]

    for dataset_key in datasets:
        if dataset_key not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset '{dataset_key}'.")

        data_dir, save_dir = resolve_data_save_dirs(dataset_key, args.data_root, args.save_root)
        checkpoint_path = None
        if args.checkpoint_name:
            candidate = os.path.join(save_dir, args.checkpoint_name)
            if os.path.exists(candidate):
                checkpoint_path = candidate
            else:
                print(f"[warn] Missing checkpoint for '{dataset_key}': {candidate}")

        run_args = argparse.Namespace(**vars(args))
        run_args.dataset = dataset_key
        run_args.data_dir = data_dir
        run_args.save_dir = save_dir
        run_args.checkpoint = checkpoint_path

        try:
            run_analysis(run_args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[warn] Skip dataset '{dataset_key}': {exc}")
            continue


if __name__ == "__main__":
    main()
