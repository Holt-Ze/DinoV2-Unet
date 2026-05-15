"""DINOv2-UNet evaluation entry point.

Evaluate a trained DINOv2-UNet checkpoint on the test split and report
all segmentation metrics described in the paper (Section 4.2).

Usage:
    # Evaluate with TTA (default)
    python test.py --dataset kvasir --data-dir ./data/Kvasir-SEG \\
        --checkpoint runs/dinov2_unet_kvasir/best.pt

    # Evaluate without TTA
    python test.py --dataset kvasir --data-dir ./data/Kvasir-SEG \\
        --checkpoint runs/dinov2_unet_kvasir/best.pt --no-tta

    # Evaluate and export predicted masks
    python test.py --dataset kvasir --data-dir ./data/Kvasir-SEG \\
        --checkpoint runs/dinov2_unet_kvasir/best.pt --export-masks
"""

import argparse
import os

DATASET_KEYS = ("clinicdb", "colondb", "etis", "kvasir")


def parse_args(args=None):
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DINOv2-UNet checkpoint."
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset key ({', '.join(DATASET_KEYS)}).",
    )
    parser.add_argument(
        "--data-dir", dest="data_dir", type=str, default=None,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze-blocks-until", type=int, default=6)
    parser.add_argument("--decoder-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable Test Time Augmentation.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run evaluation on (default: auto).")

    # Ablation support
    parser.add_argument(
        "--pretrained-type", type=str, default="dinov2",
        choices=["dinov2", "imagenet_supervised"],
    )
    parser.add_argument(
        "--decoder-type", type=str, default="simple",
        choices=["simple", "complex"],
    )

    # Export options
    parser.add_argument("--export-masks", action="store_true",
                        help="Export predicted binary masks to disk.")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory for exported masks (default: alongside checkpoint).")

    return parser.parse_args(args)


def main():
    """Main evaluation entry point."""
    import torch
    from torch.utils.data import DataLoader

    from seg.checkpoints import load_model_state
    from seg.data import DATASET_SPECS, resolve_data_dir_from_root, resolve_dataset_key
    from seg.inference import export_dataset_masks
    from seg.models import DinoV2UNet
    from seg.training import evaluate

    args = parse_args()
    dataset_key = resolve_dataset_key(args.dataset)

    if dataset_key not in DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. "
            f"Supported: {', '.join(sorted(DATASET_SPECS.keys()))}"
        )

    spec = DATASET_SPECS[dataset_key]
    data_dir = resolve_data_dir_from_root(spec, args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Dataset: {dataset_key}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"TTA: {'enabled' if not args.no_tta else 'disabled'}")

    # Build model (inference mode: no deep supervision)
    model = DinoV2UNet(
        backbone=args.backbone,
        out_indices=(2, 5, 8, 11),
        pretrained=False,
        freeze_blocks_until=args.freeze_blocks_until,
        num_classes=1,
        decoder_dropout=args.decoder_dropout,
        pretrained_type=args.pretrained_type,
        decoder_type=args.decoder_type,
        deep_supervision=False,
    ).to(device)

    # Load checkpoint
    missing, unexpected = load_model_state(
        model,
        args.checkpoint,
        drop_aux_heads=True,
        strict=False,
    )
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}")
    if missing:
        print(f"[warn] Missing keys: {missing}")

    # Build test dataset and loader
    test_ds = spec.cls(
        data_dir,
        "test",
        args.img_size,
        seed=args.seed,
        aug_mode="none",
        fold_idx=args.fold,
        num_folds=args.num_folds,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"\nEvaluating on {len(test_ds)} test samples...")

    # Run evaluation
    metrics = evaluate(model, test_loader, device, use_tta=not args.no_tta)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results on {dataset_key} (test split)")
    print(f"{'=' * 60}")
    print(f"  mDice:     {metrics['mDice']:.4f}")
    print(f"  mIoU:      {metrics['mIoU']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  Fw_beta:   {metrics['Fbeta_w']:.4f}")
    print(f"  S_alpha:   {metrics['s_alpha']:.4f}")
    print(f"  mE_xi:     {metrics['mE']:.4f}")
    print(f"  maxE_xi:   {metrics['mE_max']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"{'=' * 60}")

    # Optional mask export
    if args.export_masks:
        export_dir = args.export_dir or os.path.join(
            os.path.dirname(args.checkpoint), "pred_masks"
        )
        print(f"\nExporting masks to {export_dir}...")
        export_dataset_masks(
            dataset_key=dataset_key,
            data_dir=data_dir,
            save_dir=export_dir,
            backbone=args.backbone,
            out_indices=(2, 5, 8, 11),
            img_size=args.img_size,
            freeze_blocks_until=args.freeze_blocks_until,
            decoder_dropout=args.decoder_dropout,
            num_workers=args.num_workers,
            pretrained_type=args.pretrained_type,
            decoder_type=args.decoder_type,
            splits=["test"],
            checkpoint_path=args.checkpoint,
            device=device,
            seed=args.seed,
            fold_idx=args.fold,
            num_folds=args.num_folds,
        )
        print("Mask export complete.")


if __name__ == "__main__":
    main()
