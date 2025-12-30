import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seg.data import DATASET_ALIASES, DATASET_SPECS, resolve_data_dir, resolve_dataset_key
from seg.models import DinoV2UNet
from seg.analysis import (
    build_cnn_features,
    denorm_image,
    ensure_dir,
    extract_vit_attention_maps,
    extract_vit_block_features,
    feature_to_heatmap,
    gather_cnn_patch_features,
    gather_vit_patch_features,
    load_checkpoint,
    normalize_map,
    reduce_pca,
    reduce_tsne,
    save_overlay_image,
    save_points_csv,
    save_scatter,
)


def parse_args():
    dataset_choices = sorted(set(list(DATASET_SPECS.keys()) + list(DATASET_ALIASES.keys())))
    parser = argparse.ArgumentParser(description="Analyze DINOv2-UNet attention and feature separability.")
    parser.add_argument("--data", "--dataset", dest="dataset", required=True, choices=dataset_choices,
                        help=f"Dataset key ({', '.join(sorted(DATASET_SPECS.keys()))}).")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=None,
                        help="Override dataset root directory.")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default=None,
                        help="Directory to store analysis outputs.")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--freeze-blocks-until", type=int, default=6)
    parser.add_argument("--attn-blocks", type=int, nargs="+", default=None,
                        help="ViT block indices to export attention maps from.")
    parser.add_argument("--frozen-block", type=int, default=None,
                        help="Block index treated as frozen for feature comparison.")
    parser.add_argument("--trainable-block", type=int, default=None,
                        help="Block index treated as trainable for feature comparison.")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--skip-attn", action="store_true")
    parser.add_argument("--skip-feature-compare", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne-block", type=int, default=None)
    parser.add_argument("--tsne-samples-per-class", type=int, default=2000)
    parser.add_argument("--tsne-max-images", type=int, default=40)
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--compare-backbone", type=str, default=None,
                        help="Optional CNN backbone for t-SNE comparison (e.g., resnet50).")
    parser.add_argument("--compare-out-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_analysis(args):
    dataset_key = resolve_dataset_key(args.dataset)
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported keys: {', '.join(sorted(DATASET_SPECS.keys()))}")
    spec = DATASET_SPECS[dataset_key]

    data_dir = resolve_data_dir(spec, args.data_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else spec.default_save_dir
    analysis_root = os.path.join(save_dir, "analysis")
    attn_dir = os.path.join(analysis_root, "attn")
    feat_dir = os.path.join(analysis_root, "feature_compare")
    tsne_dir = os.path.join(analysis_root, "tsne")
    ensure_dir(analysis_root)
    ensure_dir(attn_dir)
    ensure_dir(feat_dir)
    ensure_dir(tsne_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = DinoV2UNet(
        backbone=args.backbone,
        out_indices=(2, 5, 8, 11),
        pretrained=True,
        freeze_blocks_until=args.freeze_blocks_until,
        num_classes=1,
    ).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    dataset = spec.cls(data_dir, args.split, args.img_size, seed=args.seed, aug_mode="none")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vit_model = model.encoder.model
    num_blocks = len(vit_model.blocks)
    attn_blocks = args.attn_blocks or [num_blocks - 1]

    frozen_block = args.frozen_block
    if frozen_block is None:
        frozen_block = args.freeze_blocks_until - 1
    trainable_block = args.trainable_block
    if trainable_block is None:
        trainable_block = args.freeze_blocks_until

    valid_blocks = set(range(num_blocks))
    if frozen_block not in valid_blocks:
        frozen_block = None
    if trainable_block not in valid_blocks:
        trainable_block = None

    seen = 0
    for imgs, msks, names in loader:
        if args.max_images and seen >= args.max_images:
            break
        imgs = imgs.to(device)
        msks = msks.to(device)
        img_denorm = denorm_image(imgs, dataset.mean, dataset.std).cpu()

        if not args.skip_attn:
            attn_maps, _, _ = extract_vit_attention_maps(vit_model, imgs, attn_blocks)
            for b in range(imgs.size(0)):
                base_name = os.path.splitext(os.path.basename(names[b]))[0]
                for block_idx, attn in attn_maps.items():
                    heat = normalize_map(attn[b:b + 1])
                    heat = F.interpolate(heat, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
                    out_path = os.path.join(attn_dir, f"{base_name}_attn_b{block_idx}.png")
                    save_overlay_image(img_denorm[b], heat.squeeze(0), out_path, mask=msks[b].cpu())

        if not args.skip_feature_compare:
            blocks = [b for b in [frozen_block, trainable_block] if b is not None]
            if blocks:
                feats, _, _ = extract_vit_block_features(vit_model, imgs, blocks)
                for b in range(imgs.size(0)):
                    base_name = os.path.splitext(os.path.basename(names[b]))[0]
                    for block_idx, fm in feats.items():
                        heat = feature_to_heatmap(fm[b:b + 1])
                        heat = F.interpolate(heat, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
                        tag = "frozen" if block_idx == frozen_block else "trainable"
                        out_path = os.path.join(feat_dir, f"{base_name}_{tag}_b{block_idx}.png")
                        save_overlay_image(img_denorm[b], heat.squeeze(0), out_path, mask=msks[b].cpu())

        seen += 1

    if args.tsne:
        tsne_block = args.tsne_block if args.tsne_block is not None else (num_blocks - 1)
        tsne_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        vit_feats, vit_labels = gather_vit_patch_features(
            vit_model,
            tsne_loader,
            device,
            tsne_block,
            args.tsne_samples_per_class,
            args.tsne_max_images,
            args.seed,
        )
        if vit_feats.shape[0] > 0:
            points = reduce_tsne(vit_feats, seed=args.seed, perplexity=args.tsne_perplexity)
            if points is None:
                points = reduce_pca(vit_feats)
            name = "dinov2"
            out_csv = os.path.join(tsne_dir, f"{name}_points.csv")
            save_points_csv(points, vit_labels, out_csv, name)
            out_png = os.path.join(tsne_dir, f"{name}_tsne.png")
            if not save_scatter(points, vit_labels, out_png, f"{name} t-SNE"):
                print("matplotlib not available, saved CSV instead.")

        if args.compare_backbone:
            cnn = build_cnn_features(args.compare_backbone, pretrained=True).to(device)
            cnn_feats, cnn_labels = gather_cnn_patch_features(
                cnn,
                tsne_loader,
                device,
                args.compare_out_index,
                args.tsne_samples_per_class,
                args.tsne_max_images,
                args.seed,
            )
            if cnn_feats.shape[0] > 0:
                points = reduce_tsne(cnn_feats, seed=args.seed, perplexity=args.tsne_perplexity)
                if points is None:
                    points = reduce_pca(cnn_feats)
                name = args.compare_backbone
                out_csv = os.path.join(tsne_dir, f"{name}_points.csv")
                save_points_csv(points, cnn_labels, out_csv, name)
                out_png = os.path.join(tsne_dir, f"{name}_tsne.png")
                if not save_scatter(points, cnn_labels, out_png, f"{name} t-SNE"):
                    print("matplotlib not available, saved CSV instead.")


def main():
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
