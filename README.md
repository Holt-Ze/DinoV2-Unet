# DINOv2-UNet Polyp Segmentation

DINOv2 encoder + UNet decoder for polyp segmentation, with partial fine-tuning, visualization, and analysis utilities.

## Highlights
- DINOv2 ViT backbone with partial fine-tuning (freeze early blocks, train later blocks)
- UNet-style decoder with multi-scale features
- Multi-dataset support: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS
- One-click training and mask export
- Analysis tools: attention maps, frozen vs trainable feature maps, t-SNE/PCA feature separability

## Requirements
- Python 3.9+
- PyTorch (CUDA optional)
- timm, albumentations, Pillow
- tifffile + imagecodecs (for TIFF datasets)
- scikit-learn + matplotlib (for t-SNE/PCA plots)

Example install:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations tifffile imagecodecs Pillow==9.5.0 scikit-learn matplotlib
```

## Data layout
Default data root is `./data` (override with `DATA_ROOT`).

- Kvasir-SEG:
  - `data/Kvasir-SEG/images`
  - `data/Kvasir-SEG/masks`
- CVC-ClinicDB:
  - `data/CVC-ClinicDB/Original`
  - `data/CVC-ClinicDB/Ground Truth`
- CVC-ColonDB:
  - `data/CVC-ColonDB/images` and `data/CVC-ColonDB/masks`
  - or `data/CVC-ColonDB/Original` and `data/CVC-ColonDB/Ground Truth`
- ETIS-LaribPolypDB:
  - `data/ETIS-LaribPolypDB/images`
  - `data/ETIS-LaribPolypDB/masks`

Use `--data-dir` to point to a custom dataset path.

Default output root is `./runs` (override with `RUNS_ROOT`).

## Train / Eval
Single dataset training:
```bash
python train.py --dataset kvasir --data-dir /path/to/Kvasir-SEG --save-dir runs/dinov2_unet_kvasir
```

Common args:
- `--img-size 448` (auto-adjust if not divisible by ViT patch size)
- `--batch-size 8`, `--epochs 80`
- `--lr 1e-3`, `--lr-backbone 1e-5`, `--weight-decay 0.01`
- `--freeze-blocks-until 6` (freeze early ViT blocks)
- `--aug-mode {strong,weak,none}`
- `--decoder-dropout 0.2`
- `--no-tta` (disable test-time horizontal flip)
- `--no-amp` (disable mixed precision)

Outputs:
- Best checkpoint: `<save_dir>/best.pt`
- Test visuals: `<save_dir>/vis_test/*_img.png|*_gt.png|*_pred.png`

## One-click training + export (all datasets)
```bash
python train_all.py --export-splits test
```
This trains each dataset and exports predicted masks to:
`<save_dir>/pred_masks/<split>`

## Analysis (attention, features, t-SNE)
Single dataset analysis:
```bash
python analyze.py --data kvasir --checkpoint runs/dinov2_unet_kvasir/best.pt --max-images 8
```

t-SNE + ResNet comparison:
```bash
python analyze.py --data kvasir --checkpoint runs/dinov2_unet_kvasir/best.pt \
  --tsne --tsne-samples-per-class 1500 --compare-backbone resnet50
```

Batch analysis for all datasets:
```bash
python analyze_all.py --datasets kvasir clinicdb colondb etis --tsne --compare-backbone resnet50
```

Analysis outputs:
- Attention overlays: `<save_dir>/analysis/attn/*_attn_b{block}.png`
- Frozen vs Trainable features: `<save_dir>/analysis/feature_compare/*_{frozen|trainable}_b{block}.png`
- t-SNE/PCA:
  - `<save_dir>/analysis/tsne/*_tsne.png`
  - `<save_dir>/analysis/tsne/*_points.csv`

## Project structure
- `train.py`: training entrypoint
- `train_all.py`: train all datasets + export
- `analyze.py`: analysis entrypoint
- `analyze_all.py`: batch analysis entrypoint
- `seg/` core package
  - `data.py`: datasets and default paths
  - `transforms.py`: augmentations + normalization
  - `models.py`: DINOv2 encoder + UNet decoder
  - `losses.py`, `metrics.py`: loss + metrics
  - `training.py`: training/eval loop + visualization
  - `analysis.py`: attention/features/t-SNE utilities

## Tips
- If you see tifffile/imagecodecs errors, install both packages.
- For quick sanity check, try: `--epochs 1 --batch-size 2 --img-size 256`.
- You can override dataset/output roots via environment:
  - `DATA_ROOT=/data`
  - `RUNS_ROOT=/checkpoints`
