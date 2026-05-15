# DINOv2-UNet: Colonoscopy Polyp Segmentation

A compact DINOv2 + U-Net style segmentation pipeline for colonoscopy polyp
datasets. The codebase now keeps the core training/evaluation path small while
retaining ablation support for validation experiments.

## Highlights

- DINOv2 ViT encoder with partial, full, or frozen fine-tuning.
- Simple U-shaped decoder by default, with an attention-gated decoder kept for ablation.
- Optional deep supervision with auxiliary decoder heads.
- Train, validate, test, export masks, K-fold, and joint multi-dataset training.
- `metrics_history.json` is written by default so ablation runs can aggregate results.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- timm, albumentations, Pillow, numpy
- tifffile + imagecodecs for TIFF-format datasets
- Optional: matplotlib for `run_ablation_studies.py --plot-heatmaps`

Install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Environment notes are in `environment.txt`.

## Dataset Layout

Place datasets under `./data` or set `DATA_ROOT`.

```text
data/
├── Kvasir-SEG/
│   ├── images/
│   └── masks/
├── CVC-ClinicDB/
│   ├── Original/
│   └── Ground Truth/
├── CVC-ColonDB/
│   ├── images/
│   └── masks/
└── ETIS/
    ├── images/
    └── masks/
```

`CVC-ClinicDB` and `CVC-ColonDB` also accept normalized `images/` + `masks/`
layouts.

## Training

Single dataset:

```bash
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG
```

Multiple datasets, trained sequentially:

```bash
python train.py --dataset kvasir clinicdb colondb etis --data-dir ./data --save-dir runs
```

Joint training with K-fold:

```bash
python train.py --dataset kvasir clinicdb --data-dir ./data \
    --joint-train --fold 0 --num-folds 5 --save-dir runs/cross_dataset_experiment
```

Common arguments:

| Argument | Default | Description |
|---|---:|---|
| `--img-size` | 448 | Input image size |
| `--batch-size` | 8 | Training batch size |
| `--epochs` | 80 | Max epochs |
| `--lr` | 1e-3 | Decoder learning rate |
| `--lr-backbone` | 1e-5 | Encoder learning rate |
| `--freeze-blocks-until` | 6 | Freeze ViT blocks before this index |
| `--optimizer-strategy` | partial_finetune | `partial_finetune`, `frozen_encoder`, `full_finetune` |
| `--pretrained-type` | dinov2 | `dinov2` or `imagenet_supervised` |
| `--decoder-type` | simple | `simple` or `complex` |
| `--aux-weight-scale` | 1.0 | Scale deep-supervision auxiliary losses |
| `--grad-clip` | 1.0 | Gradient clipping max norm |
| `--aug-mode` | strong | `strong`, `weak`, or `none` |
| `--no-export` | false | Skip post-training mask export |
| `--max-train-batches` | none | Limit train batches for smoke checks |
| `--max-eval-batches` | none | Limit val/test batches for smoke checks |

Outputs:

- `best.pt`
- `metrics_history.json`
- `vis_test/`
- `pred_masks/` unless `--no-export` is set
- logs under `log/<dataset>/`

## Evaluation

```bash
python test.py --dataset kvasir \
    --data-dir ./data/Kvasir-SEG \
    --checkpoint runs/dinov2_unet_kvasir/best.pt
```

Export masks during evaluation:

```bash
python test.py --dataset kvasir \
    --data-dir ./data/Kvasir-SEG \
    --checkpoint runs/dinov2_unet_kvasir/best.pt \
    --export-masks
```

## Ablation Studies

The ablation runner is intentionally separate from the main training path. It
runs `train.py`, reads `metrics_history.json`, and writes `summary.csv`.

Supported axes:

- `freeze_blocks_until`
- `lr_ratio`
- `aux_weight`
- `img_size`
- `warmup_epochs`
- `grad_clip`
- `optimizer_strategy`
- `decoder_type`
- `pretrained_type`

Example:

```bash
python run_ablation_studies.py --dataset kvasir \
    --data-dir ./data/Kvasir-SEG \
    --axes freeze_blocks_until lr_ratio \
    --freeze-blocks-until 0 3 6 9 12 \
    --lr-ratio 0.001 0.01 0.1 1.0 \
    --num-seeds 2 \
    --save-results ablation_results
```

Smoke-check an ablation command:

```bash
python run_ablation_studies.py --dataset kvasir \
    --data-dir data/Kvasir-SEG \
    --axes freeze_blocks_until lr_ratio \
    --freeze-blocks-until 6 \
    --lr-ratio 0.01 \
    --epochs 1 \
    --max-train-batches 1 \
    --max-eval-batches 1
```

Use `--plot-heatmaps` to generate simple 2D heatmaps when exactly two axes are
used and matplotlib is installed. Ablation runs skip mask export by default; add
`--export-masks` if predictions are needed for every run.

## Project Structure

```text
DinoV2-Unet/
├── train.py                  # Thin training CLI
├── test.py                   # Evaluation CLI
├── run_ablation_studies.py   # Experiment-layer ablation runner
├── seg/
│   ├── checkpoints.py        # Shared checkpoint cleanup/loading
│   ├── data.py               # Dataset specs and shared PolypDataset base
│   ├── inference.py          # Mask export
│   ├── losses.py             # BCE + Dice and deep supervision losses
│   ├── metrics.py            # Segmentation metrics
│   ├── metrics_tracking.py   # Per-epoch metrics JSON
│   ├── models.py             # DINOv2-UNet model variants
│   ├── pipeline.py           # Training job orchestration
│   ├── training.py           # Core train/eval loops
│   └── transforms.py         # Albumentations pipelines
└── data/
```

## Notes

The old report generation, domain analysis, failure analysis, profiling, and
gradient/activation tracking modules were removed to keep the project focused.
Use the ablation runner plus `summary.csv` for experiment validation.
