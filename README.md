# DINOv2-UNet: Colonoscopy Polyp Segmentation

**DINOv2-UNet: Transfer-Oriented Self-Supervised Vision Transformer for Robust and Real-Time Colonoscopy Polyp Segmentation**

> A transfer-oriented segmentation framework coupling a self-supervised DINOv2 ViT-B/14 encoder with a lightweight U-shaped decoder for accurate, real-time colonoscopy polyp segmentation.

## Highlights

- **DINOv2 ViT-B/14 encoder** with partial fine-tuning (freeze blocks 0–5, train blocks 6–11)
- **Streamlined U-shaped decoder** with progressive multi-scale fusion
- **Deep supervision** with auxiliary heads for improved gradient flow
- **Transfer-oriented training**: differential learning rates, cosine warm-up, gradient clipping
- **Experimental rigor**: N-Fold Cross Validation and Joint Training support
- **Zero-shot generalization**: Built-in cross-dataset evaluation pipeline
- **53.92 FPS** on a single GPU — exceeding real-time clinical requirements
- **State-of-the-art** on four public benchmarks: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0 with CUDA support
- timm, albumentations, Pillow, numpy
- tifffile + imagecodecs (for TIFF-format datasets)

### Installation

```bash
# Create virtual environment with uv
uv venv --python 3.10 dino
# Activate
# Windows:
dino\Scripts\activate
# Linux/macOS:
source dino/bin/activate

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
uv pip install -r requirements.txt
```

Or with pip:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Dataset Preparation

Download the datasets and organize them under `./data/` (or set `DATA_ROOT` environment variable):

```
data/
├── Kvasir-SEG/
│   ├── images/          # 1000 JPEG/PNG images
│   └── masks/           # Corresponding binary masks
├── CVC-ClinicDB/
│   ├── Original/        # 612 frames (TIFF or PNG)
│   └── Ground Truth/    # Corresponding masks
├── CVC-ColonDB/
│   ├── images/          # 380 colonoscopy images
│   └── masks/           # Corresponding masks
└── ETIS/
    ├── images/          # 196 PNG images
    └── masks/           # Corresponding masks
```

**Download links:**
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [CVC-ColonDB](http://vi.cvc.uab.es/colon-qa/cvccolondb/)
- [ETIS-LaribPolypDB](http://vi.cvc.uab.es/colon-qa/cvccolondb/)

## Training

Train on a single dataset:
```bash
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG --save-dir runs/dinov2_unet_kvasir
```

Train on all four datasets sequentially:
```bash
python train.py --dataset kvasir clinicdb colondb etis --data-dir ./data --save-dir runs
```

Joint training on multiple datasets with 5-Fold Cross Validation (e.g. fold 0):
```bash
python train.py --dataset kvasir clinicdb --data-dir ./data --joint-train --fold 0 --num-folds 5
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--img-size` | 448 | Input image resolution |
| `--batch-size` | 8 | Training batch size |
| `--epochs` | 80 | Maximum training epochs |
| `--lr` | 1e-3 | Decoder learning rate |
| `--lr-backbone` | 1e-5 | Encoder learning rate |
| `--freeze-blocks-until` | 6 | Number of frozen ViT blocks |
| `--aug-mode` | strong | Augmentation: `strong`, `weak`, `none` |
| `--grad-clip` | 1.0 | Gradient clipping max norm |
| `--patience` | 10 | Early stopping patience |

### Outputs

- Best checkpoint: `<save_dir>/best.pt`
- Test visualizations: `<save_dir>/vis_test/`
- Predicted masks: `<save_dir>/pred_masks/`
- Training logs: `log/<dataset>/`

## Cross-Dataset Evaluation Pipeline

To evaluate the zero-shot generalization capabilities, we provide automated pipeline scripts. The pipeline trains a joint model on Kvasir-SEG and CVC-ClinicDB (using 5-fold CV) and automatically evaluates it on unseen datasets like ETIS and CVC-ColonDB:

```bash
# Windows
run_cross_dataset.bat

# Linux / macOS
bash run_cross_dataset.sh
```

## Evaluation

Evaluate a trained checkpoint on the test split:
```bash
python test.py --dataset kvasir \
    --data-dir ./data/Kvasir-SEG \
    --checkpoint runs/dinov2_unet_kvasir/best.pt
```

Evaluate and export predicted masks:
```bash
python test.py --dataset kvasir \
    --data-dir ./data/Kvasir-SEG \
    --checkpoint runs/dinov2_unet_kvasir/best.pt \
    --export-masks
```

## Ablation Studies

The following ablation experiments are described in Section 4.9 of the paper:

### 1. Pre-training Initialization (Table 6)
```bash
# DINOv2 self-supervised (default)
python train.py --dataset kvasir --pretrained-type dinov2

# ImageNet supervised
python train.py --dataset kvasir --pretrained-type imagenet_supervised
```

### 2. Fine-tuning Strategy (Table 7)
```bash
# Partial fine-tuning (default): freeze blocks 0-5, train 6-11
python train.py --dataset kvasir --optimizer-strategy partial_finetune

# Frozen encoder (linear probing)
python train.py --dataset kvasir --optimizer-strategy frozen_encoder

# Full fine-tuning
python train.py --dataset kvasir --optimizer-strategy full_finetune
```

### 3. Decoder Design (Table 8)
```bash
# Streamlined decoder (default)
python train.py --dataset kvasir --decoder-type simple

# Complex decoder with Attention Gates
python train.py --dataset kvasir --decoder-type complex
```

## Pseudocode

### Algorithm 1: DINOv2-UNet Architecture

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DINOv2-UNet Forward Pass
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input : RGB image x ∈ R^(B × 3 × H × W)
Output: Segmentation logits ŷ ∈ R^(B × 1 × H × W)

  ── Encoder: DINOv2 ViT-B/14 ──
  1  Patch embedding: tokens ← PatchEmbed(x)       ▷ (B, N, D), D=768, patch=14
  2  Prepend [CLS] token, add positional embeddings
  3  FOR l = 0 TO 11 DO
  4  │  tokens ← TransformerBlock_l(tokens)
  5  │  IF l ∈ {2, 5, 8, 11} THEN
  6  │  │  feat ← tokens[:, 1:, :]                  ▷ discard CLS token
  7  │  │  feat ← Reshape(feat, (B, D, G_h, G_w))   ▷ G_h = H/14, G_w = W/14
  8  │  │  F_l  ← Conv1×1(feat, D → 256)            ▷ project to 256 channels
  9  │  END IF
  10 END FOR
  11 Features: {F_1, F_2, F_3, F_4}                 ▷ shallow → deep

  ── Decoder: Streamlined U-shaped ──
  12 Reverse features: {P_4, P_3, P_2, P_1} ← {F_4, F_3, F_2, F_1}
  13 out ← Lateral_4(P_4)                           ▷ Conv1×1, 256 → 256
  14 FOR i = 3 DOWNTO 1 DO
  15 │  out   ← Upsample(out, size = spatial(P_i))  ▷ bilinear interpolation
  16 │  skip  ← Lateral_i(P_i)                      ▷ Conv1×1, 256 → 256
  17 │  out   ← ConvBlock([out; skip])               ▷ concat → Conv3×3-BN-ReLU-Drop-Conv3×3-BN-ReLU
  18 END FOR

  ── Segmentation Head ──
  19 out ← Conv3×3(out, 256 → 128) → BN → ReLU
  20 ŷ   ← Conv1×1(out, 128 → 1)
  21 ŷ   ← Upsample(ŷ, size = (H, W))              ▷ restore to input resolution

RETURN ŷ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Algorithm 2: Deep Supervision

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Deep Supervision Loss (Training Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input : Decoder intermediate features {d_1, d_2, d_3}, main logits ŷ, ground truth y
Output: Total loss L_total

  λ ← {0.4, 0.3, 0.2}                              ▷ auxiliary loss weights

  1  L_main ← ComboLoss(ŷ, y)
  2  L_total ← L_main
  3  FOR k = 1 TO 3 DO
  4  │  aux_k ← Conv1×1(d_k, 256 → 1)              ▷ auxiliary segmentation head
  5  │  aux_k ← Upsample(aux_k, size = (H, W))
  6  │  L_total ← L_total + λ_k · ComboLoss(aux_k, y)
  7  END FOR

RETURN L_total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHERE ComboLoss = α · BCE(σ(ŷ), y) + β · DiceLoss(σ(ŷ), y),  α = β = 0.5
        DiceLoss  = 1 − (2·Σ(p·g) + ε) / (Σp² + Σg² + ε)
```

### Algorithm 3: Training Pipeline

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Transfer-Oriented Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hyperparameters:
  img_size = 448,  batch_size = 8,  max_epochs = 80
  lr_decoder = 1e-3,  lr_encoder = 1e-5              ▷ 100× differential
  weight_decay = 0.01,  warmup = 5 epochs,  patience = 10

  ── Initialization ──
  1  SET seed ← 42                                   ▷ reproducibility
  2  model ← DinoV2UNet(pretrained="dinov2", deep_supervision=True)
  3  FREEZE encoder blocks 0–5                       ▷ partial fine-tuning
  4  optimizer ← AdamW([
       {params: encoder_blocks_6_to_11, lr: 1e-5},
       {params: decoder + aux_heads,    lr: 1e-3}
     ], weight_decay=0.01)
  5  scheduler ← CosineAnnealingWithWarmUp(warmup=5, T_max=max_epochs, η_min=0.01·lr)
  6  scaler ← GradScaler()                           ▷ mixed precision (AMP)
  7  best_score ← 0,  wait ← 0

  ── Data Preparation ──
  8  D_train, D_val, D_test ← Split(Dataset, 80/10/10)
  9  Augment D_train with: HFlip, VFlip, Rot90, ElasticDistort, ColorJitter
  10 Normalize all splits to ImageNet μ, σ

  ── Training Loop ──
  11 FOR epoch = 1 TO max_epochs DO
  12 │  model.train()
  13 │  FOR (x, y, _) IN DataLoader(D_train) DO
  14 │  │  WITH autocast:
  15 │  │  │  outputs ← model(x)                      ▷ {"main", "aux_0", "aux_1", "aux_2"}
  16 │  │  │  loss ← DeepSupervisionLoss(outputs, y)
  17 │  │  scaler.scale(loss).backward()
  18 │  │  clip_grad_norm_(model.parameters(), max_norm=1.0)
  19 │  │  scaler.step(optimizer)
  20 │  │  scaler.update()
  21 │  │  scheduler.step()                            ▷ per-iteration update
  22 │  END FOR
  23 │
  24 │  ── Validation ──
  25 │  metrics ← Evaluate(model, D_val)
  26 │  score ← (metrics.mDice + metrics.mIoU) / 2    ▷ composite score
  27 │
  28 │  ── Early Stopping ──
  29 │  IF score > best_score THEN
  30 │  │  best_score ← score
  31 │  │  Save(model.state_dict, "best.pt")
  32 │  │  wait ← 0
  33 │  ELSE
  34 │  │  wait ← wait + 1
  35 │  │  IF wait ≥ patience THEN BREAK              ▷ early stopping
  36 │  END IF
  37 END FOR

  ── Final Evaluation ──
  38 model ← Load("best.pt")
  39 test_metrics ← Evaluate(model, D_test, TTA=True)  ▷ horizontal flip TTA
  40 Export predicted masks as binary PNGs

RETURN model, test_metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Algorithm 4: Cross-Dataset Zero-Shot Evaluation

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Joint Training + Zero-Shot Transfer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input : Source datasets S = {Kvasir-SEG, CVC-ClinicDB}
        Target datasets T = {ETIS, CVC-ColonDB}
Output: Zero-shot metrics on each target dataset

  ── Phase 1: Joint Training ──
  1  D_joint ← ConcatDataset(S)                     ▷ merge source datasets
  2  model ← Train(D_joint, K-Fold CV, fold=k)      ▷ Algorithm 3

  ── Phase 2: Zero-Shot Evaluation ──
  3  model ← Load("best.pt", deep_supervision=False)
  4  FOR each dataset D_t ∈ T DO
  5  │  D_t uses ALL samples (no train/val split)
  6  │  FOR (x, y, fname) IN DataLoader(D_t) DO
  7  │  │  ŷ ← σ(model(x)["main"])                  ▷ sigmoid activation
  8  │  │  mask ← (ŷ > 0.5).float()                  ▷ binary thresholding
  9  │  │  Accumulate metrics(mask, y)
  10 │  │  Save mask as PNG
  11 │  END FOR
  12 │  Report {mDice, mIoU, MAE, Fw_β, S_α, E_ξ, ...} for D_t
  13 END FOR

RETURN metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Project Structure

```
DinoV2-Unet/
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
├── requirements.txt      # Python dependencies
├── seg/                  # Core package
│   ├── __init__.py
│   ├── models.py         # DINOv2 encoder + U-shaped decoder + deep supervision
│   ├── data.py           # Dataset classes (Kvasir, ClinicDB, ColonDB, ETIS)
│   ├── training.py       # Training loop, optimizer, scheduler, evaluation
│   ├── losses.py         # BCE + Dice hybrid loss + deep supervision loss
│   ├── metrics.py        # Dice, IoU, MAE, Fw_β, S_α, E_ξ, Precision, Recall
│   ├── inference.py      # Mask export utilities
│   ├── transforms.py     # Data augmentation pipelines
│   ├── profiling.py      # Params / FLOPs / FPS benchmarking
│   ├── paths.py          # Default path configuration
│   └── utils.py          # Reproducibility utilities
└── data/                 # Dataset directory (not tracked)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025dinov2unet,
  title     = {DINOv2-UNet: Transfer-Oriented Self-Supervised Vision Transformer
               for Robust and Real-Time Colonoscopy Polyp Segmentation},
  author    = {Zehao Liu and Jia Yu and Zhenyu Song and Zenan Lu and
               Lixing Tan and Guodong Hu and Chengfei Cai},
  year      = {2025},
  journal   = {submitted},
}
```

## Acknowledgements

This work was supported by the Jiangsu Taizhou University Research Start-up Fund for High-Level Talents (No. TZXYQD2025B101) and Anhui Provincial Special Program for Clinical Medical Research Translation (No. 202527c10020063).

## License

This project is for academic research purposes.
