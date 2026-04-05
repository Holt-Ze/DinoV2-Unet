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

#### Key Training Arguments

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
| `--track-metrics` | False | Enable metrics history tracking (saves metrics_history.json per epoch) |
| `--track-gradients` | False | Enable gradient flow analysis (logs activation values) |
| `--save-failure-analysis` | False | Post-hoc hard example mining and semantic categorization |

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

## Advanced Analysis & Ablation Experiments

Beyond the base training pipeline, we provide a comprehensive experimental analysis infrastructure to support publication-level research:

### 1. Metrics Tracking – `--track-metrics`

Automatically logs training metrics per epoch (loss, Dice, IoU, MAE, etc.) into a JSON file:

```bash
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG \
    --track-metrics --save-dir runs/kvasir_with_metrics
```

Output: `runs/kvasir_with_metrics/metrics_history.json` — directly importable into pandas/Excel for statistical analysis.

### 2. Gradient Flow Analysis – `--track-gradients`

Enable gradient and activation value tracking during training to diagnose training dynamics:

```bash
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG \
    --track-gradients --save-dir runs/kvasir_gradient_analysis
```

### 3. Hard Example Mining & Failure Analysis – `--save-failure-analysis`

Post-hoc analysis identifying difficult samples and categorizing failure modes:

```bash
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG \
    --save-failure-analysis --save-dir runs/kvasir_failure_analysis
```

Outputs:
- `failure_analysis.json` — Confidence calibration curves and hard example rankings
- `failure_montages/` — Visual montages of failure modes (5 semantic categories)

### 4. Systematic Ablation Studies

Run multi-dimensional hyperparameter sweeps across 6 axes (freeze_blocks_until, lr_ratio, aug_mode, decoder_type, grad_clip, batch_size). Automatically aggregates results to CSV + heatmaps.

```bash
python run_ablation_studies.py --dataset kvasir \
    --axes freeze_blocks_until lr_ratio \
    --freeze-blocks-until 0 3 6 9 12 \
    --lr-ratio 0.001 0.01 0.1 1.0 \
    --num-seeds 2 --save-dir ablation_results
```

Outputs:
- `ablation_results/summary.csv` — Tabular results for external analysis
- `ablation_results/heatmaps/` — Publication-quality heatmaps

### 5. Domain Analysis & Zero-Shot Transfer Prediction

Analyze feature distribution shifts across datasets and predict zero-shot transfer performance:

```bash
python analyze_domains.py --model-dir runs/kvasir/best.pt \
    --source-dataset kvasir --target-datasets clinicdb colondb etis \
    --output-dir domain_analysis
```

Outputs:
- Wasserstein distance matrices
- t-SNE feature visualizations
- Transfer performance predictions

### 6. Automatic Report Generation

Generate publication-ready Markdown + HTML reports summarizing all experiments:

```bash
python generate_experiment_report.py --results-dir runs/ \
    --ablation-dir ablation_results/ \
    --domain-dir domain_analysis/ \
    --output-dir reports/
```

Output: `reports/experiment_report.html` — Complete summary with tables and inline plots.

## Pseudocode

For detailed algorithmic descriptions of the model architecture, deep supervision, training pipeline, and cross-dataset evaluation protocol, see **[PSEUDOCODE.md](PSEUDOCODE.md)**.

## Project Structure

```
DinoV2-Unet/
├── train.py              # Training entry point (with --track-metrics, --track-gradients, --save-failure-analysis)
├── test.py               # Evaluation entry point
├── requirements.txt      # Python dependencies
├── seg/                  # Core package
│   ├── __init__.py
│   ├── models.py         # DINOv2 encoder + U-shaped decoder + deep supervision
│   ├── data.py           # Dataset classes (Kvasir, ClinicDB, ColonDB, ETIS)
│   ├── training.py       # Training loop, optimizer, scheduler, evaluation
│   ├── losses.py         # BCE + Dice hybrid loss + deep supervision loss
│   ├── metrics.py        # Dice, IoU, MAE, Fw_β, S_α, E_ξ, Precision, Recall
│   ├── metrics_tracking.py # MetricsHistory, GradientTracker, ActivationTracker (NEW)
│   ├── inference.py      # Mask export utilities
│   ├── transforms.py     # Data augmentation pipelines
│   ├── profiling.py      # Params / FLOPs / FPS benchmarking
│   ├── paths.py          # Default path configuration
│   └── utils.py          # Reproducibility utilities
├── analysis/             # Experimental analysis modules (NEW)
│   ├── __init__.py
│   ├── visualization.py        # TrainingVisualizer, AblationVisualizer, DomainVisualizer
│   ├── failure_analysis.py     # Hard example mining, semantic categorization
│   └── domain_analysis.py      # Feature distribution, Wasserstein gap, zero-shot prediction
├── run_ablation_studies.py     # Multi-dimensional hyperparameter sweeps (NEW)
├── generate_experiment_report.py # Markdown + HTML report generation (NEW)
├── PSEUDOCODE.md               # Detailed algorithmic descriptions
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
