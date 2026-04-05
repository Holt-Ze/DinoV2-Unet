# DINOv2-UNet Scripts

All-in-one experiment automation for DINOv2-UNet colonoscopy polyp segmentation.

## Quick Start

1. **Double-click** `START.bat` in Windows Explorer
2. **Choose** your experiment mode (1-4)
3. **Wait** for completion - results will be saved automatically

## Experiment Modes

### 1. Quick Pipeline (~5 min)
- Training with metrics tracking
- Output: Model + metrics JSON

```bash
# Equivalent command:
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG --track-metrics
```

### 2. Full Pipeline (~30 min)
- Training + metrics + gradient tracking + failure analysis
- Ablation studies (2 axes: freeze_blocks_until, lr_ratio)
- Automatic visualizations & report generation

```bash
# Runs:
python train.py ... --track-metrics --track-gradients --save-failure-analysis
python run_ablation_studies.py --axes freeze_blocks_until lr_ratio ...
python generate_experiment_report.py ...
```

### 3. Advanced Pipeline (~60 min)
- Complete analysis: training + extensive ablations + domain analysis
- Ablation axes: freeze_blocks_until, lr_ratio, aug_mode, decoder_type
- Cross-domain transfer prediction
- Publication-ready HTML reports

### 4. Manual Mode
Pick individual operations:
- Train with metrics tracking
- Run custom ablation studies
- Generate failure analysis
- Generate reports
- Run combined tools

## Output Locations

| Output | Location |
|--------|----------|
| Best model | `runs/dinov2_unet_*/best.pt` |
| Training metrics | `runs/*/metrics_history.json` |
| Failure analysis | `runs/*/failure_montages/` |
| Ablation results | `ablation_results/summary.csv` |
| Reports | `reports/experiment_report.html` |

## File Structure

```
scripts/
├── START.bat                    ← Start here!
├── run_quick_pipeline.bat       (Training only)
├── run_full_pipeline.bat        (Training + ablations)
├── run_advanced_pipeline.bat    (Everything)
├── run_pipeline.bat             (Legacy)
├── run_tools.bat                (Custom combinations)
├── README.md                    (This file)
└── BATCH_SCRIPTS_GUIDE.md       (Technical details)
```

## Notes

- All scripts require:
  - `data/Kvasir-SEG/` with images and masks
  - Python 3.10+ with PyTorch installed
  - Virtual environment activated

- Results are saved to project root:
  - `runs/` — training outputs
  - `ablation_results/` — hyperparameter studies
  - `reports/` — HTML reports

- Each script is standalone and can be:
  - Double-clicked directly
  - Called from command line

Example:
```bash
cd scripts
START.bat
```

## Troubleshooting

**"Python not found"** — Activate your virtual environment first:
```bash
dino\Scripts\activate
```

**"Encoding error"** — Ensure prompts in START.bat are pure ASCII (fixed in latest version)

**Script stops early** — Check error messages and logs in `log/` directory

## Advanced: Custom Commands

Open `START.bat` in Notepad to modify parameters:
- `set DATASET=kvasir` → Change dataset
- `set BATCH_SIZE=8` → Adjust batch size
- `set EPOCHS=80` → Training duration
- `set NUM_SEEDS=2` → Ablation study repetitions

## Support

For issues with:
- **Training**: See `log/` directory for detailed error messages
- **Data loading**: Verify dataset structure in `data/`
- **Visualization**: Requires matplotlib (in requirements.txt)
