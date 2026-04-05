@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  DINOv2-UNet: Full Experiment Pipeline
echo ============================================================================
echo.

set DATASET=kvasir
set DATA_DIR=./data/Kvasir-SEG
set SAVE_DIR=runs/dinov2_unet_full
set NUM_SEEDS=2

echo [Phase 1/4] Training with metrics and failure analysis...
echo.

python train.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --save-dir %SAVE_DIR% ^
    --batch-size 8 ^
    --epochs 80 ^
    --track-metrics ^
    --track-gradients ^
    --save-failure-analysis

if errorlevel 1 (
    echo [ERROR] Training failed!
    exit /b 1
)

echo [Complete] Training finished!
echo.
echo [Phase 2/4] Running ablation studies...
echo.

python run_ablation_studies.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --axes freeze_blocks_until lr_ratio ^
    --freeze-blocks-until 0 3 6 9 12 ^
    --lr-ratio 0.001 0.01 0.1 1.0 ^
    --num-seeds %NUM_SEEDS% ^
    --save-dir ablation_results

if errorlevel 1 (
    echo [WARNING] Ablation failed, continuing...
)

echo [Complete] Ablation studies finished!
echo.
echo [Phase 3/4] Generating visualizations...
echo.

if exist %SAVE_DIR%\metrics_history.json (
    python -c "
import json
import matplotlib.pyplot as plt

with open('%SAVE_DIR%\metrics_history.json') as f:
    metrics = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
epochs = list(range(1, len(metrics['loss']) + 1))

axes[0,0].plot(epochs, metrics['loss'], 'b-', linewidth=2)
axes[0,0].set_title('Training Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].grid(True, alpha=0.3)

if 'dice' in metrics:
    axes[0,1].plot(epochs, metrics['dice'], 'g-', linewidth=2)
    axes[0,1].set_title('Dice Score')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].grid(True, alpha=0.3)

if 'iou' in metrics:
    axes[1,0].plot(epochs, metrics['iou'], 'r-', linewidth=2)
    axes[1,0].set_title('IoU Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].grid(True, alpha=0.3)

axes[1,1].axis('off')
plt.tight_layout()
plt.savefig('%SAVE_DIR%\training_curves.png', dpi=150, bbox_inches='tight')
print('Visualizations saved!')
"
)
)

echo [Complete] Visualizations finished!
echo.
echo [Phase 4/4] Generating reports...
echo.

python generate_experiment_report.py ^
    --results-dir %SAVE_DIR% ^
    --ablation-dir ablation_results ^
    --output-dir reports

echo.
echo ============================================================================
echo  SUCCESS! Full pipeline complete!
echo ============================================================================
echo.
echo Results saved to:
echo   - Models:       %SAVE_DIR%\best.pt
echo   - Metrics:      %SAVE_DIR%\metrics_history.json
echo   - Failures:     %SAVE_DIR%\failure_montages\
echo   - Ablations:    ablation_results\summary.csv
echo   - Reports:      reports\experiment_report.html
echo.
pause
