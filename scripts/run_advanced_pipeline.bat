@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  DINOv2-UNet: Advanced Experiment Pipeline
echo ============================================================================
echo.

echo [Phase 1/5] Training with full analysis...
echo.

python train.py ^
    --dataset kvasir ^
    --data-dir ./data/Kvasir-SEG ^
    --save-dir runs/dinov2_unet_advanced ^
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
echo [Phase 2/5] Running comprehensive ablation studies...
echo.

python run_ablation_studies.py ^
    --dataset kvasir ^
    --data-dir ./data/Kvasir-SEG ^
    --axes freeze_blocks_until lr_ratio aug_mode decoder_type ^
    --freeze-blocks-until 0 3 6 9 12 ^
    --lr-ratio 0.001 0.01 0.1 1.0 ^
    --aug-mode strong weak none ^
    --decoder-type simple complex ^
    --num-seeds 2 ^
    --save-dir ablation_results

echo [Complete] Ablation studies finished!
echo.
echo [Phase 3/5] Running domain analysis...
echo.

if exist run_ablation_studies.py (
    python analyze_domains.py ^
        --model-dir runs/dinov2_unet_advanced/best.pt ^
        --source-dataset kvasir ^
        --target-datasets clinicdb colondb etis ^
        --output-dir domain_analysis
)

echo [Complete] Domain analysis finished!
echo.
echo [Phase 4/5] Generating comprehensive visualizations...
echo.

python -c "
import json
import matplotlib.pyplot as plt
from pathlib import Path

if Path('runs/dinov2_unet_advanced/metrics_history.json').exists():
    with open('runs/dinov2_unet_advanced/metrics_history.json') as f:
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
    plt.savefig('runs/dinov2_unet_advanced/training_curves.png', dpi=150, bbox_inches='tight')
    print('Visualizations saved!')
"

echo [Complete] Visualizations finished!
echo.
echo [Phase 5/5] Generating advanced reports...
echo.

python generate_experiment_report.py ^
    --results-dir runs/dinov2_unet_advanced ^
    --ablation-dir ablation_results ^
    --domain-dir domain_analysis ^
    --output-dir reports

echo.
echo ============================================================================
echo  SUCCESS! Advanced pipeline complete!
echo ============================================================================
echo.
echo Results saved to:
echo   - Models:       runs/dinov2_unet_advanced/best.pt
echo   - Metrics:      runs/dinov2_unet_advanced/metrics_history.json
echo   - Failures:     runs/dinov2_unet_advanced/failure_montages/
echo   - Ablations:    ablation_results/summary.csv
echo   - Domain:       domain_analysis/
echo   - Reports:      reports/experiment_report.html
echo.
pause
