@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  DINOv2-UNet: Quick Training Pipeline
echo ============================================================================
echo.

set DATASET=kvasir
set DATA_DIR=./data/Kvasir-SEG
set SAVE_DIR=runs/dinov2_unet_quick

echo [Phase 1/1] Starting training with metrics tracking...
echo.

python train.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --save-dir %SAVE_DIR% ^
    --batch-size 8 ^
    --epochs 80 ^
    --track-metrics

if errorlevel 1 (
    echo [ERROR] Training failed!
    exit /b 1
)

echo.
echo [SUCCESS] Quick pipeline complete!
echo.
echo Results:
echo   - Model:     %SAVE_DIR%\best.pt
echo   - Metrics:   %SAVE_DIR%\metrics_history.json
echo.
pause
