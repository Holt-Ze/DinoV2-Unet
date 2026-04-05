@echo off
REM ============================================================================
REM DINOv2-UNet: 简化版全自动实验管道
REM 仅运行：训练 + 消融研究 + 报告生成 (更快)
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo ============================================================================
echo  DINOv2-UNet 快速实验管道 (无失败分析)
echo ============================================================================
echo.

REM 配置
set DATASET=kvasir
set DATA_DIR=../data/Kvasir-SEG
set SAVE_DIR=../runs/dinov2_unet_quick_pipeline
set BATCH_SIZE=8
set EPOCHS=80

REM 训练
echo [1/3] 训练模型...
python train.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --save-dir %SAVE_DIR% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --track-metrics

if errorlevel 1 (
    echo [错误] 训练失败！
    exit /b 1
)

echo [完成] ✓
echo.

REM 消融研究 (仅1个轴，加速)
echo [2/3] 消融研究 (freeze_blocks_until)...
python run_ablation_studies.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --axes freeze_blocks_until ^
    --freeze-blocks-until 0 6 12 ^
    --num-seeds 1 ^
    --save-dir ../ablation_results

if errorlevel 1 (
    echo [警告] 消融研究失败，继续下一步
)

echo [完成] ✓
echo.

REM 报告生成
echo [3/3] 生成报告...
python generate_experiment_report.py ^
    --results-dir %SAVE_DIR% ^
    --ablation-dir ../ablation_results ^
    --output-dir ../reports

echo [完成] ✓
echo.
echo ============================================================================
echo  快速管道完成！
echo ============================================================================
echo.
echo 结果位置：
echo   • 模型：       %SAVE_DIR%\best.pt
echo   • 指标：       %SAVE_DIR%\metrics_history.json
echo   • 报告：       reports\experiment_report.html
echo.
pause
