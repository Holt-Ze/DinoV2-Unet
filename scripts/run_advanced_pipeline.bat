@echo off
REM ============================================================================
REM DINOv2-UNet: 高级版全自动实验管道
REM 联合训练多数据集 + 跨域评估 + 完整分析
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo ============================================================================
echo  DINOv2-UNet 高级实验管道 (多数据集联合训练 + 跨域评估)
echo ============================================================================
echo.

REM 配置
set DATA_DIR=../data
set SAVE_DIR=../runs/dinov2_unet_advanced_pipeline
set NUM_FOLDS=5
set NUM_SEEDS=2

echo 配置信息：
echo   • 数据目录：%DATA_DIR%
echo   • 输出目录：%SAVE_DIR%
echo   • 交叉验证折数：%NUM_FOLDS%
echo   • 消融种子数：%NUM_SEEDS%
echo.

REM ============================================================================
REM 第1阶段：联合训练 Kvasir-SEG + CVC-ClinicDB (多折交叉验证)
REM ============================================================================
echo [阶段 1/4] 联合训练 Kvasir-SEG + CVC-ClinicDB (5折交叉验证)...
echo.

for /l %%i in (0,1,4) do (
    echo   [Fold %%i/%NUM_FOLDS%] 运行中...
    python train.py ^
        --dataset kvasir clinicdb ^
        --data-dir %DATA_DIR% ^
        --save-dir %SAVE_DIR%\fold_%%i ^
        --joint-train ^
        --fold %%i ^
        --num-folds %NUM_FOLDS% ^
        --track-metrics ^
        --track-gradients

    if errorlevel 1 (
        echo [错误] Fold %%i 失败
        exit /b 1
    )
)

echo [完成] 联合训练完毕 ✓
echo.

REM ============================================================================
REM 第2阶段：零样本跨域评估
REM ============================================================================
echo [阶段 2/4] 零样本跨域评估 (在 CVC-ColonDB 和 ETIS 上测试)...
echo.

set BEST_MODEL=%SAVE_DIR%\fold_0\best.pt

echo   评估数据集：CVC-ColonDB...
python test.py ^
    --dataset colondb ^
    --data-dir %DATA_DIR%\CVC-ColonDB ^
    --checkpoint %BEST_MODEL% ^
    --export-masks ^
    --save-dir %SAVE_DIR%\zero_shot_colondb

echo   评估数据集：ETIS-LaribPolypDB...
python test.py ^
    --dataset etis ^
    --data-dir %DATA_DIR%\ETIS ^
    --checkpoint %BEST_MODEL% ^
    --export-masks ^
    --save-dir %SAVE_DIR%\zero_shot_etis

echo [完成] 跨域评估完毕 ✓
echo.

REM ============================================================================
REM 第3阶段：消融研究 (多轴超参数搜索)
REM ============================================================================
echo [阶段 3/4] 消融研究 (freeze_blocks 和 lr_ratio)...
echo.

python run_ablation_studies.py ^
    --dataset kvasir ^
    --data-dir %DATA_DIR%\Kvasir-SEG ^
    --axes freeze_blocks_until lr_ratio ^
    --freeze-blocks-until 0 3 6 9 12 ^
    --lr-ratio 0.001 0.01 0.1 1.0 ^
    --num-seeds %NUM_SEEDS% ^
    --save-dir ../ablation_results

if errorlevel 1 (
    echo [警告] 消融研究失败，但继续下一步
)

echo [完成] 消融研究完毕 ✓
echo.

REM ============================================================================
REM 第4阶段：报告生成
REM ============================================================================
echo [阶段 4/4] 生成综合实验报告...
echo.

python generate_experiment_report.py ^
    --results-dir %SAVE_DIR% ^
    --ablation-dir ../ablation_results ^
    --output-dir ../reports

echo [完成] 报告生成完毕 ✓
echo.

REM ============================================================================
REM 总结
REM ============================================================================
echo.
echo ============================================================================
echo  ✓ 高级实验管道完成！
echo ============================================================================
echo.
echo 主要输出：
echo   • 联合训练模型 (5个折)： %SAVE_DIR%\fold_0-4\best.pt
echo   • 零样本评估：          %SAVE_DIR%\zero_shot_*\
echo   • 消融结果汇总：        ablation_results\summary.csv
echo   • 消融热力图：          ablation_results\heatmaps\
echo   • 综合报告：            reports\experiment_report.html
echo.
echo 建议后续步骤：
echo   1. 用Excel打开 ablation_results\summary.csv
echo   2. 在浏览器打开 reports\experiment_report.html 查看完整分析
echo   3. 比较 5个折的模型性能：
echo      - 查看各折 metrics_history.json 的Dice/IoU分布
echo      - 在 zero_shot_* 目录检查跨域泛化能力
echo.
pause
