@echo off
REM ============================================================================
REM DINOv2-UNet: 全自动实验管道脚本
REM 一键执行：训练 + 消融研究 + 失败分析 + 报告生成
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM 配置参数
set DATASET=kvasir
set DATA_DIR=../data/Kvasir-SEG
set SAVE_DIR=../runs/dinov2_unet_full_pipeline
set BATCH_SIZE=8
set EPOCHS=80
set NUM_SEEDS=2

echo.
echo ============================================================================
echo  DINOv2-UNet 全自动实验管道
echo ============================================================================
echo.

REM ============================================================================
REM 第1阶段：基础训练 + 指标追踪 + 梯度分析 + 失败分析
REM ============================================================================
echo [阶段 1/4] 开始基础训练 + 多维分析...
echo.
python train.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --save-dir %SAVE_DIR% ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --track-metrics ^
    --track-gradients ^
    --save-failure-analysis

if errorlevel 1 (
    echo [错误] 基础训练失败，中止流程！
    exit /b 1
)

echo [完成] 基础训练完毕 ✓
echo.

REM ============================================================================
REM 第2阶段：消融研究 - 超参数搜索 (2个轴)
REM ============================================================================
echo [阶段 2/4] 开始消融研究 (freeze_blocks_until 和 lr_ratio)...
echo.
python run_ablation_studies.py ^
    --dataset %DATASET% ^
    --data-dir %DATA_DIR% ^
    --axes freeze_blocks_until lr_ratio ^
    --freeze-blocks-until 0 3 6 9 12 ^
    --lr-ratio 0.001 0.01 0.1 1.0 ^
    --num-seeds %NUM_SEEDS% ^
    --save-dir ../ablation_results

if errorlevel 1 (
    echo [错误] 消融研究失败，中止流程！
    exit /b 1
)

echo [完成] 消融研究完毕 ✓
echo.

REM ============================================================================
REM 第3阶段：可视化 - 训练曲线 + 消融启发式 + 域分析
REM ============================================================================
echo [阶段 3/4] 生成可视化图表...
echo.
if exist %SAVE_DIR%\metrics_history.json (
    python -c "
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 加载指标历史
with open('%SAVE_DIR%\\metrics_history.json') as f:
    metrics = json.load(f)

# 绘制训练损失曲线
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
epochs = list(range(1, len(metrics['loss']) + 1))

axes[0,0].plot(epochs, metrics['loss'], 'b-', linewidth=2)
axes[0,0].set_title('Training Loss', fontsize=12)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].grid(True, alpha=0.3)

if 'dice' in metrics:
    axes[0,1].plot(epochs, metrics['dice'], 'g-', linewidth=2)
    axes[0,1].set_title('Dice Score', fontsize=12)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Dice')
    axes[0,1].grid(True, alpha=0.3)

if 'iou' in metrics:
    axes[1,0].plot(epochs, metrics['iou'], 'r-', linewidth=2)
    axes[1,0].set_title('IoU Score', fontsize=12)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('IoU')
    axes[1,0].grid(True, alpha=0.3)

axes[1,1].axis('off')
plt.tight_layout()
plt.savefig('%SAVE_DIR%\\training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('[✓] 生成训练曲线: %SAVE_DIR%\\training_curves.png')
"
) else (
    echo [提示] 未找到metrics_history.json，跳过可视化
)
echo.

REM ============================================================================
REM 第4阶段：报告生成
REM ============================================================================
echo [阶段 4/4] 生成实验报告...
echo.
python generate_experiment_report.py ^
    --results-dir %SAVE_DIR% ^
    --ablation-dir ../ablation_results ^
    --output-dir ../reports

if errorlevel 1 (
    echo [警告] 报告生成失败，但其他任务已完成
) else (
    echo [完成] 报告生成完毕 ✓
)
echo.

REM ============================================================================
REM 总结
REM ============================================================================
echo.
echo ============================================================================
echo  ✓ 全自动实验管道完成！
echo ============================================================================
echo.
echo 输出结果位置：
echo   • 训练模型：       %SAVE_DIR%\best.pt
echo   • 指标历史：       %SAVE_DIR%\metrics_history.json
echo   • 失败分析：       %SAVE_DIR%\failure_analysis.json
echo   • 失败样本集：     %SAVE_DIR%\failure_montages\
echo   • 消融结果：       ablation_results\summary.csv
echo   • 消融热力图：     ablation_results\heatmaps\
echo   • 实验报告：       reports\experiment_report.html
echo.
echo 次数提示：
echo   1. 用Excel打开 ablation_results\summary.csv 进行统计分析
echo   2. 在浏览器中打开 reports\experiment_report.html 查看完整报告
echo   3. 查看 %SAVE_DIR%\failure_montages\ 中的失败样本可视化
echo.
pause
