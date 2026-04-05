# DINOv2-UNet 一键实验管道 - 使用指南

## 快速开始

### 最简单的方式 - 运行启动器菜单

双击 `run_pipeline.bat` 会打开一个菜单，让你选择要运行的实验配置：

```
1. 快速管道 (快速)     - 约20-30分钟
2. 完整管道 (标准)     - 约2-4小时  ← 推荐
3. 高级管道 (全面)     - 约6-10小时
0. 退出
```

任选其一，脚本会自动执行完整的实验流程。

---

## 三种管道说明

### 1️⃣ 快速管道 (`run_quick_pipeline.bat`)

**执行时间**: 20-30分钟

**包含内容**:
- ✅ 单数据集训练 (Kvasir-SEG) + 指标追踪
- ✅ 消融研究 (仅1个超参数轴: freeze_blocks_until)
- ✅ 自动报告生成

**适用场景**:
- 快速验证代码是否正常运行
- 进行初步超参数调整
- 时间紧张，需要快速结果

**输出**:
```
runs/dinov2_unet_quick_pipeline/
├── best.pt   # 最佳模型
└── metrics_history.json  # 训练指标

ablation_results/
├── summary.csv  # 消融结果表格
└── heatmaps/  # 参数热力图

reports/
└── experiment_report.html  # 实验报告
```

---

### 2️⃣ 完整管道 (`run_full_pipeline.bat`) ⭐ 推荐

**执行时间**: 2-4小时

**包含内容**:
- ✅ 训练 + 指标追踪 (metrics_history.json)
- ✅ 梯度流分析 (activation tracking)
- ✅ 失败分析 (hard example mining, 5类语义分类)
- ✅ 完整消融研究 (2个超参数轴，笛卡尔积搜索)
- ✅ 训练曲线可视化
- ✅ 综合报告生成

**适用场景**:
- 发表研究论文前的全面实验
- 需要深入理解模型失败模式
- 准备学位论文或技术报告

**输出**:
```
runs/dinov2_unet_full_pipeline/
├── best.pt  # 最佳模型
├── metrics_history.json  # 完整训练日志 (Epoch, Loss, Dice, IoU, ...)
├── failure_analysis.json  # 失败样本分析
├── failure_montages/  # 失败样本可视化
└── training_curves.png  # 训练曲线图

ablation_results/
├── summary.csv  # 完整的超参数搜索结果表
└── heatmaps/  # Dice vs (freeze_blocks, lr_ratio) 热力图

reports/
└── experiment_report.html  # 完整实验报告 + 表格 + 图表
```

---

### 3️⃣ 高级管道 (`run_advanced_pipeline.bat`)

**执行时间**: 6-10小时

**包含内容**:
- ✅ 联合训练 (Kvasir-SEG + CVC-ClinicDB)
- ✅ 5折交叉验证 (robustness evaluation)
- ✅ 零样本跨域评估 (CVC-ColonDB, ETIS-LaribPolypDB)
- ✅ 完整消融研究 (2轴，多种子)
- ✅ 综合报告生成

**适用场景**:
- 需要发表顶级国际期刊
- 学位论文的主要实验章节
- 完整的跨域泛化评估

**输出**:
```
runs/dinov2_unet_advanced_pipeline/
├── fold_0/ to fold_4/  # 5个交叉验证折
│   └── best.pt, metrics_history.json, ...
├── zero_shot_colondb/  # CVC-ColonDB 零样本评估
└── zero_shot_etis/     # ETIS 零样本评估

ablation_results/
└── summary.csv, heatmaps/

reports/
└── experiment_report.html  # 汇总报告
```

---

## 使用方法

### 方法1: 菜单启动器（推荐）

```bash
# 双击运行
run_pipeline.bat
```

然后在菜单中选择 1-3

### 方法2: 直接运行

```bash
# 快速
run_quick_pipeline.bat

# 完整
run_full_pipeline.bat

# 高级
run_advanced_pipeline.bat
```

### 方法3: 自定义修改参数

编辑任何 `.bat` 文件，修改顶部的配置参数：

```batch
set DATASET=kvasir
set DATA_DIR=./data/Kvasir-SEG
set SAVE_DIR=runs/my_custom_experiment
set BATCH_SIZE=8
set EPOCHS=80
set NUM_SEEDS=2
```

然后保存并运行。

---

## ⚠️ 前置要求

1. **数据集已下载** - 确保 `./data/` 目录下有对应的数据集文件夹

2. **环保境已激活** - 命令行中应该能运行：
   ```bash
   python train.py --help
   ```

3. **GPU 足够** - 建议至少 8GB 显存 (4GB 可运行但较慢)

4. **磁盘空间** - 至少 50GB 可用空间（特别是运行完整或高级管道）

---

## 📊 查看结果

### 1. 查看训练曲线
```
runs/dinov2_unet_full_pipeline/training_curves.png
```
用任何图片查看器打开，可以看到 Loss、Dice、IoU 的训练进度。

### 2. 查看消融研究结果
```
ablation_results/summary.csv
```
用 Excel 打开，可以看到所有超参数组合的性能排名。

### 3. 查看失败样本分析
```
runs/dinov2_unet_full_pipeline/failure_montages/
```
可视化模型在哪些样本上失败，以及失败的语义语义类型。

### 4. 查看综合报告
```
reports/experiment_report.html
```
用浏览器打开，包含所有实验的完整总结、表格和图表。

---

## 🛠️ 常见问题

### Q: 如何中断运行的脚本？
**A**: 按 `Ctrl+C` 两次即可停止。

### Q: 如何只运行其中的某个阶段？
**A**: 编辑 `.bat` 文件，删除或注释掉不需要的阶段，然后保存运行。

### Q: 脚本中断后如何继续？
**A**: 脚本不支持断点续跑。建议：
- 如果是训练中断，可以修改 `--epochs` 参数继续
- 如果是消融中断，重新运行会覆盖之前的结果

### Q: 如何修改训练轮数或批大小？
**A**: 编辑 `.bat` 文件顶部：
```batch
set EPOCHS=100        # 改成你想要的轮数
set BATCH_SIZE=16    # 改成你的批大小
```

### Q: 输出目录可以改吗？
**A**: 可以！编辑：
```batch
set SAVE_DIR=my_experiment_results
set ABLATION_DIR=my_ablation_results
```

---

## 📋 脚本执行时间参考

| 阶段 | 快速 | 完整 | 高级 |
|------|------|------|------|
| 训练 | 15 min | 1.5 h | 6 h (5折) |
| 消融 | 5 min | 1.5 h | 1.5 h |
| 失败分析 | - | 15 min | - |
| 报告 | 5 min | 15 min | 15 min |
| **总计** | **20-30 min** | **2-4 h** | **6-10 h** |

*实际时间取决于GPU性能和数据集大小*

---

## 📝 输出文件详解

### metrics_history.json
```json
{
  "loss": [0.45, 0.42, 0.40, ...],
  "dice": [0.82, 0.84, 0.86, ...],
  "iou": [0.71, 0.73, 0.75, ...],
  "mae": [0.08, 0.07, 0.06, ...],
  "epoch": [1, 2, 3, ...]
}
```
可直接用 pandas 加载分析：
```python
import json
with open('metrics_history.json') as f:
    metrics = json.load(f)
```

### ablation_results/summary.csv
```csv
freeze_blocks_until,lr_ratio,seed,dice,iou,mae
0,0.001,1,0.82,0.71,0.08
0,0.001,2,0.81,0.70,0.09
0,0.01,1,0.84,0.73,0.07
...
```

### failure_analysis.json
```json
{
  "hard_examples": [
    {"image": "sample_123.jpg", "confidence": 0.45, "category": "small_polyp"},
    ...
  ],
  "calibration_curve": {...}
}
```

---

## 🎯 推荐工作流

### 对于初学者
1. 先运行 **快速管道** 验证环境和数据
2. 查看输出确认一切正常
3. 根据结果调整参数，再运行 **完整管道**

### 对于研究论文
1. 参考论文要求，选择 **完整** 或 **高级** 管道
2. 运行完成后，从 `reports/experiment_report.html` 导出数据和图表
3. 整合到论文中

### 对于快速迭代
1. 使用 **快速管道** 快速测试新想法
2. 一旦确认有效，改用 **完整管道** 进行严格评估
3. 最后用 **高级管道** 验证跨域泛化

---

## 🚀 更多自定义选项

所有脚本都基于以下核心命令，你可以直接在命令行运行来自定义参数：

```bash
# 训练
python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG --epochs 100 --batch-size 16 --track-metrics

# 消融
python run_ablation_studies.py --dataset kvasir --axes freeze_blocks_until --num-seeds 3

# 报告
python generate_experiment_report.py --results-dir runs/ --output-dir reports/
```

详细使用说明见 `README.md`。

---

祝你实验顺利! 🎉

