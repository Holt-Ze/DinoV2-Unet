# DINOv2-UNet 一键脚本工具集

这个文件夹包含所有一键实验脚本，用于快速运行各种实验管道。

## 🚀 快速开始

### 1. 最简单的方式 - 运行总启动菜单

**双击：`START.bat`**

这会打开一个交互式菜单，让你选择要运行的操作。

## 📋 脚本文件说明

| 文件 | 说明 | 执行时间 |
|------|------|--------|
| **START.bat** ⭐ | 总启动菜单，集合所有功能 | 1 分钟 |
| **run_quick_pipeline.bat** | 快速模式：训练+简化消融+报告 | 20-30分钟 |
| **run_full_pipeline.bat** | 完整模式(推荐)：训练+完整消融+失败分析+报告 | 2-4小时 |
| **run_advanced_pipeline.bat** | 高级模式：联合训练+交叉验证+跨域评估 | 6-10小时 |
| **run_pipeline.bat** | 管道选择器：选择要运行的管道 | - |
| **run_tools.bat** | 辅助工具：环境检查、数据验证、日志清理 | 1分钟 |
| **BATCH_SCRIPTS_GUIDE.md** | 详细使用指南 | - |

## 🎯 如何使用

### 方式1：菜单启动（推荐）
```bash
# 双击这个文件
START.bat
```
然后在菜单中选择 1-9 进行各种操作。

### 方式2：直接运行单个脚本
```bash
# 快速模式
run_quick_pipeline.bat

# 完整模式
run_full_pipeline.bat

# 高级模式
run_advanced_pipeline.bat

# 工具箱
run_tools.bat
```

### 方式3：命令行运行
```bash
cd scripts/
START.bat
```

## 📌 重要提示

1. **必须在项目根目录运行** — 脚本使用相对路径 `../data/`, `../runs/` 等
   - ✓ 正确：在 `d:\DinoV2-Unet\` 文件夹中双击 `scripts\START.bat`
   - ✗ 错误：在 `d:\DinoV2-Unet\scripts\` 文件夹中双击 `START.bat`

2. **数据集位置** — 脚本期望数据集在项目根目录的 `data/` 文件夹中
   ```
   d:\DinoV2-Unet\
   ├── scripts/          ← 脚本都在这里
   ├── data/             ← 数据集在这里
   ├── seg/
   ├── train.py
   └── ...
   ```

3. **输出位置** — 结果会保存到项目根目录（与脚本同级）
   ```
   d:\DinoV2-Unet\
   ├── runs/             ← 训练结果
   ├── ablation_results/ ← 消融结果
   └── reports/          ← 实验报告
   ```

## 🔧 前置要求

- Python 3.10+ 和 PyTorch 2.0+ 已安装
- 数据集已下载到 `data/` 目录
- 足够的GPU显存 (建议 8GB+)
- 充足的磁盘空间 (50GB+ 用于完整模式)

## 📖 查看详细文档

- **使用指南**：打开 `BATCH_SCRIPTS_GUIDE.md`
- **项目说明**：回到项目根目录打开 `README.md`

## 💡 常见问题

**Q: 脚本一直无法找到数据？**
- A: 确保从项目根目录运行脚本，不是从 scripts 文件夹内。

**Q: 如何自定义参数？**
- A: 用记事本打开相应的 .bat 文件，修改顶部的参数，保存后运行。

**Q: 脚本中断了怎么办？**
- A: 脚本不支持断点续跑。重新运行会覆盖之前的结果。

**Q: 如何查看结果？**
- A:
  - 训练曲线：`runs/*/training_curves.png`
  - 消融结果：用 Excel 打开 `ablation_results/summary.csv`
  - 完整报告：用浏览器打开 `reports/experiment_report.html`

## 🎓 推荐工作流

1. **快速验证**
   ```bash
   START.bat  # 选择 1: 快速管道
   ```
   用来检查环境和数据是否正常（20-30分钟）

2. **完整实验**（推荐用于论文）
   ```bash
   START.bat  # 选择 2: 完整管道
   ```
   进行全面的实验分析（2-4小时）

3. **最终评估**
   ```bash
   START.bat  # 选择 3: 高级管道
   ```
   多数据集联合训练和跨域泛化评估（6-10小时）

---

祝你实验顺利! 🎉

有问题？打开 `BATCH_SCRIPTS_GUIDE.md` 查看完整文档。
