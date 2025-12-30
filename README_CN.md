# DINOv2-UNet 息肉分割

基于 DINOv2 编码器 + UNet 解码器的息肉分割框架，支持 Partial Fine-Tuning、多数据集训练与可视化分析。

## 亮点
- DINOv2 ViT 主干 + UNet 解码器
- Partial Fine-Tuning（冻结前若干 block，训练后若干 block）
- 支持多数据集：Kvasir-SEG、CVC-ClinicDB、CVC-ColonDB、ETIS
- 一键训练 + 掩膜导出
- 注意力热力图 / Frozen vs Trainable 特征对比 / t-SNE/PCA 可视化

## 环境依赖
- Python 3.9+
- PyTorch（可选 CUDA）
- timm、albumentations、Pillow
- tifffile + imagecodecs（处理 TIFF 数据集）
- scikit-learn + matplotlib（t-SNE/PCA 绘图）

示例安装：
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations tifffile imagecodecs Pillow==9.5.0 scikit-learn matplotlib
```

## 数据集结构
默认数据根目录为 `./data`（可用 `DATA_ROOT` 覆盖）。

- Kvasir-SEG：
  - `data/Kvasir-SEG/images`
  - `data/Kvasir-SEG/masks`
- CVC-ClinicDB：
  - `data/CVC-ClinicDB/Original`
  - `data/CVC-ClinicDB/Ground Truth`
- CVC-ColonDB：
  - `data/CVC-ColonDB/images` 和 `data/CVC-ColonDB/masks`
  - 或 `data/CVC-ColonDB/Original` 和 `data/CVC-ColonDB/Ground Truth`
- ETIS-LaribPolypDB：
  - `data/ETIS-LaribPolypDB/images`
  - `data/ETIS-LaribPolypDB/masks`

如目录不同，可用 `--data-dir` 指定。

默认输出根目录为 `./runs`（可用 `RUNS_ROOT` 覆盖）。

## 训练 / 评估
单数据集训练：
```bash
python train.py --dataset kvasir --data-dir /path/to/Kvasir-SEG --save-dir runs/dinov2_unet_kvasir
```

常用参数：
- `--img-size 448`（若不整除 patch size 会自动上调）
- `--batch-size 8`，`--epochs 80`
- `--lr 1e-3`，`--lr-backbone 1e-5`，`--weight-decay 0.01`
- `--freeze-blocks-until 6`（冻结前若干 ViT block）
- `--aug-mode {strong,weak,none}`
- `--decoder-dropout 0.2`
- `--no-tta`（关闭测试时水平翻转 TTA）
- `--no-amp`（关闭混合精度）

输出：
- 最佳权重：`<save_dir>/best.pt`
- 测试可视化：`<save_dir>/vis_test/*_img.png|*_gt.png|*_pred.png`

## 一键训练 + 掩膜导出（四个数据集）
```bash
python train_all.py --export-splits test
```
输出路径：
`<save_dir>/pred_masks/<split>`

## 分析与可视化
单数据集分析：
```bash
python analyze.py --data kvasir --checkpoint runs/dinov2_unet_kvasir/best.pt --max-images 8
```

t-SNE + ResNet 对比：
```bash
python analyze.py --data kvasir --checkpoint runs/dinov2_unet_kvasir/best.pt \
  --tsne --tsne-samples-per-class 1500 --compare-backbone resnet50
```

四个数据集一键分析：
```bash
python analyze_all.py --datasets kvasir clinicdb colondb etis --tsne --compare-backbone resnet50
```

分析输出位置：
- 注意力热力图：`<save_dir>/analysis/attn/*_attn_b{block}.png`
- Frozen vs Trainable 特征：`<save_dir>/analysis/feature_compare/*_{frozen|trainable}_b{block}.png`
- t-SNE/PCA：
  - `<save_dir>/analysis/tsne/*_tsne.png`
  - `<save_dir>/analysis/tsne/*_points.csv`

## 项目结构
- `train.py`：训练入口
- `train_all.py`：多数据集训练 + 导出
- `analyze.py`：分析入口
- `analyze_all.py`：批量分析入口
- `seg/` 核心代码
  - `data.py`：数据集与路径
  - `transforms.py`：增强与归一化
  - `models.py`：DINOv2 编码器 + UNet 解码器
  - `losses.py`、`metrics.py`：损失与指标
  - `training.py`：训练/评估/可视化
  - `analysis.py`：注意力/特征/t-SNE 工具

## 小贴士
- tifffile/imagecodecs 报错时请安装这两个包。
- 快速验证流程：`--epochs 1 --batch-size 2 --img-size 256`。
- 可用环境变量覆盖目录：
  - `DATA_ROOT=/data`
  - `RUNS_ROOT=/checkpoints`
