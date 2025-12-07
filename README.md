# DINOv2-UNet Polyp Segmentation

轻量 README，帮助你的搭档快速跑通训练/评估流程。

## 环境准备
- Python 3.9+，建议使用虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- 安装依赖（CUDA 版 torch 请按自己的 CUDA 版本替换）：  
  ```bash
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
  pip install timm albumentations tifffile imagecodecs Pillow==9.5.0
  ```
- 无 GPU 也可以跑，但训练会很慢；开启 AMP 默认需要 GPU，可通过 `--no-amp` 关闭。

## 数据集放置
- 默认数据根目录：`./data`（可通过环境变量 `DATA_ROOT` 覆盖）。各数据集默认子目录：
  - Kvasir-SEG：`data/Kvasir-SEG/images` 与 `data/Kvasir-SEG/masks`
  - CVC-ClinicDB：`data/CVC-ClinicDB/Original` 与 `data/CVC-ClinicDB/Ground Truth`
  - CVC-ColonDB：`data/CVC-ColonDB/images|masks` 或 `Original|Ground Truth`
  - ETIS-LaribPolypDB：`data/ETIS-LaribPolypDB/images` 与 `data/ETIS-LaribPolypDB/masks`
- 如目录不同，使用 `--data-dir` 显式指定。
- 输出根目录默认 `./runs`（可通过 `RUNS_ROOT` 覆盖）。

## 训练/评估
单机训练与测试共用同一入口：
```bash
python train.py --dataset kvasir --data-dir /path/to/Kvasir-SEG --save-dir runs/dinov2_unet_kvasir
```
- `--dataset` 支持：`kvasir`, `clinicdb`, `colondb`, `etis`（别名：`kvasir-seg`, `clinic`, `cvc-clinicdb`, `colon`, `cvc-colondb`, `etis-larib`）。
- 常用可调参数：
  - `--img-size 448`：输入分辨率，若非 patch size 整除会自动上调。
  - `--batch-size 8`，`--epochs 80`，`--lr 1e-3`，`--lr-backbone 1e-5`，`--weight-decay 0.01`
  - `--freeze-blocks-until 6`：冻结 ViT 前若干块。
  - `--aug-mode {strong,weak,none}`：训练增强强度；验证/测试固定无增强。
  - `--decoder-dropout 0.2`：UNet 解码器 dropout。
  - `--no-tta`：关闭测试阶段水平翻转 TTA。
  - `--no-amp`：关闭自动混合精度。
  - `--seed 42`，`--patience 10`
- 训练完成后：
  - 最优权重：`<save_dir>/best.pt`
  - 测试可视化：`<save_dir>/vis_test/*_img.png|*_gt.png|*_pred.png`

## 项目结构
- `train.py`：入口脚本（参数解析、调度训练）
- `seg/`：核心模块
  - `data.py`：数据集与默认目录
  - `transforms.py`：数据增强与反归一化
  - `models.py`：DINOv2 编码器 + UNet 解码器
  - `losses.py`、`metrics.py`：损失与指标
  - `training.py`：优化器/调度器、训练/评估/可视化封装
  - `utils.py`：种子设定

## 小贴士
- 若报 `tifffile`/`imagecodecs` 缺失，重新运行上面的 pip 安装命令。
- 想调试速度，可先用 `--epochs 1 --batch-size 2 --img-size 256` 验证流程。
- 想改默认数据/输出根目录，设置环境变量：`export DATA_ROOT=/data && export RUNS_ROOT=/checkpoints`。
