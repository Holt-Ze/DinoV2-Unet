import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# 配置路径
RUNS_DIR = os.path.join(os.getcwd(), "runs")
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "comparison_results")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集名称映射
DATASET_MAPPING = {
    "dinov2_unet_clinicdb": "CVC-ClinicDB",
    "dinov2_unet_colondb": "CVC-ColonDB",
    "dinov2_unet_etis": "ETIS",
    "dinov2_unet_kvasir": "Kvasir-SEG"
}

def load_image(path):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")

def create_attention_comparison(num_samples=1):
    print("正在生成 Attention 对比图...")
    
    dataset_samples = {}
    used_ids = {} # {dataset_name: [id1, id2]}

    for run_name, data_name in DATASET_MAPPING.items():
        analysis_dir = os.path.join(RUNS_DIR, run_name, "analysis", "attn")
        if not os.path.exists(analysis_dir):
            continue
            
        # 获取所有文件
        attn_files = sorted(glob.glob(os.path.join(analysis_dir, "*_attn_b11.png")))
        
        # 特殊逻辑: ETIS 数据集强制使用最后一张图片
        if run_name == "dinov2_unet_etis":
            attn_files = list(reversed(attn_files))
            print(f"[{run_name}] 启用倒序采样 (强制使用 Last Image)")

        samples = []
        ids = []
        for attn_path in attn_files:
            if len(samples) >= num_samples: break
            basename = os.path.basename(attn_path)
            img_id = basename.split("_")[0]
            img_path = os.path.join(DATA_DIR, data_name, "images", f"{img_id}.png")
            if not os.path.exists(img_path): img_path = os.path.join(DATA_DIR, data_name, "images", f"{img_id}.jpg")
            mask_path = os.path.join(DATA_DIR, data_name, "masks", f"{img_id}.png")
            if not os.path.exists(mask_path): mask_path = os.path.join(DATA_DIR, data_name, "masks", f"{img_id}.jpg")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                samples.append((img_path, mask_path, attn_path))
                ids.append(img_id)
        dataset_samples[run_name] = samples
        used_ids[run_name] = ids
        print(f"[{run_name}] Attention 使用了 ID: {ids}")

    sorted_datasets = sorted(dataset_samples.keys())
    
    rows = len(sorted_datasets)
    cols = 3
    if rows == 0: return used_ids

    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    if rows == 1: axes = axes.reshape(1, cols)
    
    for i, dataset_name in enumerate(sorted_datasets):
        samples = dataset_samples[dataset_name]
        if not samples: continue
        
        img_p, mask_p, attn_p = samples[0]
        
        img = load_image(img_p)
        mask = load_image(mask_p)
        attn = load_image(attn_p)
        
        dataset_display_name = dataset_name.replace("dinov2_unet_", "")
        
        # 1. Original
        ax = axes[i, 0]
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.1, 0.5, dataset_display_name, fontsize=12, fontweight='bold', 
                va='center', ha='right', transform=ax.transAxes, rotation=90)
        if i == 0:
            ax.set_title("Original", fontsize=12, fontweight='bold')

        # 2. GT
        ax = axes[i, 1]
        ax.imshow(mask)
        ax.axis('off')
        if i == 0:
            ax.set_title("Ground Truth", fontsize=12, fontweight='bold')

        # 3. Attention
        ax = axes[i, 2]
        ax.imshow(attn)
        ax.axis('off')
        if i == 0:
            ax.set_title("Attention Heatmap", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "comparison_attn.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Attention 对比图已保存: {save_path}")
    return used_ids

def create_feature_comparison(num_samples=1, avoid_ids=None):
    print("正在生成 Feature 对比图...")
    if avoid_ids is None: avoid_ids = {}

    dataset_samples = {}
    for run_name, data_name in DATASET_MAPPING.items():
        analysis_dir = os.path.join(RUNS_DIR, run_name, "analysis", "feature_compare")
        if not os.path.exists(analysis_dir): continue
        frozen_files = sorted(glob.glob(os.path.join(analysis_dir, "*_frozen_*.png")))
        samples = []
        
        current_dataset_avoid = avoid_ids.get(run_name, [])

        # 对于 ETIS, 用户要求强制使用 First Image。
        # 正常逻辑是顺序遍历，只要它不在 avoid_ids 里。
        # 鉴于 ETIS Attn 强制使用了 Last Image，它不在开头的 avoid_ids 里，
        # 所以这里的顺序遍历自然会取到 First Image。
        # 无需特殊处理，只需确保不会因为某种原因跳过。

        for frozen_path in frozen_files:
            if len(samples) >= num_samples: break
            basename = os.path.basename(frozen_path)
            img_id = basename.split("_")[0]
            
            # Skip if used in Attention plot
            if img_id in current_dataset_avoid:
                print(f"[{run_name}] 跳过已在 Attention 图中使用的 ID: {img_id}")
                continue

            trainable_glob = glob.glob(os.path.join(analysis_dir, f"{img_id}_trainable_*.png"))
            if not trainable_glob: continue
            trainable_path = trainable_glob[0]
            img_path = os.path.join(DATA_DIR, data_name, "images", f"{img_id}.png")
            if not os.path.exists(img_path): img_path = os.path.join(DATA_DIR, data_name, "images", f"{img_id}.jpg")
            mask_path = os.path.join(DATA_DIR, data_name, "masks", f"{img_id}.png")
            if not os.path.exists(mask_path): mask_path = os.path.join(DATA_DIR, data_name, "masks", f"{img_id}.jpg")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                samples.append((img_path, mask_path, frozen_path, trainable_path))
                print(f"[{run_name}] Feature 使用了 ID: {img_id}")
                
        dataset_samples[run_name] = samples

    sorted_datasets = sorted(dataset_samples.keys())
    
    rows = len(sorted_datasets)
    cols = 4
    if rows == 0: return

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    if rows == 1: axes = axes.reshape(1, cols)

    for i, dataset_name in enumerate(sorted_datasets):
        samples = dataset_samples[dataset_name]
        if not samples: continue
        
        img_p, mask_p, frozen_p, train_p = samples[0]
        
        img = load_image(img_p)
        mask = load_image(mask_p)
        frozen = load_image(frozen_p)
        trainable = load_image(train_p)
        
        dataset_display_name = dataset_name.replace("dinov2_unet_", "")
        
        # Original
        ax = axes[i, 0]
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.1, 0.5, dataset_display_name, fontsize=12, fontweight='bold', 
                va='center', ha='right', transform=ax.transAxes, rotation=90)
        if i == 0:
            ax.set_title("Original", fontsize=12, fontweight='bold')

        # GT
        ax = axes[i, 1]
        ax.imshow(mask)
        ax.axis('off')
        if i == 0:
            ax.set_title("Ground Truth", fontsize=12, fontweight='bold')
        
        # Frozen
        ax = axes[i, 2]
        ax.imshow(frozen)
        ax.axis('off')
        if i == 0:
            ax.set_title("Frozen Features", fontsize=12, fontweight='bold')

        # Trainable
        ax = axes[i, 3]
        ax.imshow(trainable)
        ax.axis('off')
        if i == 0:
            ax.set_title("Trainable Features", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "comparison_features.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Feature 对比图已保存: {save_path}")

def create_tsne_comparison():
    print("正在生成 t-SNE 对比图...")
    
    dataset_tsne = {}
    for run_name in DATASET_MAPPING.keys():
        tsne_dir = os.path.join(RUNS_DIR, run_name, "analysis", "tsne")
        if not os.path.exists(tsne_dir): continue
        resnet_path = os.path.join(tsne_dir, "resnet50_tsne.png")
        dinov2_path = os.path.join(tsne_dir, "dinov2_tsne.png")
        if os.path.exists(resnet_path) and os.path.exists(dinov2_path):
            dataset_tsne[run_name] = (resnet_path, dinov2_path)
    
    if not dataset_tsne:
        print("没有找到 t-SNE 图片。")
        return

    sorted_datasets = sorted(dataset_tsne.keys())
    cols = len(sorted_datasets) # 4
    rows = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 8))
    if cols == 1: axes = axes.reshape(2, 1)

    for col_idx, dataset_name in enumerate(sorted_datasets):
        resnet_p, dinov2_p = dataset_tsne[dataset_name]
        resnet_img = load_image(resnet_p)
        dinov2_img = load_image(dinov2_p)
        dataset_display_name = dataset_name.replace("dinov2_unet_", "")
        
        # Row 0: ResNet
        ax = axes[0, col_idx]
        ax.imshow(resnet_img)
        ax.axis('off')
        ax.set_title(dataset_display_name, fontsize=14, fontweight='bold')
        if col_idx == 0:
            ax.text(-0.1, 0.5, "ResNet50", fontsize=16, fontweight='bold', 
                    va='center', ha='right', transform=ax.transAxes, rotation=90)
        
        # Row 1: DinoV2
        ax = axes[1, col_idx]
        ax.imshow(dinov2_img)
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.1, 0.5, "DINOv2", fontsize=16, fontweight='bold', 
                    va='center', ha='right', transform=ax.transAxes, rotation=90)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "comparison_tsne.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"t-SNE 对比图已保存: {save_path}")

if __name__ == "__main__":
    used_stats = create_attention_comparison(num_samples=1)
    create_feature_comparison(num_samples=1, avoid_ids=used_stats)
    create_tsne_comparison()
    print("所有可视化结果生成完毕。")
