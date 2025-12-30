import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .transforms import denorm

try:
    import timm
except Exception:
    timm = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    PCA = None
    TSNE = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if isinstance(state, dict):
        state = {k: v for k, v in state.items()
                 if not (k.endswith("total_ops") or k.endswith("total_params"))}
    model.load_state_dict(state, strict=False)


def denorm_image(img: torch.Tensor, mean, std) -> torch.Tensor:
    return denorm(img, mean, std).clamp(0.0, 1.0)


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def feature_to_heatmap(feat: torch.Tensor) -> torch.Tensor:
    return normalize_map(feat.abs().mean(dim=1, keepdim=True))


def blend_heatmap(
    img: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
) -> torch.Tensor:
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)
    heatmap = heatmap.to(device=img.device, dtype=img.dtype)
    color_map = torch.stack([
        heatmap.squeeze(0) * color[0],
        heatmap.squeeze(0) * color[1],
        heatmap.squeeze(0) * color[2],
    ], dim=0)
    return (1.0 - alpha) * img + alpha * color_map


def overlay_mask(
    img: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    color: Tuple[float, float, float] = (0.2, 1.0, 0.2),
) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    mask = mask.to(device=img.device, dtype=img.dtype).clamp(0.0, 1.0)
    color_map = torch.stack([
        mask * color[0],
        mask * color[1],
        mask * color[2],
    ], dim=0)
    return (1.0 - alpha) * img + alpha * color_map


def save_overlay_image(
    img: torch.Tensor,
    heatmap: torch.Tensor,
    out_path: str,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
) -> None:
    overlay = blend_heatmap(img, heatmap, alpha=alpha)
    if mask is not None:
        overlay = overlay_mask(overlay, mask)
    save_image(overlay, out_path)


def _get_patch_size(model) -> int:
    patch = model.patch_embed.patch_size
    if isinstance(patch, tuple):
        return int(patch[0])
    return int(patch)


def _vit_patch_and_pos_embed(model, x: torch.Tensor):
    b, c, h, w = x.shape
    patch_size = _get_patch_size(model)
    x_pe = model.patch_embed(x)
    if hasattr(model, "_pos_embed"):
        pe_out = model._pos_embed(x_pe)
        if isinstance(pe_out, (list, tuple)):
            x_tokens, (gh, gw) = pe_out[0], pe_out[1]
        else:
            x_tokens, (gh, gw) = pe_out, (h // patch_size, w // patch_size)
    else:
        x_tokens, (gh, gw) = x_pe, (h // patch_size, w // patch_size)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x_tokens = x_tokens + model.pos_embed
    x_tokens = model.pos_drop(x_tokens)
    return x_tokens, (gh, gw), patch_size


def _qkv_to_attn(qkv: torch.Tensor, num_heads: int, embed_dim: int, scale: Optional[float]):
    b, n, _ = qkv.shape
    head_dim = embed_dim // num_heads
    if scale is None:
        scale = head_dim ** -0.5
    qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * scale
    return attn.softmax(dim=-1)


def extract_vit_attention_maps(
    vit_model,
    x: torch.Tensor,
    block_indices: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int], int]:
    block_set = set(block_indices)
    tokens, (gh, gw), patch_size = _vit_patch_and_pos_embed(vit_model, x)
    attn_maps: Dict[int, torch.Tensor] = {}
    for i, blk in enumerate(vit_model.blocks):
        if i in block_set:
            tokens_norm = blk.norm1(tokens) if hasattr(blk, "norm1") else tokens
            qkv = blk.attn.qkv(tokens_norm)
            scale = getattr(blk.attn, "scale", None)
            attn = _qkv_to_attn(qkv, blk.attn.num_heads, tokens_norm.shape[-1], scale)
            if tokens_norm.shape[1] == gh * gw + 1:
                attn_patch = attn[:, :, 0, 1:]
            else:
                attn_patch = attn.mean(dim=2)
            attn_patch = attn_patch.mean(dim=1).reshape(-1, 1, gh, gw)
            attn_maps[i] = attn_patch
        tokens = blk(tokens)
    return attn_maps, (gh, gw), patch_size


def extract_vit_block_features(
    vit_model,
    x: torch.Tensor,
    block_indices: Sequence[int],
    apply_norm: bool = False,
) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int], int]:
    block_set = set(block_indices)
    tokens, (gh, gw), patch_size = _vit_patch_and_pos_embed(vit_model, x)
    feats: Dict[int, torch.Tensor] = {}
    for i, blk in enumerate(vit_model.blocks):
        tokens = blk(tokens)
        if i in block_set:
            feat_tokens = vit_model.norm(tokens) if apply_norm and hasattr(vit_model, "norm") else tokens
            if feat_tokens.shape[1] == gh * gw + 1:
                feat_tokens = feat_tokens[:, 1:, :]
            fm = feat_tokens.transpose(1, 2).reshape(x.size(0), -1, gh, gw)
            feats[i] = fm
    return feats, (gh, gw), patch_size


def build_cnn_features(backbone: str, pretrained: bool = True):
    if timm is None:
        raise RuntimeError("timm is required for CNN feature extraction.")
    return timm.create_model(backbone, pretrained=pretrained, features_only=True)


def _sample_indices(idx: torch.Tensor, remaining: int, rng: np.random.Generator) -> torch.Tensor:
    if remaining <= 0 or idx.numel() == 0:
        return idx.new_empty((0,), dtype=torch.long)
    idx_np = idx.detach().cpu().numpy()
    if idx_np.size > remaining:
        idx_np = rng.choice(idx_np, size=remaining, replace=False)
    return torch.from_numpy(idx_np).to(idx.device)


def gather_vit_patch_features(
    vit_model,
    loader: Iterable,
    device: str,
    block_index: int,
    samples_per_class: int,
    max_images: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vit_model.eval()
    rng = np.random.default_rng(seed)
    feats_list = []
    labels_list = []
    counts = {0: 0, 1: 0}
    seen = 0
    for imgs, msks, _ in loader:
        if max_images and seen >= max_images:
            break
        imgs = imgs.to(device)
        msks = msks.to(device)
        with torch.no_grad():
            feats, (gh, gw), _ = extract_vit_block_features(vit_model, imgs, [block_index])
        fm = feats[block_index]
        mask_small = F.interpolate(msks.float(), size=(gh, gw), mode="nearest")
        flat_feats = fm.permute(0, 2, 3, 1).reshape(-1, fm.shape[1])
        flat_mask = mask_small.view(-1)
        polyp_idx = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(1)
        bg_idx = torch.nonzero(flat_mask <= 0.5, as_tuple=False).squeeze(1)
        polyp_keep = _sample_indices(polyp_idx, samples_per_class - counts[1], rng)
        bg_keep = _sample_indices(bg_idx, samples_per_class - counts[0], rng)
        if polyp_keep.numel() > 0:
            feats_list.append(flat_feats[polyp_keep].detach().cpu().numpy())
            labels_list.append(np.ones(polyp_keep.numel(), dtype=np.int64))
            counts[1] += polyp_keep.numel()
        if bg_keep.numel() > 0:
            feats_list.append(flat_feats[bg_keep].detach().cpu().numpy())
            labels_list.append(np.zeros(bg_keep.numel(), dtype=np.int64))
            counts[0] += bg_keep.numel()
        seen += 1
        if counts[0] >= samples_per_class and counts[1] >= samples_per_class:
            break
    if not feats_list:
        return np.empty((0, 1)), np.empty((0,), dtype=np.int64)
    feats_arr = np.concatenate(feats_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    return feats_arr, labels_arr


def gather_cnn_patch_features(
    cnn_model,
    loader: Iterable,
    device: str,
    out_index: int,
    samples_per_class: int,
    max_images: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cnn_model.eval()
    rng = np.random.default_rng(seed)
    feats_list = []
    labels_list = []
    counts = {0: 0, 1: 0}
    seen = 0
    for imgs, msks, _ in loader:
        if max_images and seen >= max_images:
            break
        imgs = imgs.to(device)
        msks = msks.to(device)
        with torch.no_grad():
            feats = cnn_model(imgs)
        if isinstance(feats, (list, tuple)):
            fm = feats[out_index]
        else:
            fm = feats
        mask_small = F.interpolate(msks.float(), size=fm.shape[-2:], mode="nearest")
        flat_feats = fm.permute(0, 2, 3, 1).reshape(-1, fm.shape[1])
        flat_mask = mask_small.view(-1)
        polyp_idx = torch.nonzero(flat_mask > 0.5, as_tuple=False).squeeze(1)
        bg_idx = torch.nonzero(flat_mask <= 0.5, as_tuple=False).squeeze(1)
        polyp_keep = _sample_indices(polyp_idx, samples_per_class - counts[1], rng)
        bg_keep = _sample_indices(bg_idx, samples_per_class - counts[0], rng)
        if polyp_keep.numel() > 0:
            feats_list.append(flat_feats[polyp_keep].detach().cpu().numpy())
            labels_list.append(np.ones(polyp_keep.numel(), dtype=np.int64))
            counts[1] += polyp_keep.numel()
        if bg_keep.numel() > 0:
            feats_list.append(flat_feats[bg_keep].detach().cpu().numpy())
            labels_list.append(np.zeros(bg_keep.numel(), dtype=np.int64))
            counts[0] += bg_keep.numel()
        seen += 1
        if counts[0] >= samples_per_class and counts[1] >= samples_per_class:
            break
    if not feats_list:
        return np.empty((0, 1)), np.empty((0,), dtype=np.int64)
    feats_arr = np.concatenate(feats_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    return feats_arr, labels_arr


def reduce_tsne(features: np.ndarray, seed: int = 42, perplexity: int = 30) -> Optional[np.ndarray]:
    if TSNE is None:
        return None
    x = features
    if PCA is not None and x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity)
    return tsne.fit_transform(x)


def reduce_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    if PCA is not None:
        return PCA(n_components=n_components, random_state=0).fit_transform(features)
    x = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:n_components].T


def save_points_csv(points: np.ndarray, labels: np.ndarray, out_path: str, model_name: str) -> None:
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "source"])
        for (x, y), label in zip(points, labels):
            writer.writerow([float(x), float(y), int(label), model_name])


def save_scatter(points: np.ndarray, labels: np.ndarray, out_path: str, title: str) -> bool:
    if plt is None:
        return False
    plt.figure(figsize=(6, 6))
    bg = labels == 0
    fg = labels == 1
    plt.scatter(points[bg, 0], points[bg, 1], s=8, c="#2b6cb0", alpha=0.55, label="background")
    plt.scatter(points[fg, 0], points[fg, 1], s=8, c="#e53e3e", alpha=0.55, label="polyp")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True
