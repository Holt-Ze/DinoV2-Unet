"""Dataset definitions for supported polyp segmentation benchmarks."""

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Type

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .paths import DEFAULT_DATA_ROOT, DEFAULT_RUNS_ROOT
from .transforms import build_polyp_transform, _normalize_aug_mode

try:
    import tifffile
except ImportError:
    tifffile = None


SPLITS = {"train", "val", "test"}
STANDARD_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TIFF_EXTS = (".tif", ".tiff")


def _do_split(names: list, split: str, fold_idx: int, num_folds: int) -> list:
    """Split names into train/val/test or deterministic K-fold partitions."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

    n = len(names)
    if num_folds <= 1:
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == "train":
            return names[:n_train] or names
        if split == "val":
            return names[n_train : n_train + n_val] or names
        return names[n_train + n_val :] or names

    if fold_idx < 0 or fold_idx >= num_folds:
        raise ValueError(f"fold_idx={fold_idx} out of range for num_folds={num_folds}.")

    chunk_size = n // num_folds
    test_start = fold_idx * chunk_size
    test_end = test_start + chunk_size if fold_idx < num_folds - 1 else n
    test_names = names[test_start:test_end]
    remain_names = names[:test_start] + names[test_end:]

    val_size = max(1, int(n * 0.1))
    val_names = remain_names[:val_size]
    train_names = remain_names[val_size:]

    if split == "train":
        return train_names or names
    if split == "val":
        return val_names or names
    return test_names or names


def _read_image(path: str, is_mask: bool) -> np.ndarray:
    """Read standard image and TIFF files into numpy arrays."""
    ext = os.path.splitext(path)[1].lower()
    if ext in TIFF_EXTS:
        if tifffile is None:
            raise RuntimeError(
                "tifffile is required for TIFF files. "
                "Install with `pip install tifffile imagecodecs`."
            )
        arr = tifffile.imread(path)
    else:
        mode = "L" if is_mask else "RGB"
        arr = np.array(Image.open(path).convert(mode))

    if is_mask:
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


class PolypDataset(Dataset):
    """Base dataset for image/mask segmentation folders."""

    layout_options: Sequence[Tuple[str, str]] = (("images", "masks"),)
    image_exts: Sequence[str] = STANDARD_EXTS
    mask_exts: Sequence[str] = STANDARD_EXTS
    skip_bad_files: bool = False

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 448,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        seed: int = 42,
        aug_mode: str = "strong",
        fold_idx: int = 0,
        num_folds: int = 1,
    ):
        super().__init__()
        self.img_dir, self.msk_dir = self._resolve_layout(data_dir)
        names = self._list_images()
        random.Random(seed).shuffle(names)

        self.names = _do_split(names, split, fold_idx, num_folds)
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_mode = _normalize_aug_mode(aug_mode)
        self.transform = build_polyp_transform(
            split,
            img_size,
            mean,
            std,
            self.aug_mode,
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 10 if self.skip_bad_files else 1
        while attempts < max_attempts:
            name = self.names[idx]
            try:
                img, msk = self._load_pair(name)
                break
            except Exception as exc:
                if not self.skip_bad_files:
                    raise
                print(f"[warn] failed to load {name}: {exc}")
                idx = random.randint(0, len(self.names) - 1)
                attempts += 1
        else:
            raise RuntimeError(
                f"Failed to load {max_attempts} samples in a row from "
                f"{self.img_dir}. Dataset files may be missing or corrupted."
            )

        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"]
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk, name

    @classmethod
    def _resolve_layout(cls, data_dir: str) -> Tuple[str, str]:
        for image_subdir, mask_subdir in cls.layout_options:
            img_dir = os.path.join(data_dir, image_subdir)
            msk_dir = os.path.join(data_dir, mask_subdir)
            if os.path.isdir(img_dir) and os.path.isdir(msk_dir):
                return img_dir, msk_dir
        expected = " or ".join(
            f"{img}/{msk}" for img, msk in cls.layout_options
        )
        raise FileNotFoundError(f"Dataset layout not found under {data_dir}: {expected}")

    def _list_images(self) -> list:
        valid_exts = {ext.lower() for ext in self.image_exts}
        names = [
            name
            for name in sorted(os.listdir(self.img_dir))
            if os.path.splitext(name)[1].lower() in valid_exts
        ]
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")
        return names

    def _find_mask_path(self, name: str) -> str:
        exact = os.path.join(self.msk_dir, name)
        if os.path.exists(exact):
            return exact

        stem = os.path.splitext(name)[0]
        for ext in self.mask_exts:
            candidate = os.path.join(self.msk_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Mask for {name} not found in {self.msk_dir}")

    def _load_pair(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        img_path = os.path.join(self.img_dir, name)
        msk_path = self._find_mask_path(name)
        if self.skip_bad_files and (
            os.path.getsize(img_path) == 0 or os.path.getsize(msk_path) == 0
        ):
            raise ValueError("empty file")
        return _read_image(img_path, is_mask=False), _read_image(msk_path, is_mask=True)


class KvasirSEG(PolypDataset):
    """Kvasir-SEG dataset."""


class CVCClinicDBDataset(PolypDataset):
    """CVC-ClinicDB dataset, supporting original and normalized layouts."""

    layout_options = (("Original", "Ground Truth"), ("images", "masks"))
    image_exts = TIFF_EXTS + (".png", ".jpg", ".jpeg")
    mask_exts = TIFF_EXTS + (".png", ".jpg", ".jpeg")
    skip_bad_files = True


class CVCColonDB(PolypDataset):
    """CVC-ColonDB dataset, supporting original and normalized layouts."""

    layout_options = (("images", "masks"), ("Original", "Ground Truth"))
    image_exts = TIFF_EXTS + STANDARD_EXTS
    mask_exts = TIFF_EXTS + STANDARD_EXTS


class ETISLaribDataset(PolypDataset):
    """ETIS-LaribPolypDB dataset."""

    image_exts = (".png",)
    mask_exts = (".png",)


@dataclass(frozen=True)
class DatasetSpec:
    """Static configuration for a supported dataset."""

    cls: Type[Dataset]
    default_subdir: Optional[str]
    default_save_dir: str
    requires_tifffile: bool = False


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "kvasir": DatasetSpec(
        KvasirSEG,
        "Kvasir-SEG",
        os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_kvasir"),
    ),
    "clinicdb": DatasetSpec(
        CVCClinicDBDataset,
        "CVC-ClinicDB",
        os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_clinicdb"),
        True,
    ),
    "colondb": DatasetSpec(
        CVCColonDB,
        "CVC-ColonDB",
        os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_colondb"),
    ),
    "etis": DatasetSpec(
        ETISLaribDataset,
        "ETIS",
        os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_etis"),
    ),
}

DATASET_ALIASES: Dict[str, str] = {
    "kvasir-seg": "kvasir",
    "clinic": "clinicdb",
    "cvc-clinicdb": "clinicdb",
    "colon": "colondb",
    "cvc-colondb": "colondb",
    "etis-larib": "etis",
}

DATASET_SUBDIR_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "etis": ("ETIS-LaribPolypDB",),
}


def resolve_dataset_key(name: str) -> str:
    """Resolve a user dataset name or alias to a canonical key."""
    key = name.lower()
    return DATASET_ALIASES.get(key, key)


def resolve_data_dir(spec: DatasetSpec, override: Optional[str]) -> str:
    """Resolve the dataset directory path from an override or DATA_ROOT."""
    if override:
        return os.path.abspath(override)
    if spec.default_subdir is None:
        raise ValueError("No default data directory defined; please provide --data-dir.")

    primary = os.path.abspath(os.path.join(DEFAULT_DATA_ROOT, spec.default_subdir))
    if os.path.isdir(primary):
        return primary

    for key, fallbacks in DATASET_SUBDIR_FALLBACKS.items():
        if spec is DATASET_SPECS.get(key):
            for name in fallbacks:
                candidate = os.path.abspath(os.path.join(DEFAULT_DATA_ROOT, name))
                if os.path.isdir(candidate):
                    return candidate
    return primary


def resolve_data_dir_from_root(spec: DatasetSpec, override: Optional[str]) -> str:
    """Resolve either an exact dataset dir or a root containing default_subdir."""
    if not override:
        return resolve_data_dir(spec, None)

    candidate = os.path.abspath(override)
    if os.path.isdir(candidate) and spec.default_subdir:
        subdir = os.path.join(candidate, spec.default_subdir)
        if os.path.isdir(subdir):
            return subdir
    return candidate
