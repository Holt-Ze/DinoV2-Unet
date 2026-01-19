import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .paths import DEFAULT_DATA_ROOT, DEFAULT_RUNS_ROOT
from .transforms import build_polyp_transform, _normalize_aug_mode, VALID_AUG_MODES

try:
    import tifffile
except ImportError:
    tifffile = None


class KvasirSEG(Dataset):
    def __init__(self, data_dir: str, split: str = "train", img_size: int = 448,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seed: int = 42,
                 aug_mode: str = "strong", subset_ratio: float = 1.0):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not os.path.isdir(self.msk_dir):
            raise FileNotFoundError(f"Mask folder not found: {self.msk_dir}")
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        names = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[1].lower() in exts]
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")
        random.Random(seed).shuffle(names)
        n = len(names)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == "train":
            self.names = names[:n_train] or names
            if subset_ratio < 1.0:
                limit = max(1, int(len(self.names) * subset_ratio))
                self.names = self.names[:limit]
        elif split == "val":
            self.names = names[n_train:n_train + n_val] or names
        else:
            self.names = names[n_train + n_val:] or names
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_mode = _normalize_aug_mode(aug_mode)
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.names)

    def _get_transform(self):
        return build_polyp_transform(self.split, self.img_size, self.mean, self.std, self.aug_mode)

    def _load_pair(self, name):
        img_path = os.path.join(self.img_dir, name)
        stem = os.path.splitext(name)[0]
        candidates = [
            os.path.join(self.msk_dir, stem + ext) for ext in (".png", ".jpg", ".jpeg", ".bmp")
        ]
        for msk_path in candidates:
            if os.path.exists(msk_path):
                break
        else:
            raise FileNotFoundError(f"Mask for {name} not found in {self.msk_dir}")
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("L"))
        return img, msk

    def __getitem__(self, idx):
        name = self.names[idx]
        img, msk = self._load_pair(name)
        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"]
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk, name


class CVCClinicDBDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", img_size: int = 448,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seed: int = 42,
                 aug_mode: str = "strong", subset_ratio: float = 1.0):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "Original")
        self.msk_dir = os.path.join(data_dir, "Ground Truth")
        if not (os.path.isdir(self.img_dir) and os.path.isdir(self.msk_dir)):
            alt_img = os.path.join(data_dir, "images")
            alt_msk = os.path.join(data_dir, "masks")
            if os.path.isdir(alt_img) and os.path.isdir(alt_msk):
                self.img_dir, self.msk_dir = alt_img, alt_msk
            else:
                raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        names = [f for f in sorted(os.listdir(self.img_dir)) if os.path.splitext(f)[1].lower() in valid_exts]
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")
        random.Random(seed).shuffle(names)
        n = len(names)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == "train":
            self.names = names[:n_train] or names
            if subset_ratio < 1.0:
                limit = max(1, int(len(self.names) * subset_ratio))
                self.names = self.names[:limit]
        elif split == "val":
            self.names = names[n_train:n_train + n_val] or names
        else:
            self.names = names[n_train + n_val:] or names
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_mode = _normalize_aug_mode(aug_mode)
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.names)

    def _get_transform(self):
        return build_polyp_transform(self.split, self.img_size, self.mean, self.std, self.aug_mode)

    @staticmethod
    def _read_image(path: str, is_mask: bool):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".tif", ".tiff"}:
            if tifffile is None:
                raise RuntimeError("tifffile is required for TIFF files. Install with `pip install tifffile imagecodecs`.")
            return tifffile.imread(path)
        mode = "L" if is_mask else "RGB"
        return np.array(Image.open(path).convert(mode))

    def _load_pair(self, name):
        img_path = os.path.join(self.img_dir, name)
        msk_path = os.path.join(self.msk_dir, name)
        try:
            if os.path.getsize(img_path) == 0 or os.path.getsize(msk_path) == 0:
                raise ValueError("empty file")
            img = self._read_image(img_path, is_mask=False)
            msk = self._read_image(msk_path, is_mask=True)
        except Exception as exc:
            print(f"[warn] failed to load {name}: {exc}")
            return None, None
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        return img, msk

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            name = self.names[idx]
            img, msk = self._load_pair(name)
            if img is not None:
                break
            idx = random.randint(0, len(self.names) - 1)
            attempts += 1
        else:
            raise RuntimeError(
                f"Failed to load {max_attempts} samples in a row from {self.img_dir}. Dataset files may be missing or corrupted."
            )
        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"]
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk, name


class CVCColonDB(Dataset):
    def __init__(self, data_dir: str, split: str = "train", img_size: int = 448,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seed: int = 42,
                 aug_mode: str = "strong", subset_ratio: float = 1.0):
        super().__init__()
        if os.path.exists(os.path.join(data_dir, "images")):
            self.img_dir = os.path.join(data_dir, "images")
            self.msk_dir = os.path.join(data_dir, "masks")
        else:
            self.img_dir = os.path.join(data_dir, "Original")
            self.msk_dir = os.path.join(data_dir, "Ground Truth")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not os.path.isdir(self.msk_dir):
            raise FileNotFoundError(f"Mask folder not found: {self.msk_dir}")
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        names = [f for f in sorted(os.listdir(self.img_dir)) if os.path.splitext(f)[1].lower() in exts]
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")
        random.Random(seed).shuffle(names)
        n = len(names)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == "train":
            self.names = names[:n_train] or names
            if subset_ratio < 1.0:
                limit = max(1, int(len(self.names) * subset_ratio))
                self.names = self.names[:limit]
        elif split == "val":
            self.names = names[n_train:n_train + n_val] or names
        else:
            self.names = names[n_train + n_val:] or names
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_mode = _normalize_aug_mode(aug_mode)
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.names)

    def _get_transform(self):
        return build_polyp_transform(self.split, self.img_size, self.mean, self.std, self.aug_mode)

    def _load_pair(self, name):
        img_path = os.path.join(self.img_dir, name)
        msk_path = os.path.join(self.msk_dir, name)
        if not os.path.exists(msk_path):
            stem = os.path.splitext(name)[0]
            for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
                candidate = os.path.join(self.msk_dir, stem + ext)
                if os.path.exists(candidate):
                    msk_path = candidate
                    break
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("L"))
        return img, msk

    def __getitem__(self, idx):
        name = self.names[idx]
        img, msk = self._load_pair(name)
        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"]
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk, name


class ETISLaribDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", img_size: int = 448,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seed: int = 42,
                 aug_mode: str = "strong", subset_ratio: float = 1.0):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not os.path.isdir(self.msk_dir):
            raise FileNotFoundError(f"Mask folder not found: {self.msk_dir}")
        exts = {".png"}
        names = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[1].lower() in exts]
        if not names:
            raise RuntimeError(f"No images found in {self.img_dir}")
        random.Random(seed).shuffle(names)
        n = len(names)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if split == "train":
            self.names = names[:n_train] or names
            if subset_ratio < 1.0:
                limit = max(1, int(len(self.names) * subset_ratio))
                self.names = self.names[:limit]
        elif split == "val":
            self.names = names[n_train:n_train + n_val] or names
        else:
            self.names = names[n_train + n_val:] or names
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.aug_mode = _normalize_aug_mode(aug_mode)
        self.transform = self._get_transform()

    def __len__(self):
        return len(self.names)

    def _get_transform(self):
        return build_polyp_transform(self.split, self.img_size, self.mean, self.std, self.aug_mode)

    def _load_pair(self, name):
        img_path = os.path.join(self.img_dir, name)
        stem = os.path.splitext(name)[0]
        msk_path = os.path.join(self.msk_dir, stem + ".png")
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask for {name} not found: {msk_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path).convert("L"))
        return img, msk

    def __getitem__(self, idx):
        name = self.names[idx]
        img, msk = self._load_pair(name)
        transformed = self.transform(image=img, mask=msk)
        img = transformed["image"]
        msk = transformed["mask"]
        msk = (msk > 127).float().unsqueeze(0)
        return img, msk, name


@dataclass(frozen=True)
class DatasetSpec:
    cls: Type[Dataset]
    default_subdir: Optional[str]
    default_save_dir: str
    requires_tifffile: bool = False


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "kvasir": DatasetSpec(KvasirSEG, "Kvasir-SEG", os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_kvasir")),
    "clinicdb": DatasetSpec(CVCClinicDBDataset, "CVC-ClinicDB", os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_clinicdb"), True),
    "colondb": DatasetSpec(CVCColonDB, "CVC-ColonDB", os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_colondb")),
    "etis": DatasetSpec(ETISLaribDataset, "ETIS", os.path.join(DEFAULT_RUNS_ROOT, "dinov2_unet_etis")),
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
    key = name.lower()
    return DATASET_ALIASES.get(key, key)


def resolve_data_dir(spec: DatasetSpec, override: Optional[str]) -> str:
    if override:
        return os.path.abspath(override)
    if spec.default_subdir is None:
        raise ValueError("No default data directory defined; please provide --data-dir.")
    primary = os.path.abspath(os.path.join(DEFAULT_DATA_ROOT, spec.default_subdir))
    if os.path.isdir(primary):
        return primary
    # Try known alternative folder names.
    for key, fallbacks in DATASET_SUBDIR_FALLBACKS.items():
        if spec is DATASET_SPECS.get(key):
            for name in fallbacks:
                candidate = os.path.abspath(os.path.join(DEFAULT_DATA_ROOT, name))
                if os.path.isdir(candidate):
                    return candidate
    return primary

