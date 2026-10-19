"""Data loading and preprocessing for Fashion-MNIST (Xiao et al., 2017).

The dataset is fetched from the official Zalando GitHub repository the first
time it is needed and cached under ``data/raw``. The 60,000 official training
images are split (stratified, fixed seed) into 55,000 train / 5,000 validation;
the official 10,000 test images are never used for model selection.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLIT_SEED = 42  # data split is fixed so every model sees identical splits


def _fetch(fname: str, raw_dir: Path) -> Path:
    path = raw_dir / fname
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(BASE_URL + fname, path)
    except Exception as exc:  # pragma: no cover - network dependent
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Could not download {fname} ({exc}).\n"
            f"Download the four *.gz files from {BASE_URL} (or the Kaggle "
            f"'zalando-research/fashionmnist' dataset) and place them in {raw_dir}."
        ) from exc
    return path


def _read_idx(path: Path, images: bool) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        if images:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"bad image magic number in {path}"
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).copy()
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic number in {path}"
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


def load_raw(root: Path | str = DEFAULT_ROOT):
    """Return (x_train, y_train, x_test, y_test) as uint8 numpy arrays."""
    root = Path(root)
    xtr = _read_idx(_fetch(FILES["train_images"], root), images=True)
    ytr = _read_idx(_fetch(FILES["train_labels"], root), images=False)
    xte = _read_idx(_fetch(FILES["test_images"], root), images=True)
    yte = _read_idx(_fetch(FILES["test_labels"], root), images=False)
    assert xtr.shape == (60000, 28, 28) and xte.shape == (10000, 28, 28)
    return xtr, ytr, xte, yte


def get_data(root: Path | str = DEFAULT_ROOT, val_size: int = 5000, subset: int | None = None):
    """Return normalised float tensors for train / val / test plus normalisation stats.

    ``subset`` (for smoke tests) keeps only that many training images.
    """
    xtr, ytr, xte, yte = load_raw(root)
    idx = np.arange(len(xtr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, stratify=ytr, random_state=SPLIT_SEED
    )
    if subset:
        tr_idx, va_idx = tr_idx[:subset], va_idx[: max(subset // 5, 200)]

    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)

    x_train, x_val, x_test = to_tensor(xtr[tr_idx]), to_tensor(xtr[va_idx]), to_tensor(xte)
    mean, std = x_train.mean().item(), x_train.std().item()  # train statistics only
    norm = lambda t: (t - mean) / std
    return {
        "x_train": norm(x_train), "y_train": torch.from_numpy(ytr[tr_idx]).long(),
        "x_val": norm(x_val), "y_val": torch.from_numpy(ytr[va_idx]).long(),
        "x_test": norm(x_test), "y_test": torch.from_numpy(yte).long(),
        "mean": mean, "std": std,
    }
