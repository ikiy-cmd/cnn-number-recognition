"""Minimal MNIST IDX reader.

The dataset is fetched from the public MNIST mirror hosted by PyTorch
(ossci-datasets).  Files are cached under ``data/mnist`` so the app keeps
working offline once the first download has completed.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mnist"

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / FILES[name]
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    last_error: Exception | None = None
    for mirror in MIRRORS:
        url = mirror + FILES[name]
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise IOError(f"unexpected payload size {len(payload)}")
            dest.write_bytes(payload)
            return dest
        except Exception as exc:  # pragma: no cover - network fallback
            last_error = exc
    raise RuntimeError(f"Could not download MNIST file {name}: {last_error}")


def _read_idx(path: Path, expected_magic: int) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != expected_magic:
            raise ValueError(f"bad magic number in {path.name}: {magic}")
        if expected_magic == 2051:
            rows, cols = struct.unpack(">II", handle.read(8))
            data = np.frombuffer(handle.read(), dtype=np.uint8)
            return data.reshape(count, rows, cols)
        data = np.frombuffer(handle.read(), dtype=np.uint8)
        return data.reshape(count)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images[uint8, N, 28, 28], labels[uint8, N])`` for train/test."""
    images = _read_idx(_download(f"{split}_images"), 2051)
    labels = _read_idx(_download(f"{split}_labels"), 2049)
    return images, labels


def load_all() -> dict[str, np.ndarray]:
    train_x, train_y = load_split("train")
    test_x, test_y = load_split("test")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }


def ensure_downloaded() -> None:
    for name in FILES:
        _download(name)


if __name__ == "__main__":
    ensure_downloaded()
    data = load_all()
    for key, value in data.items():
        print(key, value.shape, value.dtype)
