"""PyTorch Dataset / DataLoader for Google's Quick, Draw! dataset.
CREDIT goes to Claude Opus 4.7.

Uses the `numpy_bitmap` format: one .npy per category, shape (N, 784) uint8,
each row a 28x28 flattened grayscale drawing (0 = background, 255 = ink).
Files are downloaded on demand from the official GCS bucket and cached.

Bucket layout:
  https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy

The 345-category name list is fetched from the official GitHub repo.
"""

from __future__ import annotations

import bisect
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


QUICKDRAW_BASE_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
)
QUICKDRAW_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/"
    "quickdraw-dataset/master/categories.txt"
)


class QuickDrawDataset(Dataset):
    """Quick, Draw! dataset in 28x28 bitmap form.

    Args:
        root: Directory where .npy files (and the category list) are cached.
        categories: Category names, e.g. ["cat", "dog", "airplane"]. If None,
            uses all 345 official categories (downloaded on first use).
        train: If True (default), use the train split. If False, the test
            split. The split is deterministic and per-category: the first
            (1 - test_fraction) samples are train, the remainder test.
        test_fraction: Fraction of each category reserved for the test split.
        max_per_category: Optional cap per class, applied *before* the
            train/test split. Useful since some categories have >100k drawings.
        transform: Callable applied to each (28, 28) uint8 numpy array. If
            None, the default returns a float tensor in [0, 1] of shape
            (1, 28, 28).
        download: If True, fetch missing files from GCS / GitHub.
        mmap: If True (default), memory-map the .npy files instead of loading
            them entirely into RAM.
    """

    # Cached after the first call to list_categories(), keyed by root path.
    _all_categories_cache: dict[Path, list[str]] = {}

    def __init__(
        self,
        root: str | Path,
        categories: Optional[Sequence[str]] = None,
        train: bool = True,
        test_fraction: float = 0.1,
        max_per_category: Optional[int] = None,
        transform: Optional[Callable] = None,
        download: bool = True,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.train = train
        self.test_fraction = test_fraction
        self.transform = transform

        # Always fetch the canonical 345-name list and store it on the
        # instance for convenience (e.g. label -> name lookups).
        self.all_categories = self.list_categories(self.root, download=download)

        if categories is None:
            categories = self.all_categories
        self.categories = list(categories)

        self._arrays: list[np.ndarray] = []
        # Cumulative end-offsets; bisect into this to map a global index
        # to (category, local index).
        self._cum_lengths: list[int] = []

        running = 0
        for category in self.categories:
            path = self._local_path(category)
            if not path.exists():
                if not download:
                    raise FileNotFoundError(
                        f"Missing {path} and download=False."
                    )
                self._download(category, path)

            arr = np.load(path, mmap_mode="r" if mmap else None)
            if max_per_category is not None:
                arr = arr[:max_per_category]

            # Deterministic per-category split: first n_train rows are train,
            # last n_test are test. This is stratified by class for free.
            n = len(arr)
            n_test = int(round(n * test_fraction))
            n_train = n - n_test
            arr = arr[:n_train] if train else arr[n_train:]

            self._arrays.append(arr)
            running += len(arr)
            self._cum_lengths.append(running)

    # ---- public API -------------------------------------------------------

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        label = bisect.bisect_right(self._cum_lengths, idx)
        prev = self._cum_lengths[label - 1] if label > 0 else 0
        local_idx = idx - prev

        # .copy() because mmap'd arrays are read-only and torch.from_numpy
        # would otherwise produce a non-writable tensor.
        flat = np.array(self._arrays[label][local_idx], dtype=np.uint8, copy=True)
        img = flat.reshape(28, 28)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)

        return img, label

    @classmethod
    def list_categories(
        cls, root: str | Path, download: bool = True
    ) -> list[str]:
        """Return the official 345-category list, downloading if needed.

        Cached on disk at ``<root>/categories.txt`` and in-memory on the class
        so subsequent calls are free.
        """
        root = Path(root)
        if root in cls._all_categories_cache:
            return cls._all_categories_cache[root]

        root.mkdir(parents=True, exist_ok=True)
        path = root / "categories.txt"
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{path} not found and download=False."
                )
            print(f"[QuickDraw] downloading category list -> {path}")
            urllib.request.urlretrieve(QUICKDRAW_CATEGORIES_URL, path)

        names = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        cls._all_categories_cache[root] = names
        return names

    # ---- internals --------------------------------------------------------

    def _local_path(self, category: str) -> Path:
        return self.root / f"{category}.npy"

    @staticmethod
    def _download(category: str, dest: Path) -> None:
        url = f"{QUICKDRAW_BASE_URL}/{urllib.parse.quote(category)}.npy"
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"[QuickDraw] downloading {category!r} -> {dest}")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)


def make_quickdraw_loader(
    root: str | Path,
    categories: Optional[Sequence[str]] = None,
    train: bool = True,
    batch_size: int = 128,
    max_per_category: Optional[int] = 10_000,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    **loader_kwargs,
) -> DataLoader:
    """Convenience builder for a typical loader.

    `shuffle` defaults to True for train, False for test.
    """
    if shuffle is None:
        shuffle = train

    dataset = QuickDrawDataset(
        root=root,
        categories=categories,
        train=train,
        max_per_category=max_per_category,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        **loader_kwargs,
    )


if __name__ == "__main__":
    common = dict(
        root="./data/quickdraw",
        categories=["cat", "dog", "airplane"],
        batch_size=64,
        max_per_category=5_000,
        num_workers=2,
    )
    train_loader = make_quickdraw_loader(train=True, **common)
    test_loader = make_quickdraw_loader(train=False, **common)

    images, labels = next(iter(train_loader))
    print(f"train | images={tuple(images.shape)} labels={tuple(labels.shape)}")
    print(
        f"  train size={len(train_loader.dataset)}  "
        f"test size={len(test_loader.dataset)}"
    )
    print(
        f"  total categories available: "
        f"{len(train_loader.dataset.all_categories)}"
    )