"""Utility helpers for CUB image loading compatible with CLIP/HyCoCLIP preprocess."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def _read_cub_splits(cub_root: Path) -> Tuple[List[str], List[str]]:
    images_file = cub_root / "images.txt"
    split_file = cub_root / "train_test_split.txt"
    if not images_file.exists() or not split_file.exists():
        raise FileNotFoundError(
            "CUB metadata not found. Expected images.txt and train_test_split.txt under"
            f" {cub_root}."
        )

    image_map: Dict[int, str] = {}
    with images_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_str, rel_path = line.split(" ", 1)
            image_map[int(idx_str)] = rel_path.strip()

    train_paths: List[str] = []
    test_paths: List[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_str, is_train = line.split(" ", 1)
            rel_path = image_map.get(int(idx_str))
            if rel_path is None:
                continue
            (train_paths if is_train.strip() == "1" else test_paths).append(rel_path)

    return train_paths, test_paths


def _index_dataset(dataset: datasets.ImageFolder, image_root: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, (path, _label) in enumerate(dataset.samples):
        rel = Path(path).resolve().relative_to(image_root).as_posix()
        mapping[rel] = idx
    return mapping


def _subset_from_rel_paths(dataset: datasets.ImageFolder, indices: Sequence[str], mapping: Dict[str, int]) -> Subset:
    subset_indices = []
    for rel in indices:
        key = rel.replace("\\", "/")
        if key not in mapping:
            raise KeyError(f"Image {rel} from split file not found under dataset root")
        subset_indices.append(mapping[key])
    return Subset(dataset, subset_indices)


def build_cub_dataloaders(
    cub_root: str | Path,
    transform,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, Dict[int, str], List[str]]:
    """Return train/test loaders that respect the official CUB split files."""

    cub_root = Path(cub_root).expanduser().resolve()
    images_root = cub_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"CUB images directory not found at {images_root}")

    dataset = datasets.ImageFolder(images_root, transform=transform)
    path_to_idx = _index_dataset(dataset, images_root)
    train_rel, test_rel = _read_cub_splits(cub_root)

    train_subset = _subset_from_rel_paths(dataset, train_rel, path_to_idx)
    test_subset = _subset_from_rel_paths(dataset, test_rel, path_to_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    classes = dataset.classes
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    return train_loader, test_loader, idx_to_class, classes
