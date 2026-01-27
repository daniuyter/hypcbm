"""Common utility functions for hypcbm."""

import random
import re
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as tv_transforms
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_identifier(value: str) -> str:
    """Sanitize a string to be used as a file identifier."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def format_fraction(value: float) -> str:
    """Format a fraction value for use in filenames."""
    formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def infer_normalize_stats(
    transform, seen: Optional[set[int]] = None
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Walk a torchvision-style transform to find a Normalize mean/std.

    A small guard prevents accidental self-referential transform graphs from
    recursing indefinitely (observed when a transform has a .transforms that
    points back to itself).
    """
    if seen is None:
        seen = set()

    if transform is None:
        return None, None

    obj_id = id(transform)
    if obj_id in seen:
        return None, None
    seen.add(obj_id)

    if hasattr(transform, "mean") and hasattr(transform, "std"):
        mean = torch.as_tensor(transform.mean, dtype=torch.float32)
        std = torch.as_tensor(transform.std, dtype=torch.float32)
        return mean, std

    if isinstance(transform, tv_transforms.Compose):
        sequence = transform.transforms
    elif isinstance(transform, (list, tuple)):
        sequence = list(transform)
    elif hasattr(transform, "transforms"):
        sequence = list(transform.transforms)
    else:
        sequence = [transform]

    for item in reversed(sequence):
        mean, std = infer_normalize_stats(item, seen)
        if mean is not None:
            return mean, std
    return None, None


def denormalize_for_display(
    tensor: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Denormalize a tensor for display purposes."""
    img = tensor.detach().clone()
    if mean is not None and std is not None and img.shape[0] == mean.shape[0]:
        img = img * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    img = img.clamp(0.0, 1.0)
    return img


def get_stratified_subset(dataset, fraction: float, seed: int):
    """Return a stratified subset of the dataset preserving class balance."""
    features, labels = dataset
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("train_fraction must be in the (0, 1] range.")
    if fraction >= 0.9999:
        return features, labels

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    indices, _ = next(splitter.split(features, labels))
    print(f"Sampling {len(indices)} / {len(labels)} training examples (fraction={fraction:.4f}).")
    return features[indices], labels[indices]
