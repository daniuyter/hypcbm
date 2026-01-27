"""Concept loading and processing utilities for hypcbm."""

import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Add hycoclip to path
REPO_ROOT = Path(__file__).resolve().parent.parent
HYCOCLIP_ROOT = REPO_ROOT / "hycoclip"
if str(HYCOCLIP_ROOT) not in sys.path:
    sys.path.append(str(HYCOCLIP_ROOT))

from hycoclip.hycoclip import lorentz as L


def extract_concept_vector(entry) -> np.ndarray:
    """Return a numpy array for a single concept entry."""
    if isinstance(entry, dict):
        for key in ("tensor", "text_features", "embedding", "vector", "cav"):
            if key in entry:
                entry = entry[key]
                break

    if isinstance(entry, (list, tuple)):
        if not entry:
            raise ValueError("Empty concept entry encountered.")
        entry = entry[0]

    if isinstance(entry, torch.Tensor):
        entry = entry.detach().cpu().numpy()

    array = np.asarray(entry, dtype=np.float32).squeeze()
    if array.ndim == 0:
        raise ValueError("Concept entry did not contain a valid vector.")
    return array


def maybe_resolve_concept_pickle(path: Path) -> dict:
    """Load a concept dictionary, following pointers stored inside checkpoints if needed."""
    resolved = Path(path).expanduser()
    with resolved.open("rb") as handle:
        obj = pickle.load(handle)

    # Direct concept dict
    if isinstance(obj, dict):
        sample = next(iter(obj.values()), None)
        if sample is not None:
            try:
                extract_concept_vector(sample)
                return obj
            except Exception:
                pass

        checkpoint_path = obj.get("concept_bank") if isinstance(obj.get("concept_bank"), str) else None
        if checkpoint_path:
            nested_path = Path(checkpoint_path)
            if not nested_path.is_absolute():
                nested_path = (resolved.parent / nested_path).resolve()
            if not nested_path.exists():
                raise FileNotFoundError(
                    f"Concept checkpoint refers to missing concept bank at {nested_path}"
                )
            print(
                f"[info] {resolved.name} looks like a checkpoint; reloading concept bank from {nested_path}"
            )
            with nested_path.open("rb") as handle:
                nested_obj = pickle.load(handle)
            if not isinstance(nested_obj, dict):
                raise ValueError(f"Concept bank at {nested_path} is not a dictionary")
            return nested_obj

    raise ValueError(
        "Provided concept file does not contain raw concept embeddings. "
        "Pass a concept bank .pkl exported by encode_concepts.py"
    )


def compute_tangent_norms(
    embeddings: torch.Tensor,
    model,
    curvature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tangent space norms for hyperbolic embeddings."""
    curv = curvature if curvature is not None else model.curv.exp()
    tangent = L.log_map0(embeddings, curv)
    return torch.linalg.norm(tangent, dim=-1)


def load_concept_tensors(
    path: str,
    device: torch.device,
    model,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Load concept tensors from a pickle file.
    
    Returns:
        concept_names: List of concept name strings
        concept_tensor: Tensor of concept embeddings
        text_norms: Tensor of tangent space norms
    """
    concept_obj = maybe_resolve_concept_pickle(Path(path))
    concept_names = sorted(concept_obj.keys())
    concept_array = np.stack([
        extract_concept_vector(concept_obj[name]) for name in concept_names
    ], axis=0)
    concept_tensor = torch.from_numpy(concept_array).to(device=device)
    curvature = model.curv.exp()
    text_norms = compute_tangent_norms(concept_tensor, model, curvature)
    return concept_names, concept_tensor, text_norms


def encode_batch_images(args, backbone, images: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images using the backbone model."""
    backbone_name = getattr(args, "backbone_name", "")
    if "hycoclip" in backbone_name.lower():
        embeddings = backbone.encode_image(images, project=True)
    elif "clip" in backbone_name.lower():
        embeddings = backbone.encode_image(images, project=True)
    elif "meru" in backbone_name.lower():
        embeddings = backbone.encode_image(images, project=True)
    else:
        embeddings = backbone(images)
    return embeddings.detach().float()
