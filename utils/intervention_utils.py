"""Shared utilities for concept intervention workflows."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from hyperbolic_interventions import propagate_intervention


def apply_interventions_to_features(
    features: np.ndarray,
    interventions: List[Dict],
    concept_vectors: torch.Tensor,
    model,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Apply manual interventions (with optional propagation) to feature matrices.

    Args:
        features: Concept activation matrix (N, C).
        interventions: List of dicts, each containing 'idx', 'value', and optional metadata.
        concept_vectors: Tensor of concept embeddings (C, D).
        model: Object exposing ``curv.exp()`` (e.g., HyCoCLIP backbone wrapper).
        eta: Entailment scaling factor for propagation. Ignored if adaptive eta is used.
    Returns:
        A NumPy array with the same dtype as ``features`` reflecting the intervention.
    """
    if not interventions:
        return features

    feat_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32, order="C"))
    feat_tensor = feat_tensor.to(device=concept_vectors.device, dtype=concept_vectors.dtype)

    for intervention in interventions:
        idx = int(intervention["idx"])
        value = float(intervention.get("value", 0.0))
        feat_tensor[:, idx] = value
        if abs(value) <= 1e-8:
            feat_tensor = propagate_intervention(
                activations=feat_tensor,
                concept_vectors=concept_vectors,
                parent_idx=idx,
                model=model,
            )

    return feat_tensor.detach().to("cpu").numpy().astype(features.dtype, copy=False)
