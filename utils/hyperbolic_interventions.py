"""Utilities for intervening on hyperbolic concept activations."""

from __future__ import annotations
from typing import Optional
import torch
from hycoclip.hycoclip import lorentz as L

def calculate_half_aperture(concept_vectors: torch.Tensor, model) -> torch.Tensor:
    """Fallback wrapper that dispatches to HyCoCLIP's half-aperture implementation."""
    curvature = model.curv.exp()
    return L.half_aperture(concept_vectors, curvature)

def calculate_exterior_angle(vectors_A: torch.Tensor, vectors_B: torch.Tensor, model) -> torch.Tensor:
    """Fallback wrapper leveraging HyCoCLIP's oxy-angle (positive) evaluation."""
    curvature = model.curv.exp()
    if vectors_B.shape[0] == 1 and vectors_A.shape[0] != 1:
        vectors_B = vectors_B.expand_as(vectors_A)
    elif vectors_A.shape[0] == 1 and vectors_B.shape[0] != 1:
        vectors_A = vectors_A.expand_as(vectors_B)
    elif vectors_A.shape[0] != vectors_B.shape[0]:
        raise ValueError("vectors_A and vectors_B must have broadcastable leading dimensions")
    return L.oxy_angle(vectors_A, vectors_B, curvature)


def propagate_intervention(
    activations: torch.Tensor,
    concept_vectors: torch.Tensor,
    parent_idx: int,
    model,
    eta: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Propagate a manual intervention down the hyperbolic concept hierarchy.

    When a parent concept is zeroed-out by the user, every geometrically entailed
    child concept (i.e. concepts whose exterior angle to the parent lies inside
    the parent's entailment cone) must also be zeroed. This function performs the
    cascade in a fully vectorized manner.

    Args:
        activations: Concept activation tensor of shape [batch, n_concepts].
        concept_vectors: Hyperbolic concept vectors of shape [n_concepts, dim].
        parent_idx: Index of the manually intervened parent concept.
        model: Hyperbolic text model that provides curvature parameters.
        eta: Scaling factor controlling the cone size (smaller -> stricter entailment).
             If None, an adaptive eta is calculated based on the concept norm.
        mask: Optional boolean mask of trainable concepts (shape [n_concepts]). If
            provided, only masked concepts are considered for propagation.

    Returns:
        A cloned tensor containing the updated activations with the intervention
        propagated to every entailed child concept.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D tensor of shape [batch, n_concepts]")
    n_concepts = concept_vectors.shape[0]
    if not 0 <= parent_idx < n_concepts:
        raise IndexError(f"parent_idx {parent_idx} is out of bounds for {n_concepts} concepts")

    updated = activations.clone()
    updated[:, parent_idx] = 0.0  # manual intervention on the parent itself

    # Compute the half-aperture (ω) of every concept's entailment cone.
    half_apertures = calculate_half_aperture(concept_vectors, model).reshape(-1)
    parent_aperture = half_apertures[parent_idx]

    # Exterior angle ϕ(x_j, x_parent) for every concept j relative to the parent.
    parent_vec = concept_vectors[parent_idx].unsqueeze(0)
    exterior_angles = calculate_exterior_angle(concept_vectors, parent_vec, model).reshape(-1)

    # Adaptive eta calculation
    parent_norm = torch.norm(concept_vectors[parent_idx], p=2)

    # we can add + 0.05 , 0.1, 0.15 etc. to propagate intervention to more child concepts 
    adaptive_eta = 8.62 * (parent_norm - 0.09) # + 0.05
    
    # use adaptive eta
    effective_eta = adaptive_eta if eta is None else eta

    # A child is any concept whose angle lies strictly inside the parent's cone: ϕ < η · ω_parent.
    threshold = effective_eta * parent_aperture
    child_mask = exterior_angles < threshold
    child_mask[parent_idx] = False  # never reapply the parent itself

    if mask is not None:
        if mask.shape[0] != n_concepts:
            raise ValueError("mask must have shape [n_concepts]")
        child_mask &= mask.to(dtype=torch.bool, device=child_mask.device)

    child_indices = torch.nonzero(child_mask, as_tuple=False).squeeze(1)
    if child_indices.numel() == 0:
        return updated

    child_indices = child_indices.to(updated.device)
    updated.index_fill_(dim=1, index=child_indices, value=0.0)
    return updated
