"""Entailment-cone feature construction for HyCoCLIP CBMs."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from hycoclip.hycoclip import lorentz as L


@torch.inference_mode()
def entailment_cone_feature_matrix(
    model,
    image_embeddings: np.ndarray,
    concept_feats: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    eta: float = 1.0,
) -> np.ndarray:
    if concept_feats.device != device:
        concept_feats = concept_feats.to(device)
    concept_feats = concept_feats.detach()

    if concept_feats.ndim != 2:
        raise ValueError("concept_feats must be 2-D")

    emb_dim = image_embeddings.shape[-1]
    if concept_feats.shape[-1] != emb_dim:
        if concept_feats.shape[0] == emb_dim:
            concept_feats = concept_feats.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "Concept embeddings must align with image embeddings: "
                f"concept_feats={concept_feats.shape}, image_embs={image_embeddings.shape}"
            )

    curvature = model.curv.exp()

    cone_aperture = L.half_aperture(concept_feats, curvature, min_radius=0.04).view(1, -1)
    total_items = int(image_embeddings.shape[0])
    total_batches = max(1, (total_items + chunk_size - 1) // chunk_size)
    output = np.empty((total_items, concept_feats.size(0)), dtype=np.float32)

    for batch_idx, start in enumerate(range(0, total_items, chunk_size)):
        stop = min(start + chunk_size, total_items)
        batch_np = np.asarray(image_embeddings[start:stop], dtype=np.float32)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=concept_feats.dtype)
        phi = L.pairwise_oxy_angle(concept_feats, batch, curv=curvature)

        # hard entailment: clamped at 0
        margin = torch.clamp((eta*cone_aperture - phi) / cone_aperture, min=0.0)

        # soft entailment: allow margin of inclusion to be negative
        # margin = (eta*cone_aperture - phi) / cone_aperture
        
        # distance based: compute Lorentzian distance to cone boundary
        # margin = L.pairwise_dist(batch, concept_feats, curv=curvature)

        print('margin details' , ' min, max mean std', margin.min().item(), margin.max().item(), margin.mean().item(), margin.std().item())
        output[start:stop] = margin.cpu().numpy()

        pct = 100.0 * (batch_idx + 1) / total_batches
        print(
            f"Entailment-cone feature progress: {batch_idx + 1}/{total_batches} batches ({pct:.1f}%)",
            end="\r" if batch_idx + 1 < total_batches else "\n",
        )

    return output
