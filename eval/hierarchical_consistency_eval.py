"""Hierarchical consistency evaluation for HypCBM (hyperbolic) vs. PCBM (Euclidean).

- `concept_names`: list[str], length K (column order aligns with activation matrices).
- `hyp_acts`: numpy.ndarray | torch.Tensor, shape (N, K), sparse activations for HypCBM.
- `pcbm_acts`: numpy.ndarray | torch.Tensor, shape (N, K), dense cosine scores for PCBM.

Outputs a summary dictionary and prints key metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def canonicalize_name(raw: str) -> str:
    """Lowercase, replace separators, lemmatize tokens, and collapse spaces."""
    cleaned = raw.strip().lower().replace("_", " ").replace("-", " ")
    tokens = [lemmatizer.lemmatize(tok) for tok in cleaned.split() if tok]
    return " ".join(tokens)


def synsets_for_name(name: str):
    """Return noun synsets for a concept name, trying a few surface variants."""
    variants = {name}
    variants.add(name.replace(" ", "_"))
    variants.add(name.replace("_", " "))
    synsets = []
    for cand in variants:
        synsets.extend(wn.synsets(cand, pos=wn.NOUN))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in synsets:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


@dataclass
class PairMiningResult:
    pairs: List[Tuple[int, int]]
    normalized_names: List[str]
    matched_concepts: int
    unmatched_concepts: int


def mine_hypernym_pairs(concept_names: Sequence[str]) -> PairMiningResult:
    """Mine (child_idx, parent_idx) pairs where parent is a WordNet hypernym of child.

    Only keeps pairs where the parent concept also appears in `concept_names`
    (after normalization).
    """
    normalized = [canonicalize_name(name) for name in concept_names]
    index: Dict[str, int] = {}
    for idx, name in enumerate(normalized):
        index.setdefault(name, idx)  # keep first occurrence

    pairs = set()
    matched = 0
    unmatched = 0
    for child_idx, canon_name in enumerate(tqdm(normalized, desc="Mining WordNet pairs")):
        synsets = synsets_for_name(canon_name)
        if not synsets:
            unmatched += 1
            continue
        matched += 1
        for syn in synsets:
            for ancestor in syn.closure(lambda s: s.hypernyms()):
                anc_name = canonicalize_name(ancestor.name().split(".")[0])
                parent_idx = index.get(anc_name)
                if parent_idx is None or parent_idx == child_idx:
                    continue
                pairs.add((child_idx, parent_idx))
    return PairMiningResult(
        pairs=sorted(pairs),
        normalized_names=normalized,
        matched_concepts=matched,
        unmatched_concepts=unmatched,
    )


# ------------------------- Sparsity calibration -------------------------

def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def binarize_sparse(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return (x > threshold).to(torch.bool)


def calibrate_threshold_to_kavg(dense_scores: torch.Tensor, target_kavg: float) -> Tuple[float, torch.Tensor, float]:
    """Find a scalar threshold so dense_scores > T matches target_kavg (avg act/img)."""
    dense = dense_scores.float()
    n, k = dense.shape
    total_needed = target_kavg * float(n)
    if total_needed <= 0:
        mask = torch.zeros_like(dense, dtype=torch.bool)
        return float("inf"), mask, 0.0

    flat = dense.reshape(-1)
    keep_fraction = min(max(total_needed / float(flat.numel()), 0.0), 1.0)
    q = 1.0 - keep_fraction
    q = float(min(max(q, 0.0), 1.0))
    threshold = float("nan")
    if flat.numel():
        flat_cpu = flat.detach().cpu()
        try:
            threshold = float(torch.quantile(flat_cpu, q))
        except RuntimeError:
            threshold = float(np.quantile(flat_cpu.numpy(), q))

    mask = dense > threshold
    achieved_kavg = mask.sum(dim=1).float().mean().item()
    return threshold, mask, achieved_kavg


# ------------------------- Consistency scoring -------------------------

def compute_consistency(binary_mask: torch.Tensor, pairs: Iterable[Tuple[int, int]]):
    """Compute total child activations, violations, and consistency score."""
    if binary_mask.numel() == 0:
        return 0, 0, float("nan")
    total_child = 0
    violations = 0
    for child_idx, parent_idx in pairs:
        child_active = binary_mask[:, child_idx]
        parent_active = binary_mask[:, parent_idx]
        child_count = int(child_active.sum().item())
        total_child += child_count
        if child_count == 0:
            continue
        violations += int((child_active & (~parent_active)).sum().item())
    if total_child == 0:
        score = float("nan")
    else:
        score = 1.0 - (violations / float(total_child))
    return total_child, violations, score


# ------------------------- Main evaluation -------------------------

def evaluate_hierarchical_consistency(
    hyp_acts,
    pcbm_acts,
    concept_names: Sequence[str],
    hyp_threshold: float = 1e-5,
):
    """
    Evaluate hierarchical consistency for HypCBM and PCBM.

    Returns a summary dict and prints key metrics.
    """
    hyp = to_tensor(hyp_acts)
    pcbm = to_tensor(pcbm_acts)
    if hyp.shape != pcbm.shape:
        raise ValueError(f"Shape mismatch: hyp {hyp.shape} vs pcbm {pcbm.shape}")

    hyp = hyp.float()
    pcbm = pcbm.float()

    mining = mine_hypernym_pairs(concept_names)
    pairs = mining.pairs

    # HypCBM binarization
    hyp_mask = binarize_sparse(hyp, hyp_threshold)
    hyp_kavg = hyp_mask.sum(dim=1).float().mean().item()

    # PCBM threshold calibration
    pcbm_threshold, pcbm_mask, pcbm_kavg = calibrate_threshold_to_kavg(pcbm, hyp_kavg)

    # Consistency scores
    hyp_child, hyp_viol, hyp_score = compute_consistency(hyp_mask, pairs)
    pcbm_child, pcbm_viol, pcbm_score = compute_consistency(pcbm_mask, pairs)

    summary = {
        "num_images": int(hyp.shape[0]),
        "num_concepts": int(hyp.shape[1]),
        "hyp_threshold": hyp_threshold,
        "hyp_k_avg": hyp_kavg,
        "pcbm_threshold": pcbm_threshold,
        "pcbm_k_avg": pcbm_kavg,
        "num_pairs": len(pairs),
        "wordnet": {
            "matched_concepts": mining.matched_concepts,
            "unmatched_concepts": mining.unmatched_concepts,
        },
        "pairs": pairs,
        "hyp": {
            "child_activations": hyp_child,
            "violations": hyp_viol,
            "consistency": hyp_score,
        },
        "pcbm": {
            "child_activations": pcbm_child,
            "violations": pcbm_viol,
            "consistency": pcbm_score,
        },
    }

    # Human-readable prints
    print(f"Found {len(pairs)} WordNet hypernym pairs.")
    print(f"WordNet coverage: {mining.matched_concepts} matched, {mining.unmatched_concepts} unmatched.")
    print(f"HypCBM k_avg: {hyp_kavg:.4f} (threshold={hyp_threshold})")
    print(f"PCBM calibrated threshold: {pcbm_threshold:.6f}, achieved k_avg={pcbm_kavg:.4f}")
    print(
        f"HypCBM consistency: {hyp_score * 100:.2f}% | child activations={hyp_child}, violations={hyp_viol}"
    )
    print(
        f"PCBM consistency: {pcbm_score * 100:.2f}% | child activations={pcbm_child}, violations={pcbm_viol}"
    )

    return summary

def main():
    hyp_acts_path = '/path/to/hypcbm_acts.pt'
    pcbm_acts_path = '/path/to/pcbm_acts.pt'
    hypcbm = torch.load(hyp_acts_path, map_location='cpu', weights_only=False)
    concept_names = hypcbm['concept_names']
    hyp_acts = hypcbm['acts']
    pcbm = torch.load(pcbm_acts_path, map_location='cpu', weights_only=False)
    pcbm_acts = pcbm['acts']
    summary = evaluate_hierarchical_consistency(hyp_acts, pcbm_acts, concept_names)
    print(summary['pairs'])

if __name__ == "__main__":
    main()