#!/usr/bin/env python3
"""Utility to tune entailment eta using a SUN397 validation split.

Mirrors the CUB tuning workflow: compute entailment features for each
eta in the provided grid, sweep lambda values to train linear probes,
record validation accuracy, and persist a JSON summary for downstream
analysis.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets

from models.entailment_cbm import entailment_cone_feature_matrix
from models import get_model
from hypcbm.train_hypcbm import (
    get_stratified_subset,
    load_concept_tensors,
    load_or_compute_visual_embeddings,
    run_linear_probe,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune entailment eta on SUN397 by sweeping lambda values over a held-out validation split.",
    )
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank (pkl).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to store tuning summaries.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sun397",
        help="Dataset identifier for caching (default: sun397).",
    )
    parser.add_argument(
        "--sun397-root",
        type=str,
        default=str(REPO_ROOT / "data" / "sun397" / "SUN_dataset" / "data"),
        help="Root directory pointing to SUN_dataset/data.",
    )
    parser.add_argument("--sun397-download", action="store_true", help="Allow torchvision to download SUN397 if absent.")
    parser.add_argument("--eta-grid", nargs="+", type=float, required=True, help="List of eta values to evaluate.")
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        required=True,
        help="Lambda values for the linear-probe sweep (logspace recommended).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--backbone-name", type=str, default="meru")
    parser.add_argument("--hycoclip-config", type=str, default="hycoclip/configs/train_meru_vit_b.py")
    parser.add_argument("--hycoclip-checkpoint", type=str, default="hycoclip/meru_vit_b.pth")
    parser.add_argument("--embedding-cache-dir", type=str, default="outputs/visual_embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.2,
        help="Fraction of training samples used before forming the validation split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the (possibly subsampled) set reserved for validation.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="sun397_eta_tuning.json",
        help="Filename for the tuning summary JSON (saved under out-dir).",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-2,
        help="Absolute weight magnitude threshold for counting concepts as active.",
    )
    return parser.parse_args()


def _format_fraction(value: float) -> str:
    formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _format_value_token(value: float) -> str:
    return _format_fraction(value).replace(".", "p")


def _resolve_sun_root(path_str: str, *, download: bool) -> Path:
    root = Path(path_str).expanduser().resolve()
    if not root.exists() and not download:
        raise FileNotFoundError(
            f"SUN397 root '{root}' not found. Pass --sun397-root or enable --sun397-download."
        )
    return root


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    eta_grid = [float(x) for x in args.eta_grid]
    lambda_grid = [max(float(x), 1e-8) for x in args.lambda_grid]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_path = Path(args.concept_bank)
    if not concept_path.exists():
        raise FileNotFoundError(f"Concept bank not found at {concept_path}")

    print("Loading concept bank metadata...")
    all_concepts = pickle.load(concept_path.open("rb"))

    print("Initializing HyCoCLIP backbone...")
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(device).eval()

    print("Preparing SUN397 dataset...")
    sun_root = _resolve_sun_root(args.sun397_root, download=args.sun397_download)
    sun_dataset = datasets.SUN397(
        root=str(sun_root),
        transform=preprocess,
        download=args.sun397_download,
    )

    print("Loading or computing SUN397 embeddings...")
    train_embs, train_labels = load_or_compute_visual_embeddings(
        args,
        backbone,
        device,
        sun_dataset,
        split="train",
    )
    train_embs = np.asarray(train_embs)
    train_labels = np.asarray(train_labels)

    if args.train_fraction < 1.0:
        print(f"Subsampling training data to fraction={args.train_fraction:.3f}")
        train_embs, train_labels = get_stratified_subset(
            (train_embs, train_labels),
            args.train_fraction,
            args.seed,
        )
        train_embs = np.asarray(train_embs)
        train_labels = np.asarray(train_labels)

    print("Loading concept tensors...")
    concept_names, concept_tensor, text_norms = load_concept_tensors(
        args.concept_bank,
        device,
        backbone,
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=max(1e-6, float(args.val_fraction)),
        random_state=args.seed,
    )
    train_idx, val_idx = next(splitter.split(train_embs, train_labels))
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    print(
        f"Training samples: {len(train_idx)} | Validation samples: {len(val_idx)} "
        f"(val_fraction={args.val_fraction:.3f})"
    )

    all_results: List[Dict] = []
    best_global: Dict | None = None
    per_eta_paths: List[str] = []
    results_stem = Path(args.results_name).stem

    for eta_value in eta_grid:
        print(f"\n=== Evaluating eta={eta_value:.3f} ===")
        sweep_args = copy.deepcopy(args)
        sweep_args.entail_eta = float(eta_value)

        train_features = entailment_cone_feature_matrix(
            backbone,
            train_embs,
            concept_tensor,
            chunk_size=int(args.chunk_size),
            device=device,
            eta=sweep_args.entail_eta,
        )

        train_dataset = (train_features[train_idx], train_labels[train_idx])
        val_dataset = (train_features[val_idx], train_labels[val_idx])

        lam_results: List[Dict] = []
        best_entry: Dict | None = None
        ordered_lams = sorted(lambda_grid, reverse=True)
        for lam_value in ordered_lams:
            probe_args = copy.deepcopy(sweep_args)
            probe_args.lam = float(lam_value)

            run_info, weights, _ = run_linear_probe(
                probe_args,
                train_dataset,
                val_dataset,
            )

            weights_arr = np.atleast_2d(np.asarray(weights))
            active_mask = np.abs(weights_arr) > float(args.active_threshold)
            concepts_per_class = active_mask.sum(axis=1).astype(int).tolist()
            active_total = int(active_mask.sum())

            entry = {
                "eta": float(eta_value),
                "lambda": float(lam_value),
                "val_acc": float(run_info["test_acc"]),
                "train_acc": float(run_info["train_acc"]),
                "active_concepts_total": active_total,
                "active_concepts_per_class": concepts_per_class,
            }
            lam_results.append(entry)

            if best_entry is None or entry["val_acc"] > best_entry["val_acc"]:
                best_entry = entry
                print(
                    f"New best for eta={eta_value:.3f}: lam={lam_value:.2e}, "
                    f"val_acc={entry['val_acc']:.2f}%, active={active_total}",
                )

        eta_summary = {
            "eta": float(eta_value),
            "lambda_results": lam_results,
            "best_lambda": best_entry["lambda"] if best_entry else None,
            "best_val_acc": best_entry["val_acc"] if best_entry else None,
            "best_train_acc": best_entry["train_acc"] if best_entry else None,
            "active_threshold": float(args.active_threshold),
        }
        all_results.append(eta_summary)

        eta_token = _format_value_token(float(eta_value))
        eta_results_path = out_dir / f"{results_stem}_eta{eta_token}.json"
        with eta_results_path.open("w", encoding="utf-8") as handle:
            json.dump(eta_summary, handle, indent=2)
        per_eta_paths.append(str(eta_results_path))

        if best_entry and (best_global is None or best_entry["val_acc"] > best_global["best_val_acc"]):
            best_global = {
                "eta": float(eta_value),
                "lambda": best_entry["lambda"],
                "best_val_acc": best_entry["val_acc"],
                "train_acc": best_entry["train_acc"],
            }

    summary = {
        "eta_grid": eta_grid,
        "lambda_grid": lambda_grid,
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "active_threshold": float(args.active_threshold),
        "best_overall": best_global,
        "results": all_results,
        "per_eta_files": per_eta_paths,
    }

    results_path = out_dir / args.results_name
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nEta tuning complete.")
    if best_global:
        print(
            f"Best configuration: eta={best_global['eta']:.3f}, "
            f"lambda={best_global['lambda']:.2e}, val_acc={best_global['best_val_acc']:.2f}%"
        )
    print(f"Detailed results written to {results_path}")


if __name__ == "__main__":
    main()
