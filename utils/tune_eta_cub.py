#!/usr/bin/env python3
"""Utility to tune entailment eta using the CUB validation set.

For each eta in the provided grid this script computes entailment-cone
features, performs a lambda sweep identical to the visualization routine in
``train_pcbm_entailment.py``, and reports the validation accuracy. Results are
persisted as JSON for further analysis.
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

from data.cub_loader import build_cub_dataloaders
from models import get_model
from models.entailment_cbm import entailment_cone_feature_matrix
from hypcbm.train_hypcbm import (
    get_stratified_subset,
    load_concept_tensors,
    load_or_compute_visual_embeddings,
    run_linear_probe,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune entailment eta on the CUB validation split by sweeping lambda values.",
    )
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank (pkl).")
    parser.add_argument("--out-dir", required=True, type=str, help="Directory to store tuning summaries.")
    parser.add_argument("--dataset", type=str, default="cub", help="Dataset identifier used for caching (default: cub).")
    parser.add_argument("--cub-root", type=str, default=str(Path("CUB/CUB_200_2011").resolve()))
    parser.add_argument("--eta-grid", nargs="+", type=float, required=True, help="List of eta values to evaluate.")
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        required=True,
        help="Lambda values used during the linear probe sweep (logspace recommended).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size used when constructing entailment features.")
    parser.add_argument("--backbone-name", type=str, default="hycoclip")
    parser.add_argument("--hycoclip-config", type=str, default="hycoclip/configs/train_hycoclip_vit_b.py")
    parser.add_argument("--hycoclip-checkpoint", type=str, default="hycoclip/hycoclip_vit_b.pth")
    parser.add_argument("--embedding-cache-dir", type=str, default="outputs/visual_embeddings")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the (possibly subsampled) training set reserved for validation.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="cub_eta_tuning.json",
        help="Filename for the tuning summary JSON (saved under out-dir).",
    )
    return parser.parse_args()


def _format_fraction(value: float) -> str:
    formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    eta_grid = [float(x) for x in args.eta_grid]
    lambda_grid = [max(float(x), 1e-8) for x in args.lambda_grid]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading concept bank metadata...")
    if not Path(args.concept_bank).exists():
        raise FileNotFoundError(f"Concept bank not found at {args.concept_bank}")
    all_concepts = pickle.load(open(args.concept_bank, "rb"))

    print("Initializing HyCoCLIP backbone...")
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(device).eval()

    print("Preparing CUB dataloaders...")
    train_loader, _, idx_to_class, classes = build_cub_dataloaders(
        args.cub_root,
        preprocess,
        args.batch_size,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Loading or computing CUB train embeddings...")
    train_embs, train_labels = load_or_compute_visual_embeddings(
        args,
        backbone,
        device,
        train_loader.dataset,
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

            run_info, _, _ = run_linear_probe(
                probe_args,
                train_dataset,
                val_dataset,
            )

            entry = {
                "eta": float(eta_value),
                "lambda": float(lam_value),
                "val_acc": float(run_info["test_acc"]),
                "train_acc": float(run_info["train_acc"]),
            }
            lam_results.append(entry)

            if best_entry is None or entry["val_acc"] > best_entry["val_acc"]:
                best_entry = entry
                print(
                    f"New best for eta={eta_value:.3f}: lam={lam_value:.2e}, "
                    f"val_acc={entry['val_acc']:.2f}%",
                )

        eta_summary = {
            "eta": float(eta_value),
            "lambda_results": lam_results,
            "best_lambda": best_entry["lambda"] if best_entry else None,
            "best_val_acc": best_entry["val_acc"] if best_entry else None,
            "best_train_acc": best_entry["train_acc"] if best_entry else None,
        }
        all_results.append(eta_summary)

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
        "best_overall": best_global,
        "results": all_results,
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
