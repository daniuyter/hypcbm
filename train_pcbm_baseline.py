import argparse
import copy
import json
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import get_dataset
from data.cub_loader import build_cub_dataloaders
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections

from utils.common import (
    sanitize_identifier,
    format_fraction,
    get_stratified_subset,
)
from utils.concept_utils import encode_batch_images

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_cache_dir(args: argparse.Namespace) -> Path:
    if args.embedding_cache_dir is not None:
        cache_dir = Path(args.embedding_cache_dir)
    else:
        cache_dir = Path(args.out_dir) / "visual_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_or_compute_visual_embeddings(
    args: argparse.Namespace,
    backbone,
    device: torch.device,
    dataset,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir = _resolve_cache_dir(args)
    prefix = f"{sanitize_identifier(args.dataset)}__{sanitize_identifier(args.backbone_name)}"
    total = len(dataset)
    suffix = f"__{split}_n{total}"
    emb_path = cache_dir / f"{prefix}{suffix}_embeddings.npy"
    lbl_path = cache_dir / f"{prefix}{suffix}_labels.npy"

    if emb_path.exists() and lbl_path.exists():
        print(f"Loaded cached {split} embeddings from {emb_path}")
        embeddings = np.load(emb_path, mmap_mode="r")
        labels = np.load(lbl_path, mmap_mode="r")
        return embeddings, labels

    batch_size = args.embedding_batch_size or args.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    embeddings_mm = None
    labels_mm = None
    offset = 0

    for images, labels in tqdm(dataloader, desc=f"Computing {split} embeddings"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_emb = encode_batch_images(args, backbone, images)
        batch_np = batch_emb.cpu().numpy().astype(np.float32, copy=False)
        label_np = labels.cpu().numpy().astype(np.int64, copy=False)

        if embeddings_mm is None:
            emb_dim = batch_np.shape[1]
            embeddings_mm = np.lib.format.open_memmap(
                emb_path,
                mode="w+",
                dtype=np.float32,
                shape=(total, emb_dim),
            )
            labels_mm = np.lib.format.open_memmap(
                lbl_path,
                mode="w+",
                dtype=np.int64,
                shape=(total,),
            )

        end = offset + batch_np.shape[0]
        embeddings_mm[offset:end] = batch_np
        labels_mm[offset:end] = label_np
        offset = end

    if offset != total:
        raise RuntimeError(
            f"Expected to write {total} embeddings for {split} split, wrote {offset}."
        )

    embeddings_mm.flush()
    labels_mm.flush()
    embeddings = np.load(emb_path, mmap_mode="r")
    labels = np.load(lbl_path, mmap_mode="r")
    return embeddings, labels


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cifar100", type=str)
    parser.add_argument("--backbone-name", default="hycoclip", type=str,
                        help="Backbone identifier")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument(
        "--target-activation-fraction",
        default=None,
        
        type=float,
        help="Desired fraction of active concept activations for threshold matching (set <=0 to disable).",
    )
    parser.add_argument(
        "--target-activation-count",
        default=None,
        type=float,
        help="Desired average number of active concepts per sample for threshold matching (used when fraction is not provided).",
    )
    parser.add_argument(
        "--train-fraction",
        default=1.0,
        type=float,
        help="Fraction of the training set to use when fitting the linear probe (0 < frac <= 1).",
    )
    parser.add_argument("--hycoclip-config", default="hycoclip/configs/train_clip_vit_b.py", type=str,
                        help="LazyConfig path for HyCoCLIP model definition.")
    parser.add_argument("--hycoclip-checkpoint", default="hycoclip/clip_vit_b.pth", type=str,
                        help="Checkpoint path for the hycoclip backbone.")
    parser.add_argument("--use-hyco-projections", action="store_true",
                        help="Tag outputs as hyperbolic CBM checkpoints.")
    parser.add_argument("--no-hyco-projections", action="store_false", dest="use_hyco_projections",
                        help="Save checkpoint with standard PCBM tag.")
    parser.set_defaults(use_hyco_projections=True)
    parser.add_argument("--viz-concept-accuracy", action="store_true",
                        help="When set, evaluates accuracy vs. active concepts across lambda values.")
    parser.add_argument("--viz-lambdas", nargs="+", type=float, default=None,
                        help="Lambda values to sweep when visualizing accuracy vs. active concepts.")
    parser.add_argument("--viz-weight-threshold", default=1e-2, type=float,
                        help="Threshold for considering a concept weight as active during visualization.")
    parser.add_argument("--viz-save-path", default=None, type=str,
                        help="Optional path to save the visualization figure (defaults under out-dir).")
    parser.add_argument(
        "--cub-root",
        default=str((REPO_ROOT / "CUB" / "CUB_200_2011").resolve()),
        type=str,
        help="Path to the CUB_200_2011 dataset root (used when dataset=cub).",
    )
    parser.add_argument("--imagenet-root",default=None,type=str,help="Root directory containing ImageNet train/val folders.")
    parser.add_argument("--embedding-cache-dir",default='outputs/visual_embeddings',type=str,help="Optional directory to store cached visual embeddings.")
    parser.add_argument("--embedding-batch-size",default=None,type=int,help="Batch size used when caching visual embeddings (defaults to batch-size).")
    parser.add_argument("--imagenet-subset-size",default=None,type=int,help="Limit ImageNet train split to this many samples for quick tests.")
    parser.add_argument("--imagenet-val-subset-size",default=None,type=int,help="Limit ImageNet validation split to this many samples (defaults to imagenet-subset-size).")
    return parser.parse_args()


def _compute_activation_threshold(
    features: np.ndarray,
    target_fraction: Optional[float],
    target_count: Optional[float],
):
    """Return threshold stats that match the requested activation sparsity."""
    if features is None or features.size == 0:
        return None

    num_concepts = features.shape[1]
    if target_fraction is None and target_count is None:
        return None

    fraction_input = target_fraction
    if target_count is not None:
        target_count = float(target_count)
        target_fraction = target_count / max(float(num_concepts), 1.0)
    elif target_fraction is not None:
        target_fraction = float(target_fraction)
        target_count = target_fraction * num_concepts

    if target_fraction is None:
        return None

    if fraction_input is not None and abs(fraction_input - target_fraction) > 1e-3:
        print(
            "[activation-threshold] Warning: target fraction {:.3f} mismatched target count {:.2f}; "
            "using {:.3f} implied by count.".format(fraction_input, target_count, target_fraction)
        )

    target_fraction = float(np.clip(target_fraction, 1e-6, 1.0 - 1e-6))
    target_count = float(target_count if target_count is not None else target_fraction * num_concepts)

    abs_vals = np.abs(features.reshape(-1))
    quantile = float(np.clip(1.0 - target_fraction, 0.0, 1.0))
    threshold = float(np.quantile(abs_vals, quantile))

    active_mask = np.abs(features) > threshold
    achieved_fraction = float(active_mask.mean())
    avg_active = float(active_mask.sum(axis=1).mean())

    return {
        "threshold": threshold,
        "target_fraction": target_fraction,
        "target_count": target_count,
        "achieved_fraction": achieved_fraction,
        "achieved_count": avg_active,
        "num_concepts": num_concepts,
    }


def _build_sgd_classifier(args, *, warm_start: bool = False) -> SGDClassifier:
    return SGDClassifier(
        random_state=args.seed,
        loss="log_loss",
        alpha=args.lam,
        l1_ratio=args.alpha,
        verbose=0,
        penalty="elasticnet",
        max_iter=5000,
        n_jobs=-1,
        warm_start=warm_start,
    )

def run_linear_probe(args, train_data, test_data, classifier=None):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    # torch.save({"acts": test_features}, "cifar100_clip_pcbm_test_acts.pt")
    print(f"Training linear probe. Feature shape: {train_features.shape}")
    
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    if classifier is None:
        classifier = _build_sgd_classifier(args, warm_start=False)
    else:
        classifier.set_params(alpha=args.lam)
        if not classifier.warm_start:
            classifier.warm_start = True

    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        # Handle edge case where a class might not be in test set
        if np.sum(test_lbl_mask) > 0:
             cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        if np.sum(train_lbl_mask) > 0:
             cls_acc["train"][lbl] = np.mean((train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        # print(f"Class {lbl} Test Acc: {cls_acc['test'].get(lbl, 'N/A')}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    run_info["classifier_type"] = "sgd"

    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))

    return run_info, classifier.coef_, classifier.intercept_


def _default_lambda_grid(base_lambda: float):
    """Return a default sweep of lambda values around the provided base."""
    base = max(float(base_lambda), 1e-8)
    factors = [0.1, 0.3, 1.0, 3.0, 10.0]
    values = sorted({max(base * factor, 1e-8) for factor in factors})
    return values


def visualize_accuracy_vs_active_concepts(args, train_data, test_data):
    """Sweep lambda values and visualize accuracy vs. active concepts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError: 
        print("matplotlib is not installed; skipping accuracy vs. active concepts visualization.")
        return

    train_features, train_labels = train_data
    test_features, test_labels = test_data

    lambda_values = args.viz_lambdas if args.viz_lambdas else _default_lambda_grid(args.lam)
    lambda_values = [max(float(lam), 1e-8) for lam in lambda_values]
    descending = all(lambda_values[idx] <= lambda_values[idx - 1] for idx in range(1, len(lambda_values)))
    if not descending:
        print("[info] Reordering lambda grid from high to low to enable warm starting.")
        lambda_values = sorted(lambda_values, reverse=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.viz_save_path:
        figure_path = Path(args.viz_save_path)
        if not figure_path.is_absolute():
            figure_path = out_dir / figure_path
    else:
        figure_path = out_dir / f"{args.backbone_name}_accuracy_vs_active_concepts_{args.dataset}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = figure_path.with_suffix(".json")
    existing_results = []
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                existing_results = loaded
            else:
                print(f"[warn] Expected list in {summary_path}, got {type(loaded).__name__}; overwriting contents.")
                existing_results = []
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] Could not read existing summary at {summary_path}: {exc}")

    def _lambda_key(value) -> str:
        try:
            return f"{float(value):.12g}"
        except (TypeError, ValueError):
            return str(value)

    existing_keys = {
        _lambda_key(item["lambda"])
        for item in existing_results
        if isinstance(item, dict) and "lambda" in item
    }

    results = []
    weight_threshold = max(float(args.viz_weight_threshold), 0.0)
    warm_classifier = _build_sgd_classifier(args, warm_start=True)

    for lam_value in lambda_values:
        probe_args = copy.deepcopy(args)
        probe_args.lam = lam_value
        run_info, weights, _ = run_linear_probe(
            probe_args,
            (train_features, train_labels),
            (test_features, test_labels),
            classifier=warm_classifier,
        )

        weights = np.atleast_2d(np.asarray(weights))
        active_mask = np.abs(weights) > weight_threshold
        concepts_per_class = active_mask.sum(axis=1)
        active_concepts = int(np.sum(concepts_per_class))
        active_concepts_per_class = [int(count) for count in concepts_per_class]

        results.append(
            {
                "lambda": lam_value,
                "train_acc": run_info.get("train_acc"),
                "test_acc": run_info.get("test_acc"),
                "active_concepts": active_concepts,
                "active_concepts_per_class": active_concepts_per_class,
            }
        )

        key = _lambda_key(lam_value)
        if key not in existing_keys:
            existing_results.append(results[-1])
            existing_keys.add(key)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(existing_results, handle, indent=2)
            print(f"[viz] Wrote lambda {lam_value:.3e} to {summary_path}")
        else:
            print(f"[viz] Lambda {lam_value:.3e} already present in {summary_path}; skipping append.")

        print(
            f"Lambda {lam_value:.3e}: test_acc={run_info.get('test_acc'):.2f}, "
            f"train_acc={run_info.get('train_acc'):.2f}, active_concepts={active_concepts:.2f}, average_active_concepts_per_class={float(active_concepts/10):.2f}"
        )

    results_sorted = sorted(results, key=lambda item: item["active_concepts"])
    x_vals = [item["active_concepts"]/1000 for item in results_sorted]
    y_vals = [item["test_acc"] for item in results_sorted]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x_vals, y_vals, marker="o", linewidth=1.5)
    for point in results_sorted:
        ax.annotate(
            f"λ={point['lambda']:.1e}",
            xy=(point["active_concepts"], point["test_acc"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Active concepts (#)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. Active Concepts across λ")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"Saved accuracy vs. active concepts figure to: {figure_path}")
    print(f"Saved sweep summary to: {summary_path}")


def compute_projections_from_embeddings(embeddings, posthoc_layer, batch_size=512):
    num_samples = embeddings.shape[0]
    num_concepts = posthoc_layer.cavs.shape[0]
    projections = np.empty((num_samples, num_concepts), dtype=np.float32)
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Computing projections"):
        end = min(i + batch_size, num_samples)
        batch_emb = torch.from_numpy(embeddings[i:end]).to(posthoc_layer.cavs.device)
        with torch.no_grad():
            batch_proj = posthoc_layer.compute_dist(batch_emb)
        projections[i:end] = batch_proj.cpu().numpy()
    return projections


def main(args, concept_bank, backbone, preprocess):
    print(f"[diag] backbone_name={args.backbone_name}")
    if hasattr(preprocess, "transforms"):
        print(f"[diag] preprocess transforms: {preprocess.transforms}")
    else:
        print(f"[diag] preprocess: {preprocess}")

    if args.dataset.lower() == "cub":
        train_loader, test_loader, idx_to_class, classes = build_cub_dataloaders(
            args.cub_root,
            preprocess,
            args.batch_size,
            args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

    # --- ImageNet Class Mapping ---
    if args.dataset.lower() == "imagenet":
        try:
            with open("imagenet_class_index.json", "r") as f:
                imagenet_class_index = json.load(f)
            synset_to_class = {v[0]: v[1] for v in imagenet_class_index.values()}
            classes = [synset_to_class.get(c, c) for c in classes]
            idx_to_class = {k: synset_to_class.get(v, v) for k, v in idx_to_class.items()}
            print("Updated ImageNet classes to human-readable labels.")
        except FileNotFoundError:
            print("imagenet_class_index.json not found. Using synset IDs.")
    # ------------------------------

    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)

    posthoc_layer = PosthocLinearCBM(
        concept_bank,
        backbone_name=args.backbone_name,
        idx_to_class=idx_to_class,
        n_classes=num_classes
    ).to(args.device)

    if args.dataset.lower() == "imagenet":
        print("Loading or computing ImageNet embeddings...")
        train_embs, train_lbls = load_or_compute_visual_embeddings(
            args,
            backbone,
            torch.device(args.device),
            train_loader.dataset,
            split="train",
        )
        test_embs, test_lbls = load_or_compute_visual_embeddings(
            args,
            backbone,
            torch.device(args.device),
            test_loader.dataset,
            split="val",
        )
        
        print("Computing projections for ImageNet...")
        train_projs = compute_projections_from_embeddings(train_embs, posthoc_layer)
        test_projs = compute_projections_from_embeddings(test_embs, posthoc_layer)
        
    else:
        train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    train_dataset = (train_projs, train_lbls)
    if args.train_fraction < 1.0:
        train_dataset = get_stratified_subset(train_dataset, args.train_fraction, args.seed)

    threshold_stats = _compute_activation_threshold(
        train_dataset[0],
        target_fraction=args.target_activation_fraction,
        target_count=args.target_activation_count,
    )
    if threshold_stats:
        print(
            "[activation-threshold] Target fraction {:.3f} (~{:.2f}/{:d}) -> T={:.6f}. "
            "Achieved fraction {:.3f}, avg active {:.2f}".format(
                threshold_stats["target_fraction"],
                threshold_stats["target_count"],
                int(threshold_stats["num_concepts"]),
                threshold_stats["threshold"],
                threshold_stats["achieved_fraction"],
                threshold_stats["achieved_count"],
            )
        )

    if args.viz_concept_accuracy:
        visualize_accuracy_vs_active_concepts(
            args,
            train_dataset,
            (test_projs, test_lbls),
        )

    run_info, weights, bias = run_linear_probe(args, train_dataset, (test_projs, test_lbls))
    run_info["train_fraction"] = float(args.train_fraction)
    run_info["train_samples_used"] = int(len(train_dataset[1]))
    posthoc_layer.set_weights(weights=weights, bias=bias)

    model_type = "clipvitb"
    frac_tag = format_fraction(args.train_fraction)
    model_path = os.path.join(
        args.out_dir,
        f"{model_type}_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}__frac:{frac_tag}.ckpt",
    )
    torch.save(posthoc_layer, model_path)

    run_info_file = model_path.replace(f"{model_type}_", f"run_info-{model_type}_").replace(".ckpt", ".pkl")

    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    if num_classes > 1:
        # Prints the Top-10 Concept Weights for each class, including sparsity statistics.
        print(posthoc_layer.analyze_classifier(k=10))

    print(f"Model saved to : {model_path}")
    print(run_info)

if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)

    # For OpenAI CLIP:
    #backbone, preprocess = clip.load('ViT-B/16', device=args.device, jit=False)
    print(preprocess)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, concept_bank, backbone, preprocess)