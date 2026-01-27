import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as tv_transforms
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import get_dataset
from data.cub_loader import build_cub_dataloaders
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections
from models.entailment_cbm import entailment_cone_feature_matrix
from torchvision.transforms.functional import to_pil_image

from utils.common import (
    set_seed,
    sanitize_identifier,
    format_fraction,
    get_stratified_subset,
)
from utils.concept_utils import (
    load_concept_tensors,
    encode_batch_images,
)


REPO_ROOT = Path(__file__).resolve().parent
HYCOCLIP_ROOT = REPO_ROOT / "hycoclip"
if str(HYCOCLIP_ROOT) not in sys.path:
    sys.path.append(str(HYCOCLIP_ROOT))

from hycoclip.hycoclip import lorentz as L
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CBM using geodesic text-region activations.")
    parser.add_argument("--concept-bank", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--dataset", default="cifar100", type=str)
    parser.add_argument("--backbone-name", default="hycoclip", type=str)
    parser.add_argument("--hycoclip-config", default="hycoclip/configs/train_hycoclip_vit_b.py", type=str)
    parser.add_argument("--hycoclip-checkpoint", default="hycoclip/hycoclip_vit_b.pth", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--alpha", default=0.99, type=float)
    parser.add_argument("--lam", default=1e-5, type=float)
    parser.add_argument("--chunk-size",default=256,type=int,help="Number of images processed per batch.")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--imagenet-root",default=None,type=str,help="Root directory containing ImageNet train/val folders.")
    parser.add_argument("--embedding-cache-dir",default='outputs/visual_embeddings',type=str,help="Optional directory to store cached visual embeddings.")
    parser.add_argument("--embedding-batch-size",default=None,type=int,help="Batch size used when caching visual embeddings (defaults to batch-size).")
    parser.add_argument("--imagenet-subset-size",default=None,type=int,help="Limit ImageNet train split to this many samples for quick tests.")
    parser.add_argument("--imagenet-val-subset-size",default=None,type=int,help="Limit ImageNet validation split to this many samples (defaults to imagenet-subset-size).")
    parser.add_argument( "--cub-root",default=str((REPO_ROOT / "CUB" / "CUB_200_2011").resolve()),type=str,help="Path to the CUB_200_2011 dataset root (used when dataset=cub).")
    parser.add_argument("--viz-concept-accuracy",action="store_true",help="Evaluate accuracy vs active concepts across lambda values.")
    parser.add_argument("--viz-lambdas",nargs="+",type=float,default=None,help="Lambda values to sweep when visualizing accuracy vs active concepts.")
    parser.add_argument("--viz-weight-threshold",default=1e-2,type=float,help="Threshold for considering a concept weight active during visualization.")
    parser.add_argument("--viz-save-path",default=None,type=str,help="Optional path to save the visualization figure (defaults under out-dir).")
    parser.add_argument("--entail-eta",default=1.4,type=float,help="Scaling factor for entailment-cone activations.")
    parser.add_argument("--train-fraction",default=1.0,type=float,help="Fraction of training samples to use when fitting the linear probe (0 < frac <= 1).",)
    return parser.parse_args()


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
) -> tuple[np.ndarray, np.ndarray]:
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



def _build_sgd_classifier(args: argparse.Namespace, *, warm_start: bool = False) -> SGDClassifier:
    return SGDClassifier(
        random_state=args.seed,
        loss="log_loss",
        alpha=args.lam,
        l1_ratio=args.alpha,
        verbose=0,
        penalty="elasticnet",
        n_jobs=-1,
        warm_start=warm_start,
    )


def run_linear_probe(
    args,
    train_data,
    test_data,
    classifier=None,
):
    train_features, train_labels = train_data
    print(train_features)
    zero_feature_average = np.mean(train_features == 0)
    nonzero_counts = np.count_nonzero(train_features, axis=1)
    
    avg_nonzero_per_sample = float(nonzero_counts.mean()) if nonzero_counts.size else 0.0
    print("Average fraction of zero features (train):", zero_feature_average)
    print("Average non-zero activations per training sample:", avg_nonzero_per_sample)
    
    test_features, test_labels = test_data
    if classifier is None:
        classifier = _build_sgd_classifier(args, warm_start=False)
    else:
        classifier.set_params(alpha=args.lam)
        if not classifier.warm_start:
            classifier.warm_start = True

    classifier.fit(train_features, train_labels)

        
    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.0
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0

    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        lbl = int(lbl)
        train_mask = train_labels == lbl
        test_mask = test_labels == lbl
        if train_mask.any():
            cls_acc["train"][lbl] = float((train_predictions[train_mask] == train_labels[train_mask]).mean())
        if test_mask.any():
            cls_acc["test"][lbl] = float((predictions[test_mask] == test_labels[test_mask]).mean())

    run_info = {
        "train_acc": train_accuracy,
        "test_acc": test_accuracy,
        "cls_acc": cls_acc,
        "classifier_type": "sgd",
    }

    if len(np.unique(train_labels)) == 2:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))

    return run_info, classifier.coef_, classifier.intercept_

def visualize_accuracy_vs_active_concepts(args, train_data, test_data):
    """Sweep λ values using precomputed entailment features."""
    train_features, train_labels = train_data
    test_features, test_labels = test_data

    lambda_values = args.viz_lambdas
    lambda_values = [max(float(lam), 1e-8) for lam in lambda_values]

    descending = all(lambda_values[idx] <= lambda_values[idx - 1] for idx in range(1, len(lambda_values)))
    if not descending:
        print("[info] Reordering viz-lambdas from high to low to enable warm starting.")
        lambda_values = sorted(lambda_values, reverse=True)

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

        print(
            f"Lambda {lam_value:.3e}: test_acc={run_info.get('test_acc'):.2f}, "
            f"train_acc={run_info.get('train_acc'):.2f}, active_concepts={active_concepts:.2f}, "
            f"average_active_concepts_per_class={float(active_concepts/len(active_concepts_per_class)):.2f}"
        )

    results_sorted = sorted(results, key=lambda item: item["active_concepts"])
    x_vals = [item["active_concepts"]/100 for item in results_sorted]
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.viz_save_path:
        figure_path = Path(args.viz_save_path)
        if not figure_path.is_absolute():
            figure_path = out_dir / figure_path
    else:
        figure_path = out_dir / (str(args.entail_eta) + f"entailment_accuracy_vs_active_concepts_{args.dataset}.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

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
    appended = 0
    for item in results_sorted:
        key = _lambda_key(item.get("lambda"))
        if key in existing_keys:
            continue
        existing_results.append(item)
        existing_keys.add(key)
        appended += 1

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(existing_results, handle, indent=2)

    if appended:
        print(f"Appended {appended} new lambda entries to {summary_path}.")
    else:
        print(f"No new lambda entries appended; all provided lambda values already existed in {summary_path}.")

    print(f"Saved accuracy vs. active concepts figure to: {figure_path}")
    print(f"Saved sweep summary to: {summary_path}")


def print_top_concepts(posthoc_layer, k: int = 10):
    try:
        analysis = posthoc_layer.analyze_classifier(k=k)
    except Exception as exc:
        print(f"Could not analyze classifier weights: {exc}")
        return
    print("\nTop contributing concepts per class:\n")
    print(analysis)


def prepare_features(args, model, concept_tensor, text_norms, train_embs, test_embs):
    text_range = (float(text_norms.min().item()), float(text_norms.max().item()))
    print(
        f"Text tangent norm range: [{text_range[0]:.4f}, {text_range[1]:.4f}]"
        f" using {len(text_norms)} concepts"
    )
    device = torch.device(args.device)
    
    train_feats = entailment_cone_feature_matrix(
        model,
        train_embs,
        concept_tensor,
        512,
        device,
        eta=args.entail_eta,
    )
    test_feats = entailment_cone_feature_matrix(
        model,
        test_embs,
        concept_tensor,
        512,
        device,
        eta=args.entail_eta,
    )

    return train_feats, test_feats, text_range


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Initializing training...")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading concept bank...")
    all_concepts = pickle.load(open(args.concept_bank, "rb"))
    concept_bank = ConceptBank(all_concepts, args.device)

    print("Loading model backbone...")
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(device)
    backbone.eval()

    print("Loading dataset...")
    if args.dataset.lower() == "cub":
        train_loader, test_loader, idx_to_class, classes = build_cub_dataloaders(
            args.cub_root,
            preprocess,
            args.batch_size,
            args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

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

    print("Initializing posthoc layer...")
    posthoc_layer = PosthocLinearCBM(
        concept_bank,
        backbone_name=args.backbone_name,
        idx_to_class=idx_to_class,
        n_classes=len(classes),
    ).to(device)

    dataset_name = args.dataset.lower()
    if dataset_name in {"imagenet", "places365"}:
        split_train = "train" if dataset_name == "imagenet" else args.places365_train_split
        split_val = "val" if dataset_name == "imagenet" else args.places365_val_split
        print(f"Loading or computing {args.dataset} embeddings...")
        train_embs, train_lbls = load_or_compute_visual_embeddings(
            args,
            backbone,
            device,
            train_loader.dataset,
            split=split_train,
        )
        test_embs, test_lbls = load_or_compute_visual_embeddings(
            args,
            backbone,
            device,
            test_loader.dataset,
            split=split_val,
        )
    else:
        print("Loading or computing projections...")
        (
            train_embs,
            _train_projs,
            train_lbls,
            test_embs,
            _test_projs,
            test_lbls,
        ) = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    print("Loading concept tensors...")
    concept_names, concept_tensor, text_norms = load_concept_tensors(
        args.concept_bank,
        device,
        backbone,
    )

    print("Preparing features...")
    train_features, test_features, text_range = prepare_features(
        args,
        backbone,
        concept_tensor,
        text_norms,
        train_embs,
        test_embs,
    )
    
    train_dataset = (train_features, train_lbls)
    if args.train_fraction < 1.0:
        print(f"Using a fraction ({args.train_fraction}) of the training data...")
        train_dataset = get_stratified_subset(train_dataset, args.train_fraction, args.seed)

    if getattr(args, "viz_concept_accuracy", False):
        print("Visualizing accuracy vs. active concepts...")
        visualize_accuracy_vs_active_concepts(
            args,
            train_dataset,
            (test_features, test_lbls),
        )

    print("Running linear probe...")
    run_info, weights, bias = run_linear_probe(
        args,
        train_dataset,
        (test_features, test_lbls),
    )
    run_info["train_fraction"] = float(args.train_fraction)
    run_info["train_samples_used"] = int(len(train_dataset[1]))

    posthoc_layer.set_weights(weights, bias)
    print_top_concepts(posthoc_layer)

    artifact = {
        "weights": weights,
        "bias": bias,
        "concept_names": concept_names,
        "text_range": text_range,
        "chunk_size": args.chunk_size,
        "entail_eta": float(getattr(args, "entail_eta", 0.9)),
        "train_fraction": float(args.train_fraction),
        "train_samples_used": int(len(train_dataset[1])),
        "run_info": run_info,
        "dataset": args.dataset,
        "backbone": args.backbone_name,
        "concept_bank": args.concept_bank,
    }
    frac_tag = format_fraction(args.train_fraction)
    artifact_path = out_dir / f"entailment_cbm_{args.dataset}_{args.backbone_name}_{frac_tag}frac.pkl"

    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)

    print(f"Saved entailment CBM artifacts to {artifact_path}")
    print(run_info)



if __name__ == "__main__":
    main()
