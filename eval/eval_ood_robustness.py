#!/usr/bin/env python3
"""Evaluate CBM OOD robustness on corruption benchmarks."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from models.entailment_cbm import entailment_cone_feature_matrix

try:
    from imagecorruptions import corrupt as apply_image_corruption
except ImportError as import_error:
    apply_image_corruption = None
    IMAGECORRUPTIONS_IMPORT_ERROR = import_error


CIFAR100_CORRUPTIONS: tuple[str, ...] = (
    # Noise
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    # Blur
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    # Weather
    "snow",
    "frost",
    "fog",
    "brightness",
    # Digital
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression"
)


class CIFAR100CorruptedDataset(Dataset):
    """Wrap the CIFAR-100 test set with on-the-fly corruptions."""
    def __init__(
        self,
        base_dataset: datasets.CIFAR100,
        preprocess,
        corruption: str | None = None,
        severity: int = 3,
    ) -> None:
        self.base = base_dataset
        self.preprocess = preprocess
        self.corruption = None if corruption in {None, "clean"} else corruption
        self.severity = severity
        if self.corruption and apply_image_corruption is None:
            raise ImportError(
                "The 'imagecorruptions' package is required for CIFAR100-C evaluation."
                " Install it via 'pip install imagecorruptions'."
            ) from IMAGECORRUPTIONS_IMPORT_ERROR

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        if self.corruption:
            np_img = np.array(image)
            corrupted = apply_image_corruption(
                np_img,
                corruption_name=self.corruption,
                severity=self.severity,
            )
            image = Image.fromarray(corrupted)
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image, label
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CBM robustness on CIFAR100-C style shifts.")
    parser.add_argument("--artifact-path", default=None, type=str,
                        help="Path to a saved CBM artifact (.pkl). Provide this or --ckpt-path.")
    parser.add_argument("--ckpt-path", default=None, type=str,
                        help="Path to a PosthocLinearCBM .ckpt checkpoint (e.g., from train_pcbm_hycoclip.py).")
    parser.add_argument("--ood-dataset", default="cifar100-c", choices=["cifar100-c"], help="OOD benchmark to use.")
    parser.add_argument("--corruption-type", default="all", type=str,
                        help="Specific corruption to evaluate (or 'all' for the full suite).")
    parser.add_argument("--severity", default=3, type=int, help="Corruption severity level (1-5).")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for embedding extraction.")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run inference on.")
    parser.add_argument("--feature-type", default="auto", choices=["auto", "entailment", "pcbm"],
                        help="Feature extractor to use; 'auto' infers from the artifact.")
    parser.add_argument("--chunk-size", default=None, type=int,
                        help="Chunk size for feature computation (overrides artifact value if provided).")
    parser.add_argument("--entail-eta", default=None, type=float,
                        help="Override entailment eta (defaults to artifact).")
    parser.add_argument("--concept-bank", default=None, type=str,
                        help="Optional override for the concept bank pickle path.")
    parser.add_argument("--backbone-name", default=None, type=str,
                        help="Override backbone name (defaults to artifact value).")
    parser.add_argument("--hycoclip-config", default="hycoclip/configs/train_hycoclip_vit_b.py", type=str,
                        help="Config path for HyCoCLIP/CLIP/MERU backbones.")
    parser.add_argument("--hycoclip-checkpoint", default="hycoclip/hycoclip_vit_b.pth", type=str,
                        help="Checkpoint path for HyCoCLIP/CLIP/MERU backbones.")
    parser.add_argument("--data-root", default="./data", type=str,
                        help="Root directory for torchvision datasets.")
    parser.add_argument("--out-csv", default="ood_results.csv", type=str,
                        help="Destination CSV file for robustness metrics.")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = torch.device("cpu")
    return device


def encode_batch_images(backbone_name: str, backbone, images: torch.Tensor) -> torch.Tensor:
    name = backbone_name.lower()
    if "hycoclip" in name:
        embeddings = backbone.encode_image(images, project=True)
    elif "clip" in name:
        embeddings = backbone.encode_image(images, project=True)
    elif "meru" in name:
        embeddings = backbone.encode_image(images, project=True)
    else:
        embeddings = backbone(images)
    return embeddings.detach().float()


def extract_embeddings(
    backbone_name: str,
    backbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            batch_emb = encode_batch_images(backbone_name, backbone, images)
            all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(labels.numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def compute_tangent_norms(
    embeddings: torch.Tensor,
    model,
    curvature: torch.Tensor | None = None,
) -> torch.Tensor:
    curv = curvature if curvature is not None else model.curv.exp()
    from hycoclip.hycoclip import lorentz as L
    tangent = L.log_map0(embeddings, curv)
    return torch.linalg.norm(tangent, dim=-1)


def load_concept_tensors(path: str, device: torch.device, model) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    with open(path, "rb") as handle:
        concept_obj = pickle.load(handle)
    concept_names = sorted(concept_obj.keys())
    concept_array = np.stack(
        [np.asarray(concept_obj[name][0], dtype=np.float32).squeeze() for name in concept_names],
        axis=0,
    )
    concept_tensor = torch.from_numpy(concept_array).to(device=device)
    curvature = model.curv.exp()
    text_norms = compute_tangent_norms(concept_tensor, model, curvature)
    return concept_names, concept_tensor, text_norms


def compute_entailment_features(
    backbone,
    embeddings: np.ndarray,
    concept_tensor: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    eta: float,
) -> np.ndarray:
    return entailment_cone_feature_matrix(
        backbone,
        embeddings,
        concept_tensor,
        chunk_size,
        device,
        eta=eta,
    )


def compute_pcbm_features(
    embeddings: np.ndarray,
    posthoc_layer: PosthocLinearCBM,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        batch = torch.from_numpy(embeddings[start:stop]).to(device)
        with torch.no_grad():
            margins = posthoc_layer.compute_dist(batch)
        outputs.append(margins.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_accuracy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    logits = features @ weights.T + bias.reshape(1, -1)
    preds = np.argmax(logits, axis=1)
    return float((preds == labels).mean() * 100.0)


def compute_features_and_accuracy(
    loader: DataLoader,
    label: str,
    backbone_name: str,
    backbone,
    device: torch.device,
    feature_builder,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract features, labels, and accuracy for a loader."""
    embeddings, labels_arr = extract_embeddings(backbone_name, backbone, loader, device)
    features = feature_builder(embeddings)
    acc = compute_accuracy(features, labels_arr, weights, bias)
    print(f"{label:<20} -> Acc: {acc:.2f}%")
    return features, labels_arr, acc


def compute_drift_and_jaccard(clean_feats: np.ndarray, corrupted_feats: np.ndarray) -> tuple[float, float]:
    """Compute relative L2 drift and Jaccard similarity between clean and corrupted activations."""
    if clean_feats.shape != corrupted_feats.shape:
        raise ValueError("Clean and corrupted feature shapes must match for drift metrics")

    diff = corrupted_feats - clean_feats
    clean_norm = np.linalg.norm(clean_feats, axis=1) + 1e-12
    rel_l2 = np.linalg.norm(diff, axis=1) / clean_norm
    mean_rel_l2 = float(rel_l2.mean())

    # threshold pcbm = 0.222493 for active concepts
    clean_active = clean_feats > 0
    corr_active = corrupted_feats > 0
    intersection = (clean_active & corr_active).sum(axis=1).astype(np.float32)
    union = (clean_active | corr_active).sum(axis=1).astype(np.float32) + 1e-6
    jaccard = intersection / union
    mean_jaccard = float(jaccard.mean())
    return mean_rel_l2, mean_jaccard


def build_cifar100_loader(
    base_dataset: datasets.CIFAR100,
    preprocess,
    batch_size: int,
    num_workers: int,
    corruption: str | None,
    severity: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CIFAR100CorruptedDataset(
        base_dataset=base_dataset,
        preprocess=preprocess,
        corruption=corruption,
        severity=severity,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def resolve_feature_type(requested: str, artifact: dict | None, has_ckpt: bool) -> str:
    if requested != "auto":
        return requested
    if artifact and "entail_eta" in artifact:
        return "entailment"
    if has_ckpt:
        return "pcbm"
    if artifact:
        return artifact.get("feature_type", "pcbm")
    return "pcbm"


def validate_corruption_choice(corruption: str) -> Iterable[str]:
    if corruption == "all":
        return CIFAR100_CORRUPTIONS
    if corruption not in CIFAR100_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption}'. Choose from {', '.join(CIFAR100_CORRUPTIONS)} or 'all'."
        )
    return (corruption,)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    artifact_path = Path(args.artifact_path).expanduser() if args.artifact_path else None
    artifact = None
    if artifact_path:
        with open(artifact_path, "rb") as handle:
            artifact = pickle.load(handle)

    ckpt_path = Path(args.ckpt_path).expanduser() if args.ckpt_path else None
    posthoc_from_ckpt = None
    if ckpt_path:
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not isinstance(loaded, PosthocLinearCBM):
            raise TypeError("--ckpt-path must point to a PosthocLinearCBM checkpoint produced by train_pcbm_hycoclip.py")
        posthoc_from_ckpt = loaded.to(device)
        posthoc_from_ckpt.eval()

    if artifact is None and posthoc_from_ckpt is None:
        raise ValueError("Provide either --artifact-path or --ckpt-path to evaluate robustness.")
    if artifact is not None and posthoc_from_ckpt is not None:
        raise ValueError("Please supply only one of --artifact-path or --ckpt-path (not both).")

    feature_type = resolve_feature_type(args.feature_type, artifact, posthoc_from_ckpt is not None)

    if artifact is not None:
        weights = np.asarray(artifact["weights"], dtype=np.float32)
        bias = np.asarray(artifact["bias"], dtype=np.float32).reshape(weights.shape[0])
    else:
        classifier = posthoc_from_ckpt.classifier
        weights = classifier.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        bias = classifier.bias.detach().cpu().numpy().astype(np.float32, copy=False)
        bias = bias.reshape(weights.shape[0])

    backbone_name = (
        args.backbone_name
        or (artifact.get("backbone") if artifact else None)
        or (artifact.get("backbone_name") if artifact else None)
        or (getattr(posthoc_from_ckpt, "backbone_name", None) if posthoc_from_ckpt else None)
    )
    if backbone_name is None:
        raise ValueError("Backbone name must be specified via --backbone-name, the artifact, or the checkpoint.")

    default_chunk = int(artifact.get("chunk_size", 128)) if artifact else 128
    chunk_size = args.chunk_size or default_chunk
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    eta = None
    if feature_type == "entailment":
        if args.entail_eta is not None:
            eta = float(args.entail_eta)
        elif artifact and "entail_eta" in artifact:
            eta = float(artifact["entail_eta"])
        else:
            raise ValueError("Entailment eta missing. Provide --entail-eta or store it inside the artifact.")

    concept_bank_path = args.concept_bank or (artifact.get("concept_bank") if artifact else None)

    concept_bank = None
    if feature_type == "pcbm" and posthoc_from_ckpt is None:
        if concept_bank_path is None:
            raise ValueError("Concept bank path required to reconstruct PCBM layer when no checkpoint is provided.")
        with open(concept_bank_path, "rb") as handle:
            concept_data = pickle.load(handle)
        concept_bank = ConceptBank(concept_data, str(device))

    backbone_args = SimpleNamespace(
        device=str(device),
        out_dir=args.data_root,
        backbone_name=backbone_name,
        hycoclip_config=args.hycoclip_config,
        hycoclip_checkpoint=args.hycoclip_checkpoint,
    )
    backbone, preprocess = get_model(backbone_args, backbone_name=backbone_name)
    backbone = backbone.to(device)
    backbone.eval()

    concept_tensor = None
    posthoc_layer = None
    if feature_type == "entailment":
        if concept_bank_path is None:
            raise ValueError("Concept bank path is required for entailment feature evaluation.")
        _, concept_tensor, _ = load_concept_tensors(concept_bank_path, device, backbone)
    else:
        if posthoc_from_ckpt is not None:
            posthoc_layer = posthoc_from_ckpt
        else:
            n_classes = weights.shape[0]
            idx_to_class = {i: i for i in range(n_classes)}
            posthoc_layer = PosthocLinearCBM(
                concept_bank,
                backbone_name=backbone_name,
                idx_to_class=idx_to_class,
                n_classes=n_classes,
            ).to(device)
        posthoc_layer.eval()

    base_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=None,
    )
    pin_memory = device.type == "cuda"

    def feature_builder(embeddings: np.ndarray) -> np.ndarray:
        if feature_type == "entailment":
            if concept_tensor is None:
                raise RuntimeError("Concept tensor not initialized for entailment features.")
            return compute_entailment_features(backbone, embeddings, concept_tensor, chunk_size, device, eta)
        return compute_pcbm_features(embeddings, posthoc_layer, chunk_size, device)

    def evaluate_loader(loader: DataLoader, label: str) -> tuple[np.ndarray, np.ndarray, float]:
        return compute_features_and_accuracy(
            loader,
            label,
            backbone_name,
            backbone,
            device,
            feature_builder,
            weights,
            bias,
        )

    clean_loader = build_cifar100_loader(
        base_dataset,
        preprocess,
        args.batch_size,
        args.num_workers,
        corruption=None,
        severity=0,
        pin_memory=pin_memory,
    )
    clean_features, clean_labels, clean_accuracy = evaluate_loader(clean_loader, "clean")

    corruption_names = list(validate_corruption_choice(args.corruption_type))
    results = [
        {
            "corruption": "clean",
            "severity": 0,
            "accuracy": clean_accuracy,
            "delta_vs_clean": 0.0,
        }
    ]

    for corruption in corruption_names:
        loader = build_cifar100_loader(
            base_dataset,
            preprocess,
            args.batch_size,
            args.num_workers,
            corruption=corruption,
            severity=args.severity,
            pin_memory=pin_memory,
        )
        corr_features, corr_labels, acc = evaluate_loader(loader, f"{corruption}/s{args.severity}")
        if corr_labels.shape != clean_labels.shape or not np.array_equal(corr_labels, clean_labels):
            print(f"[warn] Label mismatch between clean and {corruption}; drift metrics skipped for this corruption.")
            mean_rel_l2, mean_jaccard = float('nan'), float('nan')
        else:
            mean_rel_l2, mean_jaccard = compute_drift_and_jaccard(clean_features, corr_features)
        results.append(
            {
                "corruption": corruption,
                "severity": args.severity,
                "accuracy": acc,
                "delta_vs_clean": clean_accuracy - acc,
                "l2_rel_drift": mean_rel_l2,
                "jaccard": mean_jaccard,
            }
        )

    reference_dir = artifact_path.parent if artifact_path else ckpt_path.parent
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = reference_dir / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corruption", "severity", "accuracy", "delta_vs_clean", "l2_rel_drift", "jaccard"])
        writer.writeheader()
        writer.writerows(results)

    print("\nCorruption Robustness Summary:")
    print(f"{'Corruption':<20} {'Severity':<8} {'Acc (%)':<10} {'Δ vs clean':<12}")
    for row in results:
        print(
            f"{row['corruption']:<20} {row['severity']:<8} {row['accuracy']:<10.2f} {row['delta_vs_clean']:<12.2f}"
        )
    avg_drop = np.mean([row["delta_vs_clean"] for row in results if row["corruption"] != "clean"])
    print(f"\nAverage drop across corruptions: {avg_drop:.2f} percentage points")
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()