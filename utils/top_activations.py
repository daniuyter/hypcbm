#!/usr/bin/env python3
"""Visualize top activating images per concept across CIFAR-100, SUN397, and ImageNet.

This script reuses previously cached concept projections (e.g., train-proj_*.npy) so no
new backbone forward passes are necessary.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from torch.utils.data import DataLoader


@dataclass
class DatasetSpec:
    key: str
    dataset_type: str
    proj_path: Path
    root: Path
    split: str
    display_name: str
    embeddings_path: Optional[Path] = None


def _resolve_sun_root(root: Path) -> Optional[Path]:
    """Resolve SUN397 root using known layout under SUN_dataset/data/SUN397/.

    Return None if the expected SUN397 directory is not present to trigger the
    data_zoo fallback instead of accidentally picking unrelated folders like
    projections/.
    """
    root = Path(root).expanduser()
    candidates = [
        root / "SUN397",
        root / "data" / "SUN397",
        root / "SUN_dataset" / "data" / "SUN397",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _try_imagefolder_sun(root: Path, *, transform=None):
    resolved = _resolve_sun_root(root)            
    if resolved is None:
        return None
    subdirs = [p for p in resolved.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return datasets.ImageFolder(root=str(resolved), transform=transform)


def _build_sun397_dataset_via_data_zoo(root: Path, *, transform=None):
    """Load SUN397 strictly from the expected ImageFolder layout.

    We avoid torchvision.SUN397 (removed in recent versions) and require the
    extracted SUN397 tree to exist. This prevents accidental attempts to use
    missing metadata downloads.
    """

    ds = _try_imagefolder_sun(root, transform=transform)
    if ds is not None:
        return ds

    raise FileNotFoundError(
        "SUN397 dataset not found. Point --sun397-root to the parent of the SUN397 folder (e.g., .../SUN_dataset/data)."
    )


def build_dataset(spec: DatasetSpec, *, transform=None, download: bool = False):
    dtype = spec.dataset_type.lower()
    if dtype == "cifar100":
        train_flag = spec.split.lower() == "train"
        dataset = datasets.CIFAR100(
            root=str(spec.root),
            train=train_flag,
            download=download,
            transform=transform,
        )
    elif dtype == "sun397":
        dataset = _build_sun397_dataset_via_data_zoo(Path(spec.root), transform=transform)
    elif dtype == "imagenet":
        split_dir = (spec.root / spec.split).resolve()
        if not split_dir.exists():
            raise FileNotFoundError(f"ImageNet split directory not found: {split_dir}")
        dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {spec.dataset_type}")
    return dataset


class DatasetActivations:
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.dataset = build_dataset(spec, transform=None, download=False)
        self.activations = self._load_activations()
        self.num_samples, self.num_concepts = self.activations.shape

    def _load_activations(self) -> np.ndarray:
        path = self.spec.proj_path
        if not path.exists():
            raise FileNotFoundError(f"Projection file not found: {path}")
        activations = np.load(path, mmap_mode="r")
        dataset_length = len(self.dataset)
        if activations.shape[0] > dataset_length:
            raise ValueError(
                f"Projection rows ({activations.shape[0]}) exceed dataset length ({dataset_length}) for {self.spec.display_name}."
            )
        if activations.shape[0] < dataset_length:
            print(
                f"[warn] Projection rows ({activations.shape[0]}) are fewer than dataset length ({dataset_length}) "
                f"for {self.spec.display_name}. Only the first {activations.shape[0]} samples will be considered."
            )
        return activations

    def top_indices(self, concept_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0 or self.num_samples == 0:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=float)
        scores = self.activations[:, concept_idx]
        k = min(k, self.num_samples)
        if k == self.num_samples:
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_scores = scores[top_idx]
        return top_idx, top_scores

    def fetch_image(self, idx: int) -> np.ndarray:
        if idx >= len(self.dataset):
            raise IndexError(
                f"Requested index {idx} exceeds dataset length {len(self.dataset)} for {self.spec.display_name}."
            )
        sample = self.dataset[idx]
        if isinstance(sample, tuple):
            image = sample[0]
        else:
            image = sample
        if torch.is_tensor(image):
            img_np = image.detach().cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.array(image)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_np = np.clip(img_np, 0, 1) if img_np.dtype == float else img_np
        return img_np.astype(np.uint8) if img_np.dtype != float else img_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top activating images per concept across CIFAR-100, SUN397, and ImageNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--concept-bank", required=True, type=Path, help="Path to the pickled concept bank.")
    parser.add_argument("--device", default="cpu", help="Device for loading the concept bank (cpu or cuda).")
    parser.add_argument("--num-images", type=int, default=3, help="Number of top images to plot per dataset (capped at 3 for 3x3 layout).")
    parser.add_argument("--output-dir", type=Path, default=Path("top_activation_figures"), help="Directory to store figures.")
    parser.add_argument(
        "--concepts",
        nargs="*",
        help="Optional subset of concept names to visualize (case-insensitive). If omitted, all concepts are processed.",
    )

    parser.add_argument("--fig-dpi", type=int, default=150, help="Figure DPI when saving plots.")
    parser.add_argument(
        "--compute-projections",
        action="store_true",
        help="If set, compute projection files when missing instead of raising an error.",
    )

    parser.add_argument(
        "--overwrite-projections",
        action="store_true",
        help="Recompute and overwrite projection files even if they already exist.",
    )

    parser.add_argument("--backbone-name", default="hycoclip", help="Backbone identifier to use when computing projections.")
    parser.add_argument("--hycoclip-config", default="hycoclip/configs/train_hycoclip_vit_b.py", help="HyCoCLIP config path.")
    parser.add_argument("--hycoclip-checkpoint", default="hycoclip/hycoclip_vit_b.pth", help="HyCoCLIP checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size when computing projections.")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader workers when computing projections.")

    _add_dataset_args(
        parser,
        prefix="cifar100",
        display_name="CIFAR-100",
        dataset_type="cifar100",
        default_root=Path("data"),
        default_split="train",
    )
    _add_dataset_args(
        parser,
        prefix="sun397",
        display_name="SUN397",
        dataset_type="sun397",
        default_root=Path("data/sun397/SUN_dataset/data"),
        default_split="train",
    )
    _add_dataset_args(
        parser,
        prefix="imagenet",
        display_name="ImageNet",
        dataset_type="imagenet",
        default_root=Path("/path/to/imagenet/torchvision_ImageFolder"),
        default_split="val",
    )

    return parser.parse_args()



def _add_dataset_args(
    parser: argparse.ArgumentParser,
    *,
    prefix: str,
    display_name: str,
    dataset_type: str,
    default_root: Path,
    default_split: str,
) -> None:
    group = parser.add_argument_group(f"{display_name} options")
    group.add_argument(
        f"--{prefix}-proj",
        required=False,
        type=Path,
        help="Path to cached concept projections (.npy). Defaults to <root>/projections/<split>_%s.npy." % prefix,
    )
    group.add_argument(f"--{prefix}-root", type=Path, default=default_root, help="Dataset root directory.")
    group.add_argument(f"--{prefix}-split", type=str, default=default_split, help="Dataset split identifier (train/val).")
    group.add_argument(
        f"--{prefix}-display-name",
        type=str,
        default=display_name,
        help="Label to show on the left side of the row.",
    )
    group.add_argument(
        f"--{prefix}-embeddings",
        type=Path,
        help="Optional path to cached backbone embeddings (.npy) used to derive projections without recomputing forward passes.",
    )
    group.add_argument(
        f"--{prefix}-proj-root",
        type=Path,
        help="Directory where computed projection files should be stored when --{prefix}-proj is not provided. Defaults to the dataset root if omitted.",
    )


def _resolve_proj_path(
    explicit: Optional[Path],
    dataset_root: Path,
    split: str,
    key: str,
    proj_root: Optional[Path] = None,
) -> Path:
    if explicit is not None:
        return explicit
    
    # helper to ensure dir exists
    if proj_root is not None:
        base = proj_root / "projections"
    else:
        # Default to a safe central location instead of dataset folder
        base = Path("outputs/projections")
        
    base.mkdir(parents=True, exist_ok=True)
    return (base / f"{split}_{key}.npy").resolve()


def build_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    cifar100_proj = _resolve_proj_path(
        args.cifar100_proj,
        args.cifar100_root,
        args.cifar100_split,
        "cifar100",
        args.cifar100_proj_root,
    )
    sun397_proj = _resolve_proj_path(
        args.sun397_proj,
        args.sun397_root,
        args.sun397_split,
        "sun397",
        args.sun397_proj_root,
    )
    imagenet_proj = _resolve_proj_path(
        args.imagenet_proj,
        args.imagenet_root,
        args.imagenet_split,
        "imagenet",
        args.imagenet_proj_root,
    )

    specs = [
        DatasetSpec(
            key="cifar100",
            dataset_type="cifar100",
            proj_path=cifar100_proj,
            root=args.cifar100_root,
            split=args.cifar100_split,
            display_name=args.cifar100_display_name,
            embeddings_path=args.cifar100_embeddings,
        ),
        DatasetSpec(
            key="sun397",
            dataset_type="sun397",
            proj_path=sun397_proj,
            root=args.sun397_root,
            split=args.sun397_split,
            display_name=args.sun397_display_name,
            embeddings_path=args.sun397_embeddings,
        ),
        DatasetSpec(
            key="imagenet",
            dataset_type="imagenet",
            proj_path=imagenet_proj,
            root=args.imagenet_root,
            split=args.imagenet_split,
            display_name=args.imagenet_display_name,
            embeddings_path=args.imagenet_embeddings,
        ),
    ]
    return specs



def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "concept"



def encode_batch_images(backbone_name: str, backbone, images: torch.Tensor) -> torch.Tensor:
    name = backbone_name.lower()
    if "hycoclip" in name:
        emb = backbone.encode_image(images, project=True)
    elif "clip" in name:
        emb = backbone.encode_image(images, project=True)
    elif "meru" in name:
        emb = backbone.encode_image(images, project=True)
    else:
        emb = backbone(images)
    return emb.detach().float()


def compute_projections_from_embeddings(
    spec: DatasetSpec,
    args: argparse.Namespace,
    posthoc_layer: PosthocLinearCBM,
    device: torch.device,
) -> np.ndarray:
    if spec.embeddings_path is None:
        raise ValueError(f"No embeddings path configured for {spec.display_name}.")

    if not spec.embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {spec.embeddings_path}")

    embeddings = np.load(spec.embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (N, D) for {spec.display_name}, got {embeddings.shape}."
        )

    batch_size = max(1, args.batch_size)
    projections: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, embeddings.shape[0], batch_size),
            desc=f"Projecting embeddings for {spec.display_name}",
        ):
            stop = min(start + batch_size, embeddings.shape[0])
            chunk = embeddings[start:stop]
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            chunk_tensor = torch.from_numpy(chunk).to(device)
            chunk_proj = posthoc_layer.compute_dist(chunk_tensor).detach().cpu().numpy()
            projections.append(chunk_proj)

    return np.concatenate(projections, axis=0) if projections else np.empty((0, posthoc_layer.cavs.shape[0]))


def compute_projections_for_spec(
    spec: DatasetSpec,
    args: argparse.Namespace,
    backbone,
    preprocess,
    posthoc_layer: PosthocLinearCBM,
    device: torch.device,
) -> np.ndarray:
    dataset = build_dataset(spec, transform=preprocess, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    projections: List[np.ndarray] = []
    for batch in tqdm(dataloader, desc=f"Computing projections for {spec.display_name}"):
        images, _ = batch
        images = images.to(device, non_blocking=True)
        batch_emb = encode_batch_images(args.backbone_name, backbone, images)
        batch_proj = posthoc_layer.compute_dist(batch_emb).detach().cpu().numpy()
        projections.append(batch_proj)

    return np.concatenate(projections, axis=0) if projections else np.empty((0, posthoc_layer.cavs.shape[0]))


def ensure_projection_files(
    specs: Sequence[DatasetSpec],
    args: argparse.Namespace,
    concept_bank: ConceptBank,
    device: torch.device,
) -> None:
    def needs_compute(spec: DatasetSpec) -> bool:
        return args.overwrite_projections or not spec.proj_path.exists()

    todo = [spec for spec in specs if needs_compute(spec)]
    if not todo:
        return

    if not args.compute_projections:
        missing = "\n".join(f"- {spec.display_name}: {spec.proj_path}" for spec in todo)
        raise FileNotFoundError(
            "Projection files are missing. Provide the cached .npy files or rerun with --compute-projections.\n" + missing
        )

    posthoc_layer = PosthocLinearCBM(
        concept_bank,
        backbone_name=args.backbone_name,
        idx_to_class={0: "tmp"},
        n_classes=1,
    ).to(device)
    posthoc_layer.eval()

    specs_with_embeddings = [spec for spec in todo if spec.embeddings_path is not None]
    specs_needing_backbone = [spec for spec in todo if spec.embeddings_path is None]

    for spec in specs_with_embeddings:
        spec.proj_path.parent.mkdir(parents=True, exist_ok=True)
        computed = compute_projections_from_embeddings(spec, args, posthoc_layer, device)
        np.save(spec.proj_path, computed)
        print(f"Saved projections for {spec.display_name} to {spec.proj_path} using cached embeddings")

    if not specs_needing_backbone:
        return

    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(device).eval()

    for spec in specs_needing_backbone:
        spec.proj_path.parent.mkdir(parents=True, exist_ok=True)
        computed = compute_projections_for_spec(spec, args, backbone, preprocess, posthoc_layer, device)
        np.save(spec.proj_path, computed)
        print(f"Saved projections for {spec.display_name} to {spec.proj_path}")


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    with args.concept_bank.open("rb") as handle:
        concept_payload = pickle.load(handle)
    concept_bank = ConceptBank(concept_payload, device)
    concept_names = concept_bank.concept_names

    ensure_projection_files(specs, args, concept_bank, device)

    num_images = min(args.num_images, 3)
    if args.num_images != num_images:
        print("[info] Capping num-images to 3 to enforce a 3x3 layout.")

    if args.concepts:
        requested = {name.lower() for name in args.concepts}
        selected_indices = [idx for idx, name in enumerate(concept_names) if name.lower() in requested]
        missing = requested - {concept_names[i].lower() for i in selected_indices}
        if missing:
            print(f"[warn] The following concepts were not found and will be skipped: {sorted(missing)}")
    else:
        selected_indices = list(range(len(concept_names)))

    datasets_with_scores: List[DatasetActivations] = [DatasetActivations(spec) for spec in specs]
    _validate_concept_alignment(datasets_with_scores)

    # If specific concepts are requested, plot them combined in one figure
    if args.concepts and len(selected_indices) > 0:
        print(f"Generating combined figure for {len(selected_indices)} concepts...")
        
        # Sort indices to match the order requested by the user
        requested_order = {name.lower(): i for i, name in enumerate(args.concepts)}
        selected_indices.sort(key=lambda idx: requested_order.get(concept_names[idx].lower(), 999))
        
        selected_names = [concept_names[i] for i in selected_indices]
        
        figure_path = output_dir / "combined_top_activations.pdf"
        render_combined_figure(
            concept_indices=selected_indices,
            concept_names=selected_names,
            datasets=datasets_with_scores,
            num_images=num_images,
            dpi=args.fig_dpi,
            output_path=figure_path,
        )
        print(f"Saved combined figure to {figure_path}")
    else:
        # Otherwise, generate individual figures for all concepts (legacy behavior)
        print(f"Processing {len(selected_indices)} concepts individually...")
        for concept_idx in tqdm(selected_indices, desc="Concepts"):
            concept_name = concept_names[concept_idx]
            figure_path = output_dir / f"{sanitize_filename(concept_name)}.pdf"
            render_concept_figure(
                concept_idx=concept_idx,
                concept_name=concept_name,
                datasets=datasets_with_scores,
                num_images=num_images,
                dpi=args.fig_dpi,
                output_path=figure_path,
            )

    print(f"Saved figures to {output_dir.resolve()}")


def _validate_concept_alignment(datasets_with_scores: Sequence[DatasetActivations]) -> None:
    concept_dims = {ds.spec.display_name: ds.num_concepts for ds in datasets_with_scores}
    uniques = set(concept_dims.values())
    if len(uniques) != 1:
        raise ValueError(f"Concept dimension mismatch across datasets: {concept_dims}")


def _crop_center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return img[start_y : start_y + min_dim, start_x : start_x + min_dim]


def render_combined_figure(
    *,
    concept_indices: List[int],
    concept_names: List[str],
    datasets: Sequence[DatasetActivations],
    num_images: int,
    dpi: int,
    output_path: Path,
) -> None:
    # Configure matplotlib for publication-quality plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
    })

    num_concepts = len(concept_indices)
    num_datasets = len(datasets)
    
    # Layout: 
    # Rows = Datasets
    # Columns = Concepts * Images per Concept
    
    # Dimensions (inches)
    img_size = 1.2
    padding_between_concepts = 0.5
    label_width = 1.5
    
    total_width = label_width + (num_concepts * num_images * img_size) + ((num_concepts - 1) * padding_between_concepts)
    total_height = num_datasets * (img_size + 0.4) + 0.5 # +0.5 for concept titles
    
    fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
    
    # Create GridSpec
    # We need columns for images and spacers between concept blocks
    # Total columns = (num_images * num_concepts) + (num_concepts - 1 spacers)
    # But simpler to just use subplots_adjust or manual placement? 
    # Let's use subplots with constrained layout and empty columns for spacing?
    # Actually, creating separate sub-gridspecs for each concept is cleaner.
    
    gs_outer = fig.add_gridspec(1, num_concepts, wspace=0.15)
    
    for c_idx, (concept_idx, concept_name) in enumerate(zip(concept_indices, concept_names)):
        gs_inner = gs_outer[c_idx].subgridspec(num_datasets, num_images, wspace=0.05, hspace=0.05)
        
        clean_name = concept_name.replace("_", " ").title()
        # Add concept title above the block
        ax_title = fig.add_subplot(gs_inner[0, :])
        ax_title.set_title(clean_name, fontsize=16, fontweight="bold", pad=10)
        ax_title.axis("off")
        
        for row, ds in enumerate(datasets):
            top_indices, top_scores = ds.top_indices(concept_idx, num_images)
            
            # Add dataset label only on the far left of the entire figure
            if c_idx == 0:
                ax_label = fig.add_subplot(gs_inner[row, 0])
                ax_label.text(
                    -0.25, 0.5, 
                    ds.spec.display_name, 
                    transform=ax_label.transAxes, 
                    ha="right", 
                    va="center", 
                    fontsize=14, 
                    fontweight="bold",
                    rotation=0
                )
                ax_label.axis("off") # We just used it for text positioning relative to the row

            for col in range(num_images):
                ax = fig.add_subplot(gs_inner[row, col])
                ax.axis("off")
                
                if col >= len(top_indices):
                    continue
                    
                idx = int(top_indices[col])
                # score = float(top_scores[col]) # Optional: show score
                image = ds.fetch_image(idx)
                image = _crop_center_square(image)
                
                ax.imshow(image)
                # ax.set_title(f"{score:.2f}", fontsize=8) # Optional score

    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def render_concept_figure(
    *,
    concept_idx: int,
    concept_name: str,
    datasets: Sequence[DatasetActivations],
    num_images: int,
    dpi: int,
    output_path: Path,
) -> None:
    # Legacy single-concept renderer (kept for backward compatibility if needed)
    # Re-implement using the new style or just redirect to combined with 1 concept
    render_combined_figure(
        concept_indices=[concept_idx],
        concept_names=[concept_name],
        datasets=datasets,
        num_images=num_images,
        dpi=dpi,
        output_path=output_path
    )



if __name__ == "__main__":
    main()
