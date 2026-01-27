#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# Add parent dir to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Patch for HyCoCLIP imports ---
# Validates imports for 'model_zoo.py' which uses 'hycoclip.hycoclip...'
# while keeping internal 'hycoclip...' imports working.
try:
    hycoclip_root = Path(__file__).resolve().parent.parent / "hycoclip"
    if str(hycoclip_root) not in sys.path:
        sys.path.append(str(hycoclip_root))
    
    import hycoclip
    import hycoclip.config
    import hycoclip.utils
    import hycoclip.utils.checkpointing

    hycoclip.hycoclip = hycoclip
    sys.modules['hycoclip.hycoclip'] = hycoclip
    sys.modules['hycoclip.hycoclip.config'] = hycoclip.config
    sys.modules['hycoclip.hycoclip.utils'] = hycoclip.utils
    sys.modules['hycoclip.hycoclip.utils.checkpointing'] = hycoclip.utils.checkpointing
except ImportError as e:
    print(f"Warning: Failed to patch hycoclip imports: {e}")
# ----------------------------------

try:
    from concepts import ConceptBank
    from top_activations import _try_imagefolder_sun, _resolve_sun_root
except ImportError as e:
    print(f"Error importing modules: {e}")
    def _resolve_sun_root(root: Path):
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

    def _try_imagefolder_sun(root: Path, transform=None):
        resolved = _resolve_sun_root(root)
        if resolved is None:
            return None
        subdirs = [p for p in resolved.iterdir() if p.is_dir()]
        if not subdirs:
            return None
        return datasets.ImageFolder(root=str(resolved), transform=transform)

def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if backbone_name.lower().startswith("hycoclip"):
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_hycoclip_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/hycoclip_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    elif "clip" in backbone_name:
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_clip_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/clip_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    
    elif "meru" in backbone_name:
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_meru_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/meru_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess



def main():
    parser = argparse.ArgumentParser(description="Plot norm distributions for Concept Bank and SUN397")
    parser.add_argument("--concept-bank", required=True, type=Path, help="Path to pkl concept bank")
    parser.add_argument("--sun397-root", type=Path, default=Path("data/sun397/SUN_dataset/data"), help="Root of SUN397")
    parser.add_argument("--backbone-name", default="hycoclip", help="Backbone name (hycoclip, clip, meru)")
    parser.add_argument("--hycoclip-config", default="hycoclip/configs/train_hycoclip_vit_b.py", help="Config path")
    parser.add_argument("--hycoclip-checkpoint", default="hycoclip/hycoclip_vit_b.pth", help="Checkpoint path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-file", type=Path, default=Path("outputs/norm_distribution_sun397.pdf"))
    
    # Dummy arg for get_model if it accesses out_dir
    parser.add_argument("--out-dir", type=Path, default=Path("out")) 

    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Concept Bank
    print(f"Loading Concept Bank from {args.concept_bank}...")
    if not args.concept_bank.exists():
        print(f"Error: Concept bank not found at {args.concept_bank}")
        sys.exit(1)

    with open(args.concept_bank, "rb") as f:
        concept_payload = pickle.load(f)
    concept_bank = ConceptBank(concept_payload, device)
    
    # Extract norms
    # concept_bank.norms is (N_concepts, 1) tensor on device
    cb_norms = concept_bank.norms.detach().cpu().numpy().flatten()
    print(f"Concept Bank norms: N={len(cb_norms)}, Mean={np.mean(cb_norms):.4f}, Std={np.std(cb_norms):.4f}")

    # 2. Load SUN397 and compute embeddings
    print(f"Loading SUN397 from {args.sun397_root}...")
    dataset = _try_imagefolder_sun(args.sun397_root)
    
    if dataset is None:
        print(f"Warning: _try_imagefolder_sun returned None. Trying to resolve manually.")
        resolved = _resolve_sun_root(args.sun397_root)
        if resolved:
             dataset = datasets.ImageFolder(str(resolved))
        else:
             print(f"Error: Could not find SUN397 at {args.sun397_root}")
             sys.exit(1)

    print(f"Dataset size: {len(dataset)}")
    
    # Load Model
    print(f"Loading backbone {args.backbone_name}...")
    try:
        backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure hycoclip config paths are correct relative to project root.")
        sys.exit(1)

    backbone = backbone.to(device).eval()
    
    dataset.transform = preprocess
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)

    print("Computing SUN397 norms (this may take a while)...")
    img_norms = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
                
            images = images.to(device)
            
            if hasattr(backbone, 'encode_image'):
                try:
                    emb = backbone.encode_image(images, project=True)
                except TypeError:
                    emb = backbone.encode_image(images)
            else:
                emb = backbone(images)

            if isinstance(emb, tuple):
                emb = emb[0]
                
            # Compute norms
            batch_norms = torch.norm(emb, p=2, dim=1).cpu().numpy()
            img_norms.append(batch_norms)

    img_norms = np.concatenate(img_norms)
    print(f"SUN397 Image norms: N={len(img_norms)}, Mean={np.mean(img_norms):.4f}, Std={np.std(img_norms):.4f}")

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.figsize": (10, 6),
    })
    min_val = min(cb_norms.min(), img_norms.min())
    max_val = max(cb_norms.max(), img_norms.max())
    bins = np.linspace(min_val, max_val, 200)
    
    plt.hist(
        cb_norms, 
        bins=bins, 
        color="green", 
        alpha=0.6, 
        label="Concept Vectors", 
        density=True
    )
    
    plt.hist(
        img_norms, 
        bins=bins, 
        color="red", 
        alpha=0.5, 
        label="SUN397 Images", 
        density=True
    )

    plt.title(f"Norm Distribution ({args.backbone_name})")
    plt.xlabel("L2 Norm", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    
    plt.grid(alpha=0.3, linewidth=0.5)
    plt.legend()
    
    plt.tight_layout()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot saved to {args.output_file}")

if __name__ == "__main__":
    main()