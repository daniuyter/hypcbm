import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List
import numpy as np
import torch
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parent
HYCOCLIP_ROOT = REPO_ROOT / "hycoclip"
if str(HYCOCLIP_ROOT) not in sys.path:
    sys.path.append(str(HYCOCLIP_ROOT))
from hycoclip.hycoclip import lorentz as L
from hycoclip.hycoclip.config import LazyConfig, LazyFactory
from hycoclip.hycoclip.tokenizer import Tokenizer
from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

import clip


def clean_concepts(concepts: List[str]) -> List[str]:
    """Clean and deduplicate concepts."""
    cleaned = set()
    for original in concepts:
        concept = original.strip()
        if not concept:
            continue
        # Filter out very long phrases that might be noise
        subwords = concept.split(" ")
        if len(subwords) > 5:
            continue
        cleaned.add(concept)
    return sorted(cleaned)

def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = LazyConfig.load(config_path)
    model = LazyFactory.build_model(cfg, device=device)
    CheckpointManager(model=model).load(checkpoint_path)
    model.eval()
    return model

@torch.no_grad()
def learn_conceptbank(args, concept_list: List[str], scenario: str, model: torch.nn.Module):
    concept_dict = {}
    tokenizer = Tokenizer()
    min_norm = float(getattr(args, "min_norm", 0.0))
    filtered_count = 0
    
    # Batch processing could be added here if list is huge, but loop is fine for offline
    for concept in tqdm(concept_list, desc="Encoding concepts"):
        print(concept)
        #for hycoclip/meru/clip
        text_tokens = tokenizer(concept)

        # for OpenAI CLIP:
        #text_tokens = clip.tokenize(concept).to(args.device)
        
        #for hycoclip/meru/clip
        text_features = model.encode_text(text_tokens, project=True)
        # print(text_features.shape)

        # when using CLIP:
        # text_features = L.exp_map0(
        #     text_features,
        #     torch.tensor(0.1, device=text_features.device),
        # ).cpu().numpy()
    
        # for OpenAI CLIP:
        #text_features = model.encode_text(text_tokens).cpu().numpy()

        l2_norm = float(np.linalg.norm(text_features))
        if l2_norm < min_norm:
            filtered_count += 1
            continue
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    if filtered_count:
        print(
            f"Filtered {filtered_count} concepts with norm below {min_norm:.4f}."
        )
    if not concept_dict:
        raise RuntimeError(
            "No concepts remaining after norm filtering. Lower --min-norm or adjust inputs."
        )
    concept_dict_path = os.path.join(
        args.out_dir,
        f"multimodal_concept_{args.backbone_name}_{args.scenario}_minnorm:{min_norm:.2f}.pkl",
    )
    with open(concept_dict_path, "wb") as handle:
        pickle.dump(concept_dict, handle)
    print(f"Dumped to : {concept_dict_path}")


def config():
    parser = argparse.ArgumentParser(description="Encode concepts from a text file into embeddings.")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory for the concept bank.")
    parser.add_argument(
        "--concepts-txt",
        required=True,
        type=str,
        help="Path to a .txt file with one concept per line.",
    )
    parser.add_argument("--scenario", default="custom", type=str, help="Name/label for this scenario (used in output filename).")
    parser.add_argument("--backbone-name", default="hycoclip", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--min-norm",
        default=0.0,
        type=float,
        help="Discard encoded concepts whose embedding L2 norm falls below this threshold. ONLY USE FOR HYPERBOLIC EMBEDDINGS.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    device = torch.device(args.device)
    
    print(f"Loading model {args.backbone_name}...")

    model = load_model(
        config_path="hycoclip/configs/train_hycoclip_vit_b.py",
        checkpoint_path="hycoclip/hycoclip_vit_b.pth",
        device=device,
    )
    
    # for original clip:
    # model, _ = clip.load('ViT-B/16', device=device, jit=False)

    # Load concepts from text file
    concepts_path = Path(args.concepts_txt)
    if not concepts_path.exists():
        raise FileNotFoundError(f"Concepts file not found at {concepts_path}")
    
    raw_concepts: list[str] = []
    with concepts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_concepts.append(line)
    
    concepts = clean_concepts(raw_concepts)
    print(f"Loaded {len(concepts)} concepts from text file {concepts_path}.")
    
    learn_conceptbank(args, concepts, args.scenario, model)
