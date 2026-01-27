import random
import torch
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from hycoclip.hycoclip.tokenizer import Tokenizer
from hycoclip.hycoclip import lorentz as L
from hycoclip.hycoclip.config import LazyConfig, LazyFactory
from hycoclip.hycoclip.utils.checkpointing import CheckpointManager
device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

NORM_THRESHOLD = 0.27
MAX_DISTANCE = 3         # Look up to 3 levels deep (Parent -> ... -> Great Grandchild)
MAX_PAIRS = 10000
DEEP_ROOTS = [
    "mammal", "bird", "reptile", "fish", "invertebrate",
    "vehicle", "container", "electronic_device", "tool",
    "structure", "furniture", "garment",
    "food", "produce", "dish"
]

def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = LazyConfig.load(config_path)
    model = LazyFactory.build_model(cfg, device=device)
    CheckpointManager(model=model).load(checkpoint_path)
    model.eval()
    return model

def _encode_phrases(phrases: list[str]) -> dict[str, torch.Tensor]:
    backbone, _ = load_model(
        config_path="hycoclip/configs/train_hycoclip_vit_b.py",
        checkpoint_path="hycoclip/hycoclip_vit_b.pth",
        device=device_obj,
    )
    unique_phrases = sorted(list(set(phrases)))
    tokenizer = Tokenizer()
    tokens = tokenizer(unique_phrases)
    with torch.no_grad():
        feats = backbone.encode_text(tokens, project=True).detach()
    return {phrase: feat.clone() for phrase, feat in zip(unique_phrases, feats)}

def generate_transitive_pairs(roots, max_pairs=3000, max_dist=3):
    pairs = []
    # Queue stores: (synset, list_of_ancestor_names)
    # path is a list of names: ['entity', 'object', 'animal'...]
    queue = []
    for r in roots:
        syns = wn.synsets(r, pos=wn.NOUN)
        if syns:
            # Initialize with the root itself as the start of a path
            name = syns[0].lemmas()[0].name().replace('_', ' ')
            queue.append((syns[0], [name]))
            
    seen_pairs = set()
    print(f"Starting Transitive BFS crawl (Max Dist={max_dist})...")
    
    while queue and len(pairs) < max_pairs:
        current_syn, ancestor_path = queue.pop(0)
        
        # Get immediate children
        children = current_syn.hyponyms()
        random.shuffle(children)
        
        for child_syn in children:
            child_name = child_syn.lemmas()[0].name().replace('_', ' ')
            
            # 1. Create pairs with ALL ancestors in the valid window
            # Path is [Grandparent, Parent]. Child is current.
            # We iterate backwards through the path up to max_dist
            for i in range(1, min(len(ancestor_path) + 1, max_dist + 1)):
                # Ancestor at index -i (e.g., -1 is parent, -2 is grandparent)
                ancestor_name = ancestor_path[-i]
                distance = i
                
                sig = f"{ancestor_name}->{child_name}"
                if sig not in seen_pairs:
                    pairs.append((ancestor_name, child_name, distance))
                    seen_pairs.add(sig)

            # 2. Add child to queue to go deeper
            # Limit total path depth to prevent infinite loops, though max_dist limits pairs
            if len(ancestor_path) < 6: 
                new_path = ancestor_path + [child_name]
                queue.append((child_syn, new_path))
            
            if len(pairs) >= max_pairs: break
            
    return pairs

def main():
    backbone, _ = load_model(
        config_path="hycoclip/configs/train_hycoclip_vit_b.py",
        checkpoint_path="hycoclip/hycoclip_vit_b.pth",
        device=device_obj,
    )

    print("Harvesting transitive pairs...")
    raw_pairs = generate_transitive_pairs(DEEP_ROOTS, max_pairs=MAX_PAIRS, max_dist=MAX_DISTANCE)

    unique = set()
    for p, c, d in raw_pairs: unique.add(p); unique.add(c)
    print(f"Encoding {len(unique)} concepts...")
    embs = _encode_phrases(list(unique))
    curv = backbone.curv.exp()

    data = []
    skipped_thresh = 0
    skipped_inv = 0

    print("Analyzing geometry...")
    for p, c, dist in raw_pairs:
        if p not in embs or c not in embs: continue
        pv = embs[p].unsqueeze(0).to(device_obj)
        cv = embs[c].unsqueeze(0).to(device_obj)
        
        pn = torch.norm(pv, p=2, dim=-1).item()
        cn = torch.norm(cv, p=2, dim=-1).item()
        
        if pn < NORM_THRESHOLD or cn < NORM_THRESHOLD: 
            skipped_thresh += 1
            continue 
        if pn >= cn: 
            skipped_inv += 1
            continue 
        
        aperture = L.half_aperture(pv, curv, min_radius=0.04)[0]
        angle = L.pairwise_oxy_angle(pv, cv, curv=curv)[0, 0]
        eta_req = float((angle / aperture).item())
        
        data.append({
            "parent": p, 
            "child": c, 
            "distance": dist, 
            "p_norm": pn, 
            "eta": eta_req
        })

    df = pd.DataFrame(data)
    print("-" * 30)
    print(f"Skipped (Low Norm): {skipped_thresh}")
    print(f"Skipped (Inverted): {skipped_inv}")
    print(f"Valid Data Points:  {len(df)}")
    print("-" * 30)

    if not df.empty:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.optimize import curve_fit

        plt.style.use("seaborn-v0_8-paper") 
        plt.rcParams.update({
            "font.family": "serif", 
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "lines.markersize": 0
        })
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.scatterplot(
            data=df, 
            x='p_norm', 
            y='eta', 
            ax=ax,
            color='#34495e',     
            alpha=0.2,
            s=30,                
            edgecolor=None,
            label='Valid Pairs'
        )
        
        def linear_scaling(norm, slope, intercept): 
            return slope * norm + intercept

        popt, _ = curve_fit(linear_scaling, df['p_norm'], df['eta'])
        slope, intercept = popt
        r_value = df['p_norm'].corr(df['eta'])
        
        print(f"Correlation coefficient (r): {r_value:.4f}")

        x_range = np.linspace(df['p_norm'].min(), df['p_norm'].max(), 100)
        y_fit = linear_scaling(x_range, slope, intercept)

        ax.plot(
            x_range, 
            y_fit, 
            color="#e74c3c",     
            alpha=0.9,
            linewidth=2.5, 
            label='Linear Fit'
        )

        ax.set_xlabel(r"Parent Concept Norm")
        ax.set_ylabel(r"Required Entailment Ratio ($\eta$)")
        
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls=":", alpha=0.1)
        
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='white')

        plt.tight_layout()
        save_path = "scaling_law_correlation.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved to {save_path}")
        print(f"Equation: eta = {slope:.4f} * norm + {intercept:.4f}")
    else:
        print("No valid data to plot.")