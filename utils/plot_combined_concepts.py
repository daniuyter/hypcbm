"""
    Example usage:
        python plot_combined_concepts.py \
        --json-paths .. .. .. .. \
        --num-classes 397 \
        --labels "HypCBM (HyCoCLIP)" "PCBM (HyCoCLIP)" "PCBM (CLIP-20M)" "PCBM (CLIP-400M)"\
        --out-path outputs/accuracy_vs_active_concepts_sun397.pdf
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
import matplotlib.pyplot as plt

STYLE_CONFIG = {
    "hypcbm":   {"color": "#991B1B", "ls": "-",  "lw": 3, "zorder": 10},
    "clip400m": {"color": "black",   "ls": "--", "lw": 1.5, "zorder": 5},
    "hycoclip": {"color": "#E18931", "ls": "-",  "lw": 2.0, "zorder": 4},
    "clip":     {"color": "#1154C8", "ls": "-",  "lw": 2.0, "zorder": 3}, 
    "other":    {"color": "gray",    "ls": "-",  "lw": 1.5, "zorder": 1}
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-ready CBM curves.")
    parser.add_argument("--json-paths", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--num-classes", type=float, default=10.0)
    parser.add_argument("--out-path", type=str, default="plot.pdf")
    parser.add_argument("--log-scale", action="store_true", default=True, help="Use log x-axis")
    parser.add_argument("--annotate-gap", action="store_true", help="Draw arrow at K=5 gap")
    parser.add_argument("--xlim", type=float, default=480.0, help="Max concepts to show")
    return parser.parse_args()

def _load_series(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _prepare_series(data: Iterable[dict], num_classes: float) -> tuple:
    points = []
    for entry in data:
        active = float(entry["active_concepts"]) / max(num_classes, 1e-8)
        accuracy = float(entry["test_acc"])
        points.append((active, accuracy))
    
    points.sort(key=lambda x: x[0])
    return [p[0] for p in points], [p[1] for p in points]

def _classify_label(label: str) -> str:
    l = label.lower()
    if "clip400m" in l or "400m" in l: return "clip400m"
    if "hypcbm" in l or "entail" in l:   return "hypcbm"
    if "hyco" in l:                      return "hycoclip"
    if "clip" in l:                      return "clip"
    return "other"

def _extend_palette(base_colors: List[str], num_colors: int) -> List[str]:
    """Extend a base color palette to fit the required number of colors."""
    extended_colors = base_colors.copy()
    while len(extended_colors) < num_colors:
        extended_colors.extend(base_colors)
    return extended_colors[:num_colors]

def main() -> None:
    args = _parse_args()
    paths = [Path(p).resolve() for p in args.json_paths]
    labels = args.labels if args.labels else [p.stem for p in paths]

    plt.style.use("seaborn-v0_8-paper") 
    plt.rcParams.update({
        "font.family": "serif",  
        "font.size": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.markersize": 6 
    })

    is_sun_layout = any("sun" in str(p).lower() for p in paths)

    if is_sun_layout and "nothreshold" in args.out_path:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    elif is_sun_layout:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        axes = [ax]

    plotted_data = []
    base_cub_colors = ["#991B1B", "#0072B2", "#009E73", "purple", "#CC79A7"]
    cub_colors = _extend_palette(base_cub_colors, len(paths))

    for path, label, color in zip(paths, labels, cub_colors):
        series = _load_series(path)
        xs, ys = _prepare_series(series, args.num_classes)

        style_key = _classify_label(label)
        
        current_ax = axes[0]
        style = STYLE_CONFIG.get(style_key, STYLE_CONFIG["other"])

        current_ax.plot(xs, ys, label=label, **style)

        plotted_data.append({"xs": xs, "ys": ys, "style_key": style_key})

    # ---- uncomment to enable log scale -----

    # if args.log_scale:
    #     ax.set_xscale("log")
    #     ax.set_xlim(1.0, 2000) # Ignore <1 concept (broken region)
    #     # Custom ticks for log scale to make it readable
    #     ax.set_xticks([1, 5, 10, 50, 100, 500, 1000])


    #     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    # else:
    #     ax.set_xlim(0, args.xlim)

    for i, ax_ in enumerate(axes):
        ax_.set_xlim(0.1, 300)

        if not is_sun_layout:
            ax_.set_xlabel("Average Active Concepts")

        if i == 0:
            ax_.set_ylabel("Accuracy (%)")
        
        ax_.grid(True, which="major", ls="-", alpha=0.3)
        ax_.grid(True, which="minor", ls=":", alpha=0.1)
        legend = ax_.legend(loc="lower right", frameon=True, framealpha=0.6, edgecolor='white')

    if is_sun_layout:
         plt.tight_layout(rect=[0, 0.05, 1, 1])
         fig.text(0.5, 0.06, "Average Active Concepts", ha='center')
    else:
         plt.tight_layout()
    plt.savefig(args.out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()