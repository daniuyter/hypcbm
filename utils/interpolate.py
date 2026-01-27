#!/usr/bin/env python3
"""Interpolate metrics for a target average of active concepts per class.

This utility consumes the JSON summary emitted by
``train_pcbm_hycoclip.py --viz-concept-accuracy`` and performs a simple
piecewise-linear interpolation over the average number of active concepts per
class ("k").  The script reports the interpolated lambda, accuracies, and
active-concept counts associated with an arbitrary target k value.
"""
from __future__ import annotations

import argparse
import json
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Each JSON entry must be an object/dict")
    return data


def _mean(values: Iterable[Real]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence")
    return float(sum(values) / len(values))


def _augment_with_average(
    entries: List[Dict[str, Any]],
    explicit_num_classes: int | None = None,
) -> List[Dict[str, Any]]:
    augmented = []
    for idx, entry in enumerate(entries):
        per_class = entry.get("active_concepts_per_class")
        avg = None
        num_classes = None
        if isinstance(per_class, list) and per_class and all(
            isinstance(val, (int, float)) for val in per_class
        ):
            num_classes = len(per_class)
            avg = _mean(per_class)
        else:
            total = entry.get("active_concepts")
            if total is not None:
                if explicit_num_classes is None:
                    raise ValueError(
                        "Entry #{idx} is missing 'active_concepts_per_class'; provide --num-classes"
                    )
                num_classes = int(explicit_num_classes)
                avg = float(total) / num_classes
        if avg is None or num_classes is None:
            raise ValueError(
                f"Could not determine average active concepts for entry #{idx}"
            )
        augmented_entry = dict(entry)
        augmented_entry["avg_active_concepts_per_class"] = avg
        augmented_entry["num_classes"] = num_classes
        augmented.append(augmented_entry)
    return augmented


def _find_bracketing_entries(
    entries: List[Dict[str, Any]], target_k: float
) -> Tuple[Dict[str, Any], Dict[str, Any], float, bool]:
    entries_sorted = sorted(entries, key=lambda item: item["avg_active_concepts_per_class"])
    min_entry = entries_sorted[0]
    max_entry = entries_sorted[-1]

    if target_k <= min_entry["avg_active_concepts_per_class"]:
        return min_entry, min_entry, 0.0, True  # clamped to min
    if target_k >= max_entry["avg_active_concepts_per_class"]:
        return max_entry, max_entry, 0.0, True  # clamped to max

    for low, high in zip(entries_sorted[:-1], entries_sorted[1:]):
        low_avg = low["avg_active_concepts_per_class"]
        high_avg = high["avg_active_concepts_per_class"]
        if low_avg <= target_k <= high_avg and high_avg != low_avg:
            ratio = (target_k - low_avg) / (high_avg - low_avg)
            return low, high, ratio, False
    raise RuntimeError("Failed to locate bracketing entries for interpolation")


def _interpolate_value(low: Any, high: Any, ratio: float) -> Any:
    if isinstance(low, Real) and isinstance(high, Real):
        return float(low + ratio * (high - low))
    if isinstance(low, list) and isinstance(high, list) and len(low) == len(high):
        return [_interpolate_value(lv, hv, ratio) for lv, hv in zip(low, high)]
    # Fall back to nearest neighbor when interpolation is undefined
    return high if ratio >= 0.5 else low


def _interpolate_entry(
    low: Dict[str, Any],
    high: Dict[str, Any],
    ratio: float,
) -> Dict[str, Any]:
    keys = set(low) | set(high)
    interpolated = {}
    for key in keys:
        low_val = low.get(key)
        high_val = high.get(key, low_val)
        if low_val is None:
            interpolated[key] = high_val
        elif high_val is None:
            interpolated[key] = low_val
        else:
            interpolated[key] = _interpolate_value(low_val, high_val, ratio)
    return interpolated


def _format_entry(entry: Dict[str, Any]) -> str:
    return (
        f"lambda={entry.get('lambda'):.4g}, "
        f"test_acc={entry.get('test_acc'):.3f}%, "
        f"train_acc={entry.get('train_acc'):.3f}%, "
        f"active_concepts={entry.get('active_concepts')}, "
        f"avg_per_class={entry.get('avg_active_concepts_per_class'):.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate lambda/accuracy values for a target average number of "
            "active concepts per class."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the JSON summary produced by the visualization sweep.",
    )
    parser.add_argument(
        "-k",
        "--target-k",
        type=float,
        required=True,
        help="Desired average number of active concepts per class to interpolate to.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Optional explicit number of classes, used only if the JSON entries "
            "lack 'active_concepts_per_class'."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the interpolated entry as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.json_path)
    augmented = _augment_with_average(entries, args.num_classes)
    lower, upper, ratio, clamped = _find_bracketing_entries(augmented, args.target_k)

    if clamped:
        interpolated = lower
        ratio = 0.0
        note = "(target outside range; returning nearest entry)"
    else:
        interpolated = _interpolate_entry(lower, upper, ratio)
        interpolated["avg_active_concepts_per_class"] = args.target_k
        note = ""

    print("")
    print(f"Target average active concepts per class: {args.target_k:.4f}")
    if note:
        print(note)
    print("Lower anchor : " + _format_entry(lower))
    print("Upper anchor : " + _format_entry(upper))
    print(f"Interpolation ratio: {ratio:.3f}")
    print("---")
    print("Interpolated : " + _format_entry(interpolated))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("w", encoding="utf-8") as handle:
            json.dump(interpolated, handle, indent=2)
        print(f"Saved interpolated entry to {args.save}")


if __name__ == "__main__":
    main()
