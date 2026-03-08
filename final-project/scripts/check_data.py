"""
Dataset validation (Step 1). Proposal: 80–120 paired images.
Run from project root: python scripts/check_data.py
Uses the lowlight package for discovery; requires opencv for shape check.
"""

import sys
from pathlib import Path

# Allow running without pip install -e . (add project root to path)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2

from lowlight.config import DATA_DIR, MAX_PAIRS, MIN_PAIRS
from lowlight.data import discover_pairs


def check_image_shape(image_path: str) -> tuple:
    """Load image and return shape (H, W, C). Requires opencv-python."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    return img.shape


def main() -> None:
    data_root = _project_root / "data" if (_project_root / "data").exists() else DATA_DIR
    if not data_root.is_dir():
        print(f"ERROR: Data directory not found: {data_root}")
        return

    pairs, low_only, normal_only = discover_pairs(str(data_root))
    n_pairs = len(pairs)

    print("=== Dataset check (Proposal: 80–120 paired images) ===\n")
    print(f"Total pairs found: {n_pairs}")
    print(f"Low-only IDs (no normal): {len(low_only)} — {low_only[:5]}{'...' if len(low_only) > 5 else ''}")
    print(f"Normal-only IDs (no low): {len(normal_only)} — {normal_only[:5]}{'...' if len(normal_only) > 5 else ''}\n")

    if n_pairs < MIN_PAIRS or n_pairs > MAX_PAIRS:
        print(f"WARNING: Pair count {n_pairs} is outside required range [{MIN_PAIRS}, {MAX_PAIRS}].")
    else:
        print("OK: Pair count within [80, 120].\n")

    if not pairs:
        print("No pairs to validate. Exiting.")
        return

    path_low, path_normal = pairs[0]
    try:
        shape_low = check_image_shape(path_low)
        shape_normal = check_image_shape(path_normal)
        print(f"Sample low   image shape: {shape_low}   ({path_low})")
        print(f"Sample normal image shape: {shape_normal}   ({path_normal})")
        if shape_low != shape_normal:
            print("Note: Low and normal shapes differ; preprocessing will resize to same size (e.g. 256×256).")
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    print("\nValidation: dataset ready for preprocessing (resize + normalize) in next step.")


if __name__ == "__main__":
    main()
