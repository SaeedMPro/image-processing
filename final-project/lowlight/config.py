"""
Central configuration. Paths and constants used across the project.
Resolves project root relative to this file so notebooks/scripts can run from any cwd.
"""

from pathlib import Path
from typing import Tuple

# Package root (lowlight/) -> project root (final-project/)
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent

# Data and outputs
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Preprocessing (Proposal §3)
IMAGE_SIZE: Tuple[int, int] = (256, 256)  # (H, W)
NORMALIZE_RANGE = (0.0, 1.0)

# Dataset constraint (Proposal §2)
MIN_PAIRS = 80
MAX_PAIRS = 120

# Phase 2B autoencoder (Proposal §4.2)
AE_EPOCHS = 25  # 20–30 allowed
AE_LOSS = "mse"


def get_config() -> dict:
    """Return config dict for notebook/report (paths as strings)."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "figures_dir": str(FIGURES_DIR),
        "models_dir": str(MODELS_DIR),
        "image_size": list(IMAGE_SIZE),
        "min_pairs": MIN_PAIRS,
        "max_pairs": MAX_PAIRS,
    }
