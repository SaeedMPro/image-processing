from pathlib import Path
from typing import Tuple

_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent

# Data and outputs
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Preprocessing: resize and normalization
IMAGE_SIZE: Tuple[int, int] = (256, 256)  # (H, W)
NORMALIZE_RANGE = (0.0, 1.0)

# Dataset: use 80–120 paired images
MIN_PAIRS = 80
MAX_PAIRS = 120

# Autoencoder training
AE_EPOCHS = 30
AE_LOSS = "mse"


def get_config() -> dict:
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "figures_dir": str(FIGURES_DIR),
        "models_dir": str(MODELS_DIR),
        "image_size": list(IMAGE_SIZE),
        "min_pairs": MIN_PAIRS,
        "max_pairs": MAX_PAIRS,
    }
