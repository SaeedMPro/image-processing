"""
Phase 1 handcrafted features (Proposal §4.1).
From grayscale: Mean Intensity, Standard Deviation (Contrast), Entropy, Histogram Skewness.
"""

from typing import List

import numpy as np
from scipy import stats


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,C) or (H,W) to (H,W) float [0,1]. Accepts [0,1] or [0,255]."""
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = np.asarray(img, dtype=np.float64)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extract Phase 1 features from one image (grayscale).
    Proposal §4.1: Mean Intensity, Std (Contrast), Entropy, Histogram Skewness.
    Inputs: img (H,W) or (H,W,C), float [0,1] or uint8 [0,255].
    Outputs: shape (4,) [mean, std, entropy, skewness].
    """
    gray = _to_grayscale(img).flatten()
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std == 0:
        std = 1e-8
    # Entropy: -sum(p * log2(p)) over histogram (256 bins for [0,1] scaled to 0..255)
    hist, _ = np.histogram((gray * 255).clip(0, 255).astype(np.int32), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    skewness = float(stats.skew(gray))
    return np.array([mean, std, entropy, skewness], dtype=np.float64)


def extract_features_batch(imgs: np.ndarray) -> np.ndarray:
    """
    Extract features for a batch of images.
    Inputs: imgs (N, H, W, C) or (N, H, W).
    Outputs: (N, 4) feature matrix.
    """
    return np.array([extract_features(imgs[i]) for i in range(len(imgs))])
