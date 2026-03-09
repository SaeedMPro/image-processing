"""
Classical enhancement methods: Histogram Equalization, CLAHE, Gamma Correction,
Single-Scale Retinex. Implemented with OpenCV/NumPy.
"""

from typing import Tuple

import cv2
import numpy as np


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convert [0,1] float to uint8 [0,255] for OpenCV; already uint8 passed through."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img.astype(np.uint8)


def _ensure_float01(img: np.ndarray) -> np.ndarray:
    """Convert to float [0,1] for consistent return when input was float."""
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return np.clip(img.astype(np.float64), 0, 1)


def enhance_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization on L channel (LAB). Input: (H,W,C) float [0,1] or uint8.
    Output: (H,W,C) float [0,1].
    """
    out = _ensure_uint8(img)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        out[:, :, 0] = cv2.equalizeHist(out[:, :, 0])
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    else:
        out = cv2.equalizeHist(out)
    return _ensure_float01(out)


def enhance_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE on L channel (LAB). Input: (H,W,C) float [0,1] or uint8. Output: (H,W,C) float [0,1].
    """
    out = _ensure_uint8(img)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out[:, :, 0] = clahe.apply(out[:, :, 0])
        out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        out = clahe.apply(out)
    return _ensure_float01(out)


def enhance_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction: I_out = I_in^gamma. Input: (H,W,C) float [0,1] or uint8. Output: (H,W,C) float [0,1].
    """
    out = _ensure_float01(img)
    out = np.power(out, gamma)
    return np.clip(out, 0, 1).astype(np.float64)


def enhance_ssr(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    Single-Scale Retinex: R = log(I) - log(L), L = Gaussian(I).
    Input: (H,W,C) float [0,1] or uint8. Output: (H,W,C) float [0,1].
    """
    out = _ensure_float01(img)
    out = out + 1e-6
    if out.ndim == 3:
        result = np.zeros_like(out)
        for c in range(out.shape[2]):
            channel = out[:, :, c]
            low = cv2.GaussianBlur(channel, (0, 0), sigma)
            result[:, :, c] = np.log10(channel + 1e-6) - np.log10(low + 1e-6)
        # Normalize to [0,1]
        result = result - result.min()
        result = result / (result.max() + 1e-8)
        return result.astype(np.float64)
    low = cv2.GaussianBlur(out, (0, 0), sigma)
    result = np.log10(out + 1e-6) - np.log10(low + 1e-6)
    result = result - result.min()
    result = result / (result.max() + 1e-8)
    return result.astype(np.float64)
