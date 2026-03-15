"""
Quantitative metrics: PSNR and SSIM. MAX_I=255, K1=0.01, K2=0.03.
"""

import numpy as np

# Optional: use skimage for SSIM (matches common implementation)
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Standard constants for PSNR/SSIM
MAX_I = 255.0
K1 = 0.01
K2 = 0.03
L = MAX_I
C1 = (K1 * L) ** 2
C2 = (K2 * L) ** 2


def _to_float_0_255(img: np.ndarray) -> np.ndarray:
    """Convert to float in [0, 255] for metric formulas."""
    if img.dtype == np.uint8:
        return img.astype(np.float64)
    if img.max() <= 1.0:
        return (np.clip(img, 0, 1) * MAX_I).astype(np.float64)
    return np.clip(img.astype(np.float64), 0, MAX_I)


def psnr(enhanced: np.ndarray, reference: np.ndarray) -> float:
    """
    PSNR in dB. Higher = better. PSNR = 10 * log10(MAX_I^2 / MSE).
    Inputs: enhanced and reference (H,W,C) or (H,W), [0,1] or [0,255].
    """
    e = _to_float_0_255(enhanced)
    r = _to_float_0_255(reference)
    mse = np.mean((e - r) ** 2)
    if mse == 0:
        return 100.0  # identical
    return float(10.0 * np.log10(MAX_I ** 2 / mse))


def ssim(enhanced: np.ndarray, reference: np.ndarray, channel_axis: int = -1) -> float:
    """
    SSIM. Range ~[-1,1], typically [0,1]. 1 = identical.
    Uses skimage if available (same formula with C1, C2); else simple global implementation.
    Inputs: enhanced, reference (H,W,C); channel_axis for multichannel.
    """
    e = _to_float_0_255(enhanced)
    r = _to_float_0_255(reference)
    if HAS_SKIMAGE:
        return float(ssim_skimage(r, e, data_range=MAX_I, channel_axis=channel_axis if e.ndim == 3 else None))
    # Fallback: per-channel _ssim_single then mean
    if e.ndim == 3:
        return float(np.mean([_ssim_single(e[:, :, c], r[:, :, c]) for c in range(e.shape[-1])]))
    return _ssim_single(e, r)


def _ssim_single(x: np.ndarray, y: np.ndarray, win_size: int = 7) -> float:
    """Single-channel SSIM."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    from scipy.ndimage import uniform_filter
    np.seterr(divide="ignore", invalid="ignore")
    ux = uniform_filter(x, win_size)
    uy = uniform_filter(y, win_size)
    ux2 = uniform_filter(x * x, win_size)
    uy2 = uniform_filter(y * y, win_size)
    uxy = uniform_filter(x * y, win_size)
    vx = ux2 - ux * ux
    vy = uy2 - uy * uy
    vxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux**2 + uy**2 + C1) * (vx + vy + C2)
    out = num / den
    np.seterr(divide="warn", invalid="warn")
    return float(np.nanmean(out))
