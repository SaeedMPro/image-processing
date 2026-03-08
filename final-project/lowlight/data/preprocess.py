"""
Preprocessing: load, resize, normalize, optional augmentation.
Proposal §3: Resize to fixed resolution; normalize [0,1]; optional horizontal flip or small rotation.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from lowlight.config import IMAGE_SIZE
from lowlight.data.discovery import discover_pairs


def load_image(path: str, as_rgb: bool = True) -> np.ndarray:
    """
    Load a single image from disk.
    Returns: (H, W, C) uint8 [0, 255].
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize to (height, width). Proposal: fixed resolution e.g. 256×256."""
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)


def normalize_01(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1]. Proposal: required preprocessing."""
    return img.astype(np.float64) / 255.0


def _augment_flip(img: np.ndarray) -> np.ndarray:
    """Horizontal flip. Proposal: allowed augmentation."""
    return np.flip(img, axis=1).copy()


def _augment_rotate(img: np.ndarray, angle_deg: float, border_value: float = 0.0) -> np.ndarray:
    """Small rotation. Proposal: allowed augmentation."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=border_value)


def load_and_preprocess_pair(
    path_low: str,
    path_normal: str,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    normalize: bool = True,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one low/normal pair: resize, optional augment, optional normalize.
    Returns: (low_img, normal_img) same shape, float [0,1] or uint8 [0,255].
    """
    low = load_image(path_low, as_rgb=True)
    normal = load_image(path_normal, as_rgb=True)
    low = resize_image(low, target_size)
    normal = resize_image(normal, target_size)

    if augment and rng is not None:
        if rng.random() > 0.5:
            low = _augment_flip(low)
            normal = _augment_flip(normal)
        else:
            angle = float(rng.uniform(-15, 15))
            low = _augment_rotate(low, angle)
            normal = _augment_rotate(normal, angle)

    if normalize:
        low = normalize_01(low)
        normal = normalize_01(normal)
    return low, normal


def load_dataset(
    data_root: str,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    normalize: bool = True,
    augment: bool = False,
    seed: Optional[int] = None,
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Load full dataset with preprocessing.
    Returns: low_imgs (N,H,W,C), normal_imgs (N,H,W,C), pair_paths.
    """
    pairs, _, _ = discover_pairs(data_root)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    rng = np.random.default_rng(seed) if augment else None

    low_list = []
    normal_list = []
    for path_low, path_normal in pairs:
        low, normal = load_and_preprocess_pair(
            path_low, path_normal,
            target_size=target_size,
            normalize=normalize,
            augment=augment,
            rng=rng,
        )
        low_list.append(low)
        normal_list.append(normal)

    return (
        np.stack(low_list, axis=0),
        np.stack(normal_list, axis=0),
        pairs,
    )
