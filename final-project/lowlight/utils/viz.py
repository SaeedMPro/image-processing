"""Visualization helpers for report figures."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_pair(
    low: np.ndarray,
    normal: np.ndarray,
    title: str = "Low vs Normal",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Plot low-light and normal-light pair side by side."""
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(np.clip(low, 0, 1))
    ax[0].set_title("Low-light")
    ax[0].axis("off")
    ax[1].imshow(np.clip(normal, 0, 1))
    ax[1].set_title("Normal-light (reference)")
    ax[1].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_enhancement_comparison(
    low: np.ndarray,
    enhanced_list: List[np.ndarray],
    reference: np.ndarray,
    method_names: List[str],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot: low | enhanced_1 | ... | enhanced_k | reference.
    For Phase 3 visual comparison (Proposal §4.3).
    """
    n = 2 + len(enhanced_list)
    if figsize is None:
        figsize = (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes[0].imshow(np.clip(low, 0, 1))
    axes[0].set_title("Low-light")
    axes[0].axis("off")
    for i, (img, name) in enumerate(zip(enhanced_list, method_names)):
        axes[1 + i].imshow(np.clip(img, 0, 1))
        axes[1 + i].set_title(name)
        axes[1 + i].axis("off")
    axes[-1].imshow(np.clip(reference, 0, 1))
    axes[-1].set_title("Reference")
    axes[-1].axis("off")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
