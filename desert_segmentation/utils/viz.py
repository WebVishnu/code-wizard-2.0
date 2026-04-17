"""Color overlays and side-by-side panels for segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def palette(num_classes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(32, 256, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([128, 128, 128], dtype=np.uint8)
    return colors


def colorize_mask(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """mask HxW int 0..C-1 -> RGB uint8"""
    m = mask.clip(0, len(colors) - 1)
    return colors[m]


def blend_overlay(
    image_rgb: np.ndarray,
    colored_mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    return (image_rgb.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha).clip(
        0, 255
    ).astype(np.uint8)


def save_triplet(
    out_path: Path,
    rgb: np.ndarray,
    gt: np.ndarray | None,
    pred: np.ndarray,
    class_colors: np.ndarray,
    titles: Tuple[str, str, str] = ("RGB", "GT", "Pred"),
) -> None:
    h, w = rgb.shape[:2]
    panels: List[np.ndarray] = [rgb]
    if gt is not None:
        panels.append(blend_overlay(rgb, colorize_mask(gt, class_colors)))
    else:
        panels.append(np.zeros_like(rgb))
    panels.append(blend_overlay(rgb, colorize_mask(pred, class_colors)))

    # Optional text strip (simple border)
    gap = 8
    total_w = w * len(panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    x = 0
    for p in panels:
        canvas[:, x : x + w] = p
        x += w + gap
    Image.fromarray(canvas).save(out_path)

