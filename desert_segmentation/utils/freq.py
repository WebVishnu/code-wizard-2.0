"""Estimate class pixel frequencies from mask files (fast path for loss weighting)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from desert_segmentation.data.mask_encoding import RawMaskCodec


def list_masks(dir_path: Path) -> List[str]:
    return sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".png"))


@torch.no_grad()
def estimate_pixel_frequencies(
    masks_dir: Path,
    codec: RawMaskCodec,
    max_files: int | None = 800,
) -> torch.Tensor:
    paths = list_masks(masks_dir)
    if max_files is not None:
        paths = paths[:max_files]
    counts = np.zeros(codec.num_classes, dtype=np.int64)
    for name in paths:
        raw = np.array(Image.open(masks_dir / name))
        enc, _ = codec.encode_mask(raw.astype(np.uint16))
        for c in range(codec.num_classes):
            counts[c] += int((enc == c).sum())
    freq = counts.astype(np.float64) / max(counts.sum(), 1)
    return torch.tensor(freq, dtype=torch.float32)


def per_image_sampling_weights(
    masks_dir: Path,
    image_basenames: Sequence[str],
    codec: RawMaskCodec,
    freq: torch.Tensor,
    eps: float = 1e-6,
) -> torch.DoubleTensor:
    """Weights for ``WeightedRandomSampler``: upweight images containing rare classes.

    For each mask, ``w_i = sum_{c : n_{ic}>0} 1 / (freq[c] + eps)``, then weights are
    scaled to mean 1.0. ``image_basenames`` must match the order of
    ``SegmentationDataset`` indices (same filenames as train pairs).
    """
    masks_dir = Path(masks_dir)
    f = freq.detach().cpu().numpy().astype(np.float64)
    raw_weights = np.zeros(len(image_basenames), dtype=np.float64)
    for i, name in enumerate(image_basenames):
        raw = np.array(Image.open(masks_dir / name))
        enc, _ = codec.encode_mask(raw.astype(np.uint16))
        present = np.zeros(codec.num_classes, dtype=bool)
        for c in range(codec.num_classes):
            present[c] = bool((enc == c).any())
        raw_weights[i] = sum(1.0 / (f[c] + eps) for c in range(codec.num_classes) if present[c])
    m = raw_weights.mean()
    if m <= 0:
        return torch.ones(len(image_basenames), dtype=torch.double)
    scaled = raw_weights / m
    return torch.tensor(scaled, dtype=torch.double)
