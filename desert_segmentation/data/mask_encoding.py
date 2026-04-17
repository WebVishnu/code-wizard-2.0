"""Decode 16-bit raw mask values to contiguous class indices and back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RawMaskCodec:
    """Maps dataset-specific raw label IDs (e.g. uint16 PNG values) to 0..num_classes-1."""

    raw_ids: Tuple[int, ...]
    class_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.raw_ids) != len(self.class_names):
            raise ValueError("raw_ids and class_names must have the same length")
        if len(set(self.raw_ids)) != len(self.raw_ids):
            raise ValueError("raw_ids must be unique")

    @property
    def num_classes(self) -> int:
        return len(self.raw_ids)

    @property
    def raw_to_index(self) -> Dict[int, int]:
        return {r: i for i, r in enumerate(self.raw_ids)}

    @property
    def index_to_raw(self) -> Dict[int, int]:
        return {i: r for i, r in enumerate(self.raw_ids)}

    def _build_lut(self) -> np.ndarray:
        max_id = max(self.raw_ids)
        lut = np.full(max_id + 1, 255, dtype=np.uint8)
        for i, rid in enumerate(self.raw_ids):
            lut[rid] = i
        return lut

    def encode_mask(self, raw: np.ndarray) -> Tuple[np.ndarray, float]:
        """Map raw uint16 labels to uint8 class indices 0..C-1. Returns (encoded, unknown_fraction)."""
        if raw.ndim != 2:
            raise ValueError(f"Expected HxW mask, got shape {raw.shape}")
        lut = self._build_lut()
        if int(raw.max()) >= lut.size:
            raise ValueError(f"Mask value {int(raw.max())} exceeds LUT; extend raw_ids in config.")
        out = lut[raw.astype(np.int64, copy=False)]
        unknown_frac = float((out == 255).mean())
        if unknown_frac > 0:
            bad = out == 255
            raise ValueError(
                f"Unknown mask pixels: {unknown_frac:.6f} of image. "
                f"Unique unknown raw values: {np.unique(raw[bad])[:16]}"
            )
        return out.astype(np.uint8), unknown_frac

    def decode_to_raw(self, class_indices: np.ndarray) -> np.ndarray:
        """Map class indices back to raw dataset IDs (for visualization/export)."""
        arr = np.asarray(class_indices)
        raw = np.zeros_like(arr, dtype=np.uint16)
        for i, rid in enumerate(self.raw_ids):
            raw[arr == i] = rid
        return raw


def build_codec_from_config(raw_id_list: Sequence[int], names: Sequence[str]) -> RawMaskCodec:
    pairs = sorted(zip(raw_id_list, names), key=lambda x: x[0])
    r, n = zip(*pairs)
    return RawMaskCodec(raw_ids=tuple(int(x) for x in r), class_names=tuple(str(x) for x in n))


def default_desert_codec() -> RawMaskCodec:
    """Default codec for this workspace: 10 classes with fixed raw IDs (see dataset scan)."""
    raw_ids = (100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000)
    names = (
        "id_100",
        "id_200",
        "id_300",
        "id_500",
        "id_550",
        "id_600",
        "id_700",
        "id_800",
        "id_7100",
        "id_10000",
    )
    return RawMaskCodec(raw_ids=raw_ids, class_names=names)
