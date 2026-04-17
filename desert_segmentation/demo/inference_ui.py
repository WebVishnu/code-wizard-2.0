"""Helpers for Gradio / web demo: legend, validation, composites."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


from desert_segmentation.utils.viz import palette


def validate_rgb_array(
    arr: np.ndarray,
    max_side: int = 4096,
    max_megapixels: float = 16.0,
) -> None:
    """Raises ValueError with a user-facing message if invalid or too large."""
    if arr is None:
        raise ValueError("No image provided.")
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {getattr(arr, 'shape', None)}")
    h, w = arr.shape[0], arr.shape[1]
    if h < 1 or w < 1:
        raise ValueError("Image is empty.")
    if max(h, w) > max_side:
        raise ValueError(f"Image too large: max side is {max_side}px (got {h}x{w}).")
    mp = (h * w) / 1_000_000.0
    if mp > max_megapixels:
        raise ValueError(f"Image too large: max {max_megapixels} megapixels (got {mp:.1f} MP).")


def build_legend_rows(class_names: Sequence[str], num_classes: int, seed: int = 42) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Returns list of {index, name, hex, r, g, b} and color table (same seed as viz.palette)."""
    colors = palette(num_classes, seed=seed)
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(class_names):
        r, g, b = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        rows.append(
            {
                "index": i,
                "name": str(name),
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "r": r,
                "g": g,
                "b": b,
            }
        )
    return rows, colors


def legend_table_html(rows: Sequence[Dict[str, Any]]) -> str:
    """Small HTML table with color swatches for Gradio gr.HTML."""
    parts = [
        "<table style='border-collapse:collapse;font-size:14px'>",
        "<thead><tr><th>Swatch</th><th>#</th><th>Name</th><th>Hex</th></tr></thead><tbody>",
    ]
    for row in rows:
        sw = f"background-color:{row['hex']};width:32px;height:22px;border:1px solid #888"
        safe_name = html.escape(str(row["name"]))
        parts.append(
            f"<tr><td><div style='{sw}'></div></td>"
            f"<td>{row['index']}</td><td>{safe_name}</td><td><code>{row['hex']}</code></td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def dominant_classes_markdown(pred: np.ndarray, class_names: Sequence[str], top_k: int = 3) -> str:
    flat = pred.reshape(-1).astype(np.int64, copy=False)
    n = len(class_names)
    counts = np.bincount(flat, minlength=n)
    total = int(counts.sum())
    if total == 0:
        return "_No pixels._"
    order = np.argsort(-counts)
    lines: List[str] = []
    for i in order[:top_k]:
        c = int(counts[i])
        if c == 0:
            continue
        pct = 100.0 * c / total
        name = class_names[i] if i < len(class_names) else str(i)
        lines.append(f"- **{name}** (class {i}): **{pct:.1f}%**")
    return "\n".join(lines) if lines else "_No dominant classes._"


def side_by_side_strip(rgb: np.ndarray, mask_rgb: np.ndarray, overlay_rgb: np.ndarray, gap: int = 8) -> np.ndarray:
    """Horizontal strip: RGB | colored mask | overlay."""
    h, w = rgb.shape[:2]
    gap_arr = np.zeros((h, gap, 3), dtype=np.uint8)
    return np.concatenate([rgb, gap_arr, mask_rgb, gap_arr, overlay_rgb], axis=1)
