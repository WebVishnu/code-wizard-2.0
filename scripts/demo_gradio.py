#!/usr/bin/env python3
"""Gradio demo: upload RGB image, get colored mask, overlay, legend, and timing."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from desert_segmentation.demo.inference_ui import (
    build_legend_rows,
    dominant_classes_markdown,
    legend_table_html,
    side_by_side_strip,
    validate_rgb_array,
)
from desert_segmentation.infer.predict import _load_model_for_inference, predict_image
from desert_segmentation.utils.viz import blend_overlay, colorize_mask

logger = logging.getLogger(__name__)

_STATE: Dict[str, Any] = {}


def _to_uint8_rgb(arr: Any) -> np.ndarray:
    if arr is None:
        raise gr.Error("Please upload an image.")
    if isinstance(arr, Image.Image):
        arr = np.array(arr.convert("RGB"))
    a = np.asarray(arr)
    if a.ndim == 2:
        raise gr.Error("Expected a color RGB image, got grayscale.")
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise gr.Error(f"Expected HxWx3 RGB image, got shape {a.shape}.")
    if np.issubdtype(a.dtype, np.floating) and float(a.max()) <= 1.0 + 1e-6:
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _init_state(checkpoint: Path, device: torch.device) -> None:
    global _STATE
    if _STATE:
        return
    logger.info("Loading checkpoint: %s", checkpoint)
    model, cfg, codec = _load_model_for_inference(checkpoint, device)
    icfg = cfg.get("inference") or {}
    legend_rows, colors = build_legend_rows(codec.class_names, codec.num_classes, seed=42)
    _STATE.update(
        {
            "model": model,
            "cfg": cfg,
            "codec": codec,
            "device": device,
            "icfg": icfg,
            "legend_rows": legend_rows,
            "colors": colors,
            "legend_html_static": legend_table_html(legend_rows),
        },
    )
    logger.info(
        "Model ready | classes=%s | device=%s | default tile=%s overlap=%s tta=%s",
        codec.num_classes,
        device,
        icfg.get("tile_size", 512),
        icfg.get("overlap", 0.25),
        icfg.get("tta_flip", True),
    )


def _run(
    image_input: Any,
    use_tta: bool,
    overlap: float,
    tile_size: float,
    max_side: int,
    max_megapixels: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    rgb = _to_uint8_rgb(image_input)
    try:
        validate_rgb_array(rgb, max_side=max_side, max_megapixels=max_megapixels)
    except ValueError as e:
        raise gr.Error(str(e)) from e

    st = _STATE
    model = st["model"]
    device = st["device"]
    icfg = st["icfg"]
    codec = st["codec"]
    colors = st["colors"]

    tile = int(round(float(tile_size))) if tile_size is not None else int(icfg.get("tile_size", 512))
    tile = max(256, min(tile, 2048))
    ov = float(overlap)
    ov = max(0.0, min(ov, 0.5))

    t0 = time.perf_counter()
    pred = predict_image(model, rgb, device, tile, ov, bool(use_tta))
    ms = (time.perf_counter() - t0) * 1000.0

    colored = colorize_mask(pred, colors)
    overlay = blend_overlay(rgb, colored)
    strip = side_by_side_strip(rgb, colored, overlay)

    dev_str = str(device)
    if device.type == "cpu":
        dev_str += " (CPU mode — slower than GPU)"

    stats = (
        f"**Inference:** {ms:.1f} ms  \n"
        f"**Device:** {dev_str}  \n"
        f"**Tile size:** {tile} | **Overlap:** {ov:.2f} | **TTA:** {use_tta}"
    )
    dominant = "### Dominant classes in this image\n" + dominant_classes_markdown(pred, codec.class_names, top_k=3)

    return rgb, colored, overlay, strip, stats, dominant


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Gradio demo for desert semantic segmentation")
    parser.add_argument("--root", type=str, default=os.environ.get("ROOT"), help="Workspace root (default: repo root or env ROOT)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("CHECKPOINT_PATH"),
        help="Path to best.pt (default: env CHECKPOINT_PATH or <root>/checkpoints/best.pt)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--max-side", type=int, default=4096)
    parser.add_argument("--max-megapixels", type=float, default=16.0)
    args = parser.parse_args()

    root = Path(args.root or ROOT).resolve()
    ckpt_arg = args.checkpoint or str(root / "checkpoints" / "best.pt")
    ckpt = Path(ckpt_arg)
    if not ckpt.is_absolute():
        ckpt = (root / ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _init_state(ckpt, device)

    icfg = _STATE["icfg"]
    def_tta = bool(icfg.get("tta_flip", True))
    def_ov = float(icfg.get("overlap", 0.25))
    def_tile = float(icfg.get("tile_size", 512))

    intro = """## Desert semantic segmentation demo

This is **semantic segmentation**: each pixel is assigned one of several **classes** (terrain, vegetation, sky, etc.).  
It is **not** bounding-box object detection.

**How to read the outputs:**  
- **Colored mask:** each color is one class (see legend).  
- **Overlay:** prediction blended on your photo.  
- **Strip:** original | mask | side-by-side for screenshots.

_Confidence heatmaps for full-resolution sliding windows are not in this demo (v1); see README._
"""

    cpu_note = ""
    if device.type == "cpu":
        cpu_note = "\n\n> Running on **CPU** — expect slower inference. Use a CUDA GPU for best speed.\n"

    with gr.Blocks(title="Desert segmentation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(intro + cpu_note)
        inp = gr.Image(type="numpy", label="Upload RGB image", sources=["upload"])
        with gr.Accordion("Advanced", open=False):
            use_tta = gr.Checkbox(label="TTA (horizontal flip average)", value=def_tta)
            overlap = gr.Slider(0.0, 0.5, value=def_ov, step=0.05, label="Tile overlap")
            tile_sz = gr.Slider(256, 2048, value=int(def_tile), step=64, label="Tile size (pixels)")
        run_btn = gr.Button("Run segmentation", variant="primary")

        with gr.Row():
            out_orig = gr.Image(label="Input", type="numpy")
            out_mask = gr.Image(label="Colored class mask", type="numpy")
            out_overlay = gr.Image(label="Overlay", type="numpy")
        out_strip = gr.Image(label="RGB | mask | overlay", type="numpy")
        stats_md = gr.Markdown("")
        dominant_md = gr.Markdown("")
        gr.Markdown("### Class legend (fixed palette)")
        gr.HTML(_STATE["legend_html_static"])

        def _fn(img, tta, ov, ts):
            return _run(img, tta, ov, ts, args.max_side, args.max_megapixels)

        run_btn.click(
            fn=_fn,
            inputs=[inp, use_tta, overlap, tile_sz],
            outputs=[out_orig, out_mask, out_overlay, out_strip, stats_md, dominant_md],
        )

    logger.info("Launching Gradio on http://%s:%s", args.host, args.port)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
