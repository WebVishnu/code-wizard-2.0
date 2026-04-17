"""Sliding-window inference with optional horizontal-flip TTA and ONNX export helper."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from desert_segmentation.data.mask_encoding import RawMaskCodec, build_codec_from_config
from desert_segmentation.models.factory import create_model
from desert_segmentation.utils.viz import blend_overlay, colorize_mask, palette, save_triplet

logger = logging.getLogger(__name__)


def _gaussian_2d(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sig = min(h, w) / 3.0
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig**2)))
    return g.astype(np.float32)


def _preprocess(
    rgb: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    # Keep float32 end-to-end: np.array(mean) defaults to float64 and would upcast x → conv2d dtype mismatch.
    x = rgb.astype(np.float32, copy=False) / 255.0
    m = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    s = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - m) / s
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    return t.float()


@torch.no_grad()
def _forward_logits(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    tta_flip: bool,
) -> torch.Tensor:
    logits = model(x)
    if not tta_flip:
        return logits
    xf = torch.flip(x, dims=[3])
    lf = model(xf)
    lf = torch.flip(lf, dims=[3])
    return (logits + lf) * 0.5


def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    last_pos = length - tile
    starts = list(range(0, last_pos + 1, stride))
    if not starts:
        return [0]
    if starts[-1] != last_pos:
        starts.append(last_pos)
    return sorted(set(starts))


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: float,
    tta_flip: bool,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Returns HxW int class map."""
    h, w = image_np.shape[:2]
    if h <= tile_size and w <= tile_size:
        t = _preprocess(image_np, mean, std).to(device)
        logits = _forward_logits(model, t, device, tta_flip)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    stride = max(1, int(tile_size * (1.0 - overlap)))
    g = _gaussian_2d(tile_size, tile_size)

    n_ty = len(_tile_starts(h, tile_size, stride))
    n_tx = len(_tile_starts(w, tile_size, stride))
    H_pad = (n_ty - 1) * stride + tile_size
    W_pad = (n_tx - 1) * stride + tile_size
    pad_h = max(0, H_pad - h)
    pad_w = max(0, W_pad - w)
    img_p = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W = img_p.shape[:2]

    t0 = _preprocess(img_p[0:tile_size, 0:tile_size], mean, std).to(device)
    logits0 = _forward_logits(model, t0, device, tta_flip)
    num_classes = int(logits0.shape[1])
    acc = np.zeros((num_classes, H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for y in _tile_starts(H, tile_size, stride):
        for x in _tile_starts(W, tile_size, stride):
            tile = img_p[y : y + tile_size, x : x + tile_size]
            t = _preprocess(tile, mean, std).to(device)
            logits = _forward_logits(model, t, device, tta_flip)
            probs = torch.softmax(logits, dim=1)
            ls = probs.squeeze(0).cpu().numpy()
            acc[:, y : y + tile_size, x : x + tile_size] += ls * g
            weight[y : y + tile_size, x : x + tile_size] += g

    pred = np.argmax(acc, axis=0).astype(np.int64)
    return pred[:h, :w]


def _load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict, RawMaskCodec]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    raw_ids = cfg["data"]["raw_ids"]
    names = ckpt.get("class_names") or tuple(cfg["data"].get("class_names") or ())
    if not names:
        names = tuple(str(x) for x in raw_ids)
    codec = build_codec_from_config(raw_ids, names)
    model = create_model(cfg["model"], num_classes=codec.num_classes).to(device)
    if ckpt.get("model") is not None:
        model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        for n, p in model.named_parameters():
            if n in ckpt["ema"]:
                p.data.copy_(ckpt["ema"][n].to(device))
    model.eval()
    return model, cfg, codec


@torch.no_grad()
def predict_folder(
    checkpoint_path: Path,
    image_dir: Path,
    out_dir: Path,
    device: Optional[torch.device] = None,
    limit: Optional[int] = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, codec = _load_model_for_inference(checkpoint_path, device)
    icfg = cfg.get("inference") or {}
    tile_size = int(icfg.get("tile_size", 512))
    overlap = float(icfg.get("overlap", 0.25))
    tta = bool(icfg.get("tta_flip", True))

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = palette(codec.num_classes)
    names = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if limit is not None:
        names = names[:limit]
    times: List[float] = []
    for name in tqdm(names, desc="infer"):
        ip = image_dir / name
        rgb = np.array(Image.open(ip).convert("RGB"))
        t0 = time.perf_counter()
        pred = predict_image(model, rgb, device, tile_size, overlap, tta)
        times.append(time.perf_counter() - t0)
        overlay = blend_overlay(rgb, colorize_mask(pred, colors))
        Image.fromarray(overlay).save(out_dir / f"pred_{name}")
        save_triplet(out_dir / f"triplet_{name}", rgb, None, pred, colors)

    if times:
        mean_ms = float(np.mean(times) * 1000.0)
        logger.info("mean inference time: %.2f ms (device=%s)", mean_ms, device)
        with (out_dir / "latency.txt").open("w", encoding="utf-8") as f:
            f.write(f"mean_ms_per_image={mean_ms:.4f}\n")
            f.write(f"device={device}\n")


def export_onnx(
    checkpoint_path: Path,
    out_onnx: Path,
    height: int = 512,
    width: int = 512,
    opset: int = 17,
) -> None:
    device = torch.device("cpu")
    model, _, _ = _load_model_for_inference(checkpoint_path, device)
    model.eval()
    dummy = torch.randn(1, 3, height, width, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_onnx),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "h", 3: "w"},
        },
    )
