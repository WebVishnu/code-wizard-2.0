#!/usr/bin/env python3
"""Run validation, print metrics, save confusion matrix and overlays."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from desert_segmentation.data.dataset import SegmentationDataset
from desert_segmentation.data.mask_encoding import build_codec_from_config
from desert_segmentation.data.transforms import build_val_transforms
from desert_segmentation.models.factory import create_model
from desert_segmentation.train.evaluate import evaluate
from desert_segmentation.utils.config import get_paths, load_config
from desert_segmentation.utils.logging_utils import setup_logging
from desert_segmentation.utils.seed import set_seed
from desert_segmentation.utils.viz import palette, save_triplet

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "desert_segmentation" / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    parser.add_argument("--max_viz", type=int, default=24)
    args = parser.parse_args()

    root = Path(args.root or ROOT).resolve()
    cfg = load_config(args.config, root=root)
    setup_logging()
    set_seed(int(cfg["train"]["seed"]))

    paths = get_paths(cfg)
    raw_ids = cfg["data"]["raw_ids"]
    names = tuple(cfg["data"]["class_names"])
    codec = build_codec_from_config(raw_ids, names)
    ignore_index = int(cfg["data"].get("ignore_index", 255))
    crop_size = int(cfg["data"]["crop_size"])

    val_tf = build_val_transforms(crop_size=crop_size, ignore_index=ignore_index)
    val_ds = SegmentationDataset(
        paths["val_images"],
        paths["val_masks"],
        codec=codec,
        transform=val_tf,
        mode="val",
        crop_size=crop_size,
        rare_class_crop_prob=0.0,
        ignore_index=ignore_index,
        seed=int(cfg["train"]["seed"]),
    )

    nw = 0 if os.name == "nt" else int(cfg["data"].get("num_workers", 4))
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(Path(args.checkpoint), map_location=device)
    cfg_ck = ckpt["config"]
    model = create_model(cfg_ck["model"], num_classes=codec.num_classes).to(device)
    if ckpt.get("ema") is not None:
        for n, p in model.named_parameters():
            if n in ckpt["ema"]:
                p.data.copy_(ckpt["ema"][n].to(device))
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(model, val_loader, device, num_classes=codec.num_classes, ignore_index=ignore_index)
    logger.info(
        "mIoU=%.4f fwIoU=%.4f pixelAcc=%.4f meanClsAcc=%.4f",
        metrics["miou"],
        metrics["fw_iou"],
        metrics["global_pixel_accuracy"],
        metrics["mean_class_accuracy"],
    )
    per = metrics["per_class_iou"]
    for i, name in enumerate(codec.class_names):
        logger.info("  %s IoU=%.4f", name, float(per[i]))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "miou": float(metrics["miou"]),
                "fw_iou": float(metrics["fw_iou"]),
                "global_pixel_accuracy": float(metrics["global_pixel_accuracy"]),
                "mean_class_accuracy": float(metrics["mean_class_accuracy"]),
                "per_class_iou": {codec.class_names[i]: float(per[i]) for i in range(len(codec.class_names))},
            },
            f,
            indent=2,
        )
    np.save(out_dir / "confusion.npy", metrics["confusion"])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    colors = palette(codec.num_classes)
    n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="viz"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().numpy()
            gt = masks.cpu().numpy()
            for b in range(images.shape[0]):
                if n >= args.max_viz:
                    break
                t = images[b].cpu().permute(1, 2, 0).numpy()
                rgb = (t * std + mean) * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                save_triplet(
                    out_dir / f"val_{n:04d}.png",
                    rgb,
                    gt[b],
                    pred[b],
                    colors,
                )
                n += 1
            if n >= args.max_viz:
                break


if __name__ == "__main__":
    main()
