#!/usr/bin/env python3
"""Print segmentation metrics: mIoU (all classes + valid-GT-only), fwIoU, accuracies, GT counts.

Runs a full validation pass by default (same setup as ``scripts/eval.py``). With
``--from-checkpoint-only``, only prints metrics stored inside the checkpoint file
(mIoU and per-class IoU when present); full metrics require a validation forward pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from desert_segmentation.data.dataset import SegmentationDataset
from desert_segmentation.data.mask_encoding import build_codec_from_config
from desert_segmentation.data.transforms import build_val_transforms
from desert_segmentation.metrics.iou import (
    confusion_to_accuracy_metrics,
    gt_pixel_counts,
    valid_class_miou_from_confusion,
)
from desert_segmentation.models.factory import create_model
from desert_segmentation.train.evaluate import evaluate
from desert_segmentation.utils.config import get_paths, load_config
from desert_segmentation.utils.logging_utils import setup_logging
from desert_segmentation.utils.seed import set_seed


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _print_table(rows: List[List[str]]) -> None:
    widths = [max(len(rows[i][c]) for i in range(len(rows))) for c in range(len(rows[0]))]
    for row in rows:
        line = "  ".join(row[c].ljust(widths[c]) for c in range(len(row)))
        print(line)


def run_from_checkpoint_only(ckpt_path: Path) -> int:
    ckpt = _load_checkpoint(ckpt_path, torch.device("cpu"))
    print(f"Checkpoint: {ckpt_path.resolve()}")
    print()
    if "miou" in ckpt:
        print(f"  mIoU (stored):     {float(ckpt['miou']):.6f}")
    else:
        print("  mIoU:              (not stored in this file)")
    names = ckpt.get("class_names")
    per = ckpt.get("per_class_iou")
    if per is not None and names is not None:
        print("  Per-class IoU (stored):")
        for i, name in enumerate(names):
            print(f"    [{i}] {name}: {float(per[i]):.6f}")
    elif per is not None:
        print("  Per-class IoU (stored):")
        for i, v in enumerate(per):
            print(f"    [{i}]: {float(v):.6f}")
    else:
        print("  Per-class IoU:     (not stored in this file)")
    print()
    print(
        "Note: fwIoU, global pixel accuracy, and mean class accuracy are not saved in "
        "checkpoints. Run without --from-checkpoint-only to compute them on the val set."
    )
    return 0


def run_full_eval(args: argparse.Namespace) -> int:
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
    ckpt_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(ckpt_path, device)
    cfg_ck = ckpt["config"]
    model = create_model(cfg_ck["model"], num_classes=codec.num_classes).to(device)
    if ckpt.get("ema") is not None:
        for n, p in model.named_parameters():
            if n in ckpt["ema"]:
                p.data.copy_(ckpt["ema"][n].to(device))
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(
        model,
        val_loader,
        device,
        num_classes=codec.num_classes,
        ignore_index=ignore_index,
        desc="eval_summary",
    )
    cm = metrics["confusion"]
    acc = confusion_to_accuracy_metrics(cm)
    miou_valid = float(valid_class_miou_from_confusion(cm))
    gt_counts = gt_pixel_counts(cm)

    miou = float(metrics["miou"])
    fw_iou = float(metrics["fw_iou"])
    gpa = float(acc["global_pixel_accuracy"])
    mca = float(acc["mean_class_accuracy"])
    per_iou = metrics["per_class_iou"]
    per_rec = acc["per_class_recall"]

    def _rec_str(i: int) -> str:
        v = float(per_rec[i])
        if math.isnan(v):
            return "n/a"
        return f"{v:.6f}"

    print()
    print(f"Checkpoint: {ckpt_path.resolve()}")
    print(f"Val images: {paths['val_images']}")
    print(f"Val samples: {len(val_ds)}")
    print()
    print("  mIoU (all classes):     {:.6f}".format(miou))
    print("  mIoU (classes w/ GT):   {:.6f}".format(miou_valid))
    print("  Frequency-weighted IoU: {:.6f}".format(fw_iou))
    print("  Global pixel accuracy:  {:.6f}".format(gpa))
    print("  Mean class accuracy:    {:.6f}".format(mca))
    print("    (mean of per-class recall over classes with GT pixels)")
    print()
    table: List[List[str]] = [["cls", "name", "IoU", "recall"]]
    for i, name in enumerate(codec.class_names):
        table.append(
            [
                str(i),
                name,
                f"{float(per_iou[i]):.6f}",
                _rec_str(i),
            ]
        )
    _print_table(table)
    print()
    print("  Val GT pixels per class (full val set):")
    for i, name in enumerate(codec.class_names):
        print(f"    [{i}] {name}: {int(gt_counts[i])}")
    print()

    payload = {
        "checkpoint": str(ckpt_path.resolve()),
        "val_dir": str(paths["val_images"]),
        "num_val_samples": len(val_ds),
        "miou": miou,
        "miou_all_classes": miou,
        "miou_valid_gt_classes": miou_valid,
        "fw_iou": fw_iou,
        "global_pixel_accuracy": gpa,
        "mean_class_accuracy": mca,
        "per_class_iou": {codec.class_names[i]: float(per_iou[i]) for i in range(len(codec.class_names))},
        "per_class_recall": {
            codec.class_names[i]: (None if math.isnan(float(per_rec[i])) else float(per_rec[i]))
            for i in range(len(codec.class_names))
        },
        "val_gt_pixel_counts": {codec.class_names[i]: int(gt_counts[i]) for i in range(len(codec.class_names))},
    }

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation metric summary (val set).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: <root>/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "desert_segmentation" / "configs" / "default.yaml"),
    )
    parser.add_argument("--root", type=str, default=None, help="Workspace root (defaults to repo root)")
    parser.add_argument(
        "--from-checkpoint-only",
        action="store_true",
        help="Only print mIoU/per-class IoU stored in the file (no forward pass).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write full metrics JSON (relative to --root unless absolute).",
    )
    args = parser.parse_args()
    root = Path(args.root or ROOT).resolve()
    ck_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "best.pt"

    if args.from_checkpoint_only:
        if not ck_path.is_file():
            print(f"Error: checkpoint not found: {ck_path}", file=sys.stderr)
            sys.exit(1)
        sys.exit(run_from_checkpoint_only(ck_path))

    args.checkpoint = str(ck_path)
    args.root = str(root)
    if not ck_path.is_file():
        print(f"Error: checkpoint not found: {ck_path}", file=sys.stderr)
        sys.exit(1)
    sys.exit(run_full_eval(args))


if __name__ == "__main__":
    main()
