#!/usr/bin/env python3
"""Train semantic segmentation model from YAML config."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from desert_segmentation.data.dataset import SegmentationDataset
from desert_segmentation.data.mask_encoding import build_codec_from_config
from desert_segmentation.data.transforms import build_train_transforms, build_val_transforms
from desert_segmentation.losses.combined import build_loss, compute_class_weights_from_freq
from desert_segmentation.models.factory import create_model
from desert_segmentation.train.trainer import train
from desert_segmentation.utils.config import get_paths, load_config
from desert_segmentation.utils.freq import estimate_pixel_frequencies, per_image_sampling_weights
from desert_segmentation.utils.logging_utils import setup_logging
from desert_segmentation.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "desert_segmentation" / "configs" / "default.yaml"),
    )
    parser.add_argument("--root", type=str, default=None, help="Workspace root (defaults to repo root)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke tests)")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Stop each epoch after N batches (smoke tests)")
    args = parser.parse_args()

    root = Path(args.root or ROOT).resolve()
    cfg = load_config(args.config, root=root)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    setup_logging()
    set_seed(int(cfg["train"]["seed"]))

    paths = get_paths(cfg)
    raw_ids = cfg["data"]["raw_ids"]
    names = tuple(cfg["data"]["class_names"])
    codec = build_codec_from_config(raw_ids, names)
    ignore_index = int(cfg["data"].get("ignore_index", 255))
    crop_size = int(cfg["data"]["crop_size"])

    train_tf = build_train_transforms(
        crop_size=crop_size,
        strong=bool(cfg.get("augmentation", {}).get("strong", True)),
        ignore_index=ignore_index,
    )
    val_tf = build_val_transforms(crop_size=crop_size, ignore_index=ignore_index)

    train_ds = SegmentationDataset(
        paths["train_images"],
        paths["train_masks"],
        codec=codec,
        transform=train_tf,
        mode="train",
        crop_size=crop_size,
        rare_class_crop_prob=float(cfg["data"].get("rare_class_crop_prob", 0.35)),
        ignore_index=ignore_index,
        seed=int(cfg["train"]["seed"]),
    )
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

    nw = int(cfg["data"].get("num_workers", 4))
    if os.name == "nt":
        nw = 0

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg["model"], num_classes=codec.num_classes).to(device)

    freq = estimate_pixel_frequencies(paths["train_masks"], codec, max_files=None)
    cap = float(cfg.get("loss", {}).get("class_weight_cap", 15.0))
    class_w = compute_class_weights_from_freq(freq, cap=cap).to(device)
    logger.info("class pixel frequencies (train masks): %s", freq.tolist())

    use_weighted_sampler = bool(cfg.get("data", {}).get("weighted_sampler", False))
    sampler: WeightedRandomSampler | None = None
    if use_weighted_sampler:
        eps = float(cfg.get("data", {}).get("weighted_sampler_eps", 1e-6))
        logger.info("computing per-image sampling weights (scanning train masks)...")
        sample_w = per_image_sampling_weights(
            paths["train_masks"],
            train_ds.image_names,
            codec,
            freq,
            eps=eps,
        )
        sampler = WeightedRandomSampler(
            sample_w,
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(int(cfg["train"]["seed"])),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    criterion = build_loss(
        cfg.get("loss", {}),
        num_classes=codec.num_classes,
        class_weights=class_w,
        ignore_index=ignore_index,
    ).to(device)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir

    out = train(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        cfg,
        num_classes=codec.num_classes,
        ignore_index=ignore_index,
        checkpoint_dir=ckpt_dir,
        class_names=codec.class_names,
        max_train_batches=args.max_train_batches,
    )
    logger.info(
        "finished best_score=%s (metric=%s) best_mIoU=%s path=%s",
        out.get("best_score"),
        cfg["train"].get("checkpoint_metric", "miou"),
        out["best_miou"],
        out["best_path"],
    )


if __name__ == "__main__":
    main()
