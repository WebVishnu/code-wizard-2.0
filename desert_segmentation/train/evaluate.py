"""Validation loop and metric aggregation."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from desert_segmentation.metrics.iou import IoUMetrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    desc: str = "val",
) -> dict:
    model.eval()
    metrics = IoUMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device)
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()
