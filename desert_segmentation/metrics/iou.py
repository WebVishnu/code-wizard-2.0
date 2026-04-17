"""Per-class IoU, mIoU, frequency-weighted IoU, confusion matrix."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_confusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Accumulate confusion matrix (pred rows, target columns) — shape CxC."""
    pred = logits.argmax(dim=1).view(-1)
    tgt = target.view(-1)
    valid = tgt != ignore_index
    pred = pred[valid]
    tgt = tgt[valid]
    if pred.numel() == 0:
        return torch.zeros(num_classes, num_classes, dtype=torch.int64, device=logits.device)
    idx = tgt * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def confusion_to_accuracy_metrics(
    cm: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, float | np.ndarray]:
    """Pixel accuracies from confusion ``cm[gt_i, pred_j]`` (same layout as ``IoUMetrics``).

    - **global_pixel_accuracy:** ``trace(cm) / sum(cm)`` — fraction of pixels correct.
    - **mean_class_accuracy:** mean of per-class **recall** ``cm[k,k] / sum_j cm[k,j]`` over
      classes with at least one ground-truth pixel (ignores empty rows).

    Returns ``per_class_recall`` aligned with class index for optional reporting.
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy()
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total <= eps:
        z = np.zeros(cm.shape[0], dtype=np.float64)
        return {
            "global_pixel_accuracy": 0.0,
            "mean_class_accuracy": 0.0,
            "per_class_recall": z,
        }
    trace = np.trace(cm)
    global_acc = float(trace / total)
    row_sums = cm.sum(axis=1)
    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.where(row_sums > eps, diag / np.maximum(row_sums, eps), np.nan)
    present = row_sums > eps
    mean_class_acc = (
        float(np.nanmean(per_class_recall[present])) if np.any(present) else 0.0
    )
    return {
        "global_pixel_accuracy": global_acc,
        "mean_class_accuracy": mean_class_acc,
        "per_class_recall": per_class_recall,
    }


def gt_pixel_counts(cm: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ground-truth pixel counts per class: ``sum_j cm[gt_k, pred_j]`` (row sums)."""
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy()
    cm = np.asarray(cm, dtype=np.float64)
    return np.sum(cm, axis=1).astype(np.int64)


def valid_class_miou_from_confusion(
    cm: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> float:
    """Mean IoU over classes that have at least one ground-truth pixel on the val set.

    Unlike full mIoU (mean over all classes, often many zeros when a class is absent from
    val GT), this only averages **finite** per-class IoU values for rows with ``GT > 0``.
    Returns ``0.0`` if no class has any GT pixels.
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.detach().cpu().numpy()
    cm = np.asarray(cm, dtype=np.float64)
    diag = np.diag(cm)
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    union = rows + cols - diag + eps
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diag / union
    present = rows > 0
    if not np.any(present):
        return 0.0
    finite = present & np.isfinite(iou)
    if not np.any(finite):
        return 0.0
    return float(np.mean(iou[finite]))


def confusion_to_iou(cm: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Returns per-class IoU, mean IoU, frequency-weighted IoU."""
    diag = torch.diag(cm).float()
    rows = cm.sum(dim=1).float()
    cols = cm.sum(dim=0).float()
    union = rows + cols - diag + 1e-6
    iou = diag / union
    miou = iou[torch.isfinite(iou)].mean().item()
    freq = cols / (cols.sum() + 1e-6)
    fw_iou = (iou * freq).sum().item()
    return iou, miou, fw_iou


class IoUMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self) -> None:
        self._cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        logits = logits.to(self.device)
        target = target.to(self.device)
        self._cm += compute_confusion(logits, target, self.num_classes, self.ignore_index).to(self.device)

    def compute(self) -> Dict[str, float | np.ndarray]:
        cm = self._cm.cpu()
        iou, miou, fw_iou = confusion_to_iou(cm)
        acc = confusion_to_accuracy_metrics(cm)
        return {
            "per_class_iou": iou.numpy(),
            "miou": miou,
            "fw_iou": fw_iou,
            "global_pixel_accuracy": acc["global_pixel_accuracy"],
            "mean_class_accuracy": acc["mean_class_accuracy"],
            "per_class_recall": acc["per_class_recall"],
            "confusion": cm.numpy(),
        }
