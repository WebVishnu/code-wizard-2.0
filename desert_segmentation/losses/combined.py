"""Segmentation losses: CE, weighted CE, focal, Dice, and combinations."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def _focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    weight: Optional[torch.Tensor],
    ignore_index: int,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    tgt = target.clone()
    valid = tgt != ignore_index
    tgt_clamped = tgt.clone()
    tgt_clamped[~valid] = 0
    log_pt = log_probs.gather(1, tgt_clamped.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, tgt_clamped.unsqueeze(1)).squeeze(1)
    focal = (1 - pt) ** gamma * (-log_pt)
    if weight is not None:
        focal = focal * weight[tgt_clamped]
    focal = focal * valid.float()
    return focal.sum() / (valid.float().sum().clamp_min(1.0))


def _multiclass_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    n, c, _, _ = probs.shape
    tgt = target
    valid = tgt != ignore_index
    dice_losses = []
    for k in range(c):
        pk = probs[:, k]
        tk = (tgt == k).float()
        m = valid.float()   
        pk, tk = pk * m, tk * m
        inter = (pk * tk).sum(dim=(1, 2))
        denom = pk.sum(dim=(1, 2)) + tk.sum(dim=(1, 2)) + eps
        dice = 1.0 - (2.0 * inter + eps) / denom
        dice_losses.append(dice.mean())
    return torch.stack(dice_losses).mean()


class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(num_classes))
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        w = self.class_weights
        if self.mode == "ce":
            loss = _ce(logits, target, None, self.ignore_index, self.label_smoothing)
        elif self.mode == "weighted_ce":
            loss = _ce(logits, target, w, self.ignore_index, self.label_smoothing)
        elif self.mode == "focal_ce":
            loss = _focal_ce(logits, target, self.focal_gamma, w, self.ignore_index)
        elif self.mode == "ce_dice":
            ce = _ce(logits, target, w, self.ignore_index, self.label_smoothing)
            dice = _multiclass_dice(logits, target, self.ignore_index)
            loss = ce + self.dice_weight * dice
        elif self.mode == "focal_ce_dice":
            focal = _focal_ce(logits, target, self.focal_gamma, w, self.ignore_index)
            dice = _multiclass_dice(logits, target, self.ignore_index)
            loss = focal + self.dice_weight * dice
        else:
            raise ValueError(f"Unknown loss mode {self.mode}")
        return loss, {"loss": float(loss.detach().cpu())}


def build_loss(
    loss_cfg: dict,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    ignore_index: int,
) -> CombinedSegLoss:
    mode = loss_cfg.get("name", "ce_dice")
    return CombinedSegLoss(
        mode=mode,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights,
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )


def compute_class_weights_from_freq(
    freq: torch.Tensor,
    cap: float = 15.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Inverse log frequency with mean normalization and per-class cap on max/min ratio."""
    w = 1.0 / torch.log(freq + eps)
    w = w / w.mean()
    ratio = w / w.median()
    ratio = torch.clamp(ratio, max=cap)
    w = ratio * w.median()
    return w
