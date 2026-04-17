"""Optimizer parameter groups (e.g. lower LR on pretrained SegFormer encoder)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn
from torch.optim import AdamW


def build_adamw_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> AdamW:
    """AdamW with reduced LR on `backbone.segformer` (MiT encoder) when present."""
    if backbone_lr_mult >= 0.9999:
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    enc: List[nn.Parameter] = []
    dec: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.segformer"):
            enc.append(p)
        else:
            dec.append(p)

    if not enc:
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    groups: List[Dict[str, Any]] = [
        {"params": enc, "lr": lr * backbone_lr_mult},
        {"params": dec, "lr": lr},
    ]
    return AdamW(groups, weight_decay=weight_decay)
