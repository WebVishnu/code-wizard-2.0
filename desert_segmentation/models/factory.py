"""Build segmentation models via segmentation_models_pytorch."""

from __future__ import annotations

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    arch = (model_cfg.get("architecture") or "deeplabv3plus").lower()
    encoder_name = model_cfg.get("encoder_name", "resnet50")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")

    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    if arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"Unknown architecture: {arch}")
