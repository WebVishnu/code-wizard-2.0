"""Build segmentation models: SMP backbones or HuggingFace SegFormer."""

from __future__ import annotations

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerLogitsWrapper(nn.Module):
    """HF SegFormer returns low-res logits; upsample to match input for CE/Dice/IoU."""

    def __init__(self, backbone: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values, return_dict=True)
        logits = out.logits
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


# ADE20K NVIDIA checkpoints (Transformers). Use 512x512 to match common crop sizes; 640 variants also exist on the Hub.
_DEFAULT_SEGFORMER_PRETRAINED: Dict[str, str] = {
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b5": "nvidia/segformer-b5-finetuned-ade-512-512",
}


def _create_segformer(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    arch = (model_cfg.get("architecture") or "").lower()
    pretrained_id = model_cfg.get("pretrained_id")
    if not pretrained_id:
        if arch == "segformer":
            raise ValueError(
                'model.pretrained_id is required when architecture is "segformer".'
            )
        if arch not in _DEFAULT_SEGFORMER_PRETRAINED:
            raise ValueError(
                f"Unknown SegFormer architecture {arch!r}; set model.pretrained_id in config."
            )
        pretrained_id = _DEFAULT_SEGFORMER_PRETRAINED[arch]
    ignore_idx = int(model_cfg.get("semantic_loss_ignore_index", 255))
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    hf_model.config.semantic_loss_ignore_index = ignore_idx
    hf_model.config.id2label = {i: str(i) for i in range(num_classes)}
    hf_model.config.label2id = {str(i): i for i in range(num_classes)}
    return SegFormerLogitsWrapper(hf_model)


def create_model(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    arch = (model_cfg.get("architecture") or "deeplabv3plus").lower()
    encoder_name = model_cfg.get("encoder_name", "resnet50")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")

    if arch in _DEFAULT_SEGFORMER_PRETRAINED or arch == "segformer":
        return _create_segformer(model_cfg, num_classes)

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
