"""Albumentations pipelines for images and class masks."""

from __future__ import annotations

from typing import Any, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _base_normalize() -> A.Normalize:
    return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def build_train_transforms(
    crop_size: int,
    strong: bool = True,
    ignore_index: int = 255,
) -> A.Compose:
    """Spatial crops are applied in `SegmentationDataset` (with rare-class bias)."""
    del crop_size
    geometric: list[Any] = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.12,
            rotate_limit=10,
            border_mode=0,
            mask_value=ignore_index,
            p=0.55,
        ),
    ]
    color: list[Any] = [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.55),
        A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=22, val_shift_limit=14, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.22),
        A.GaussNoise(var_limit=(8.0, 48.0), p=0.25),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.25),
        A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=0.28),
    ]
    if strong:
        color.extend(
            [
                A.RandomSunFlare(
                    flare_roi=(0.45, 0.0, 1.0, 0.42),
                    angle_lower=0.4,
                    p=0.12,
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.42, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    p=0.16,
                ),
            ]
        )
    return A.Compose(
        geometric + color + [_base_normalize(), ToTensorV2()],
        additional_targets={"mask": "mask"},
    )


def build_val_transforms(
    crop_size: int,
    ignore_index: int = 255,
) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=crop_size),
            A.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                border_mode=0,
                value=0,
                mask_value=ignore_index,
            ),
            _base_normalize(),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def apply_transform(
    transform: A.Compose,
    image,
    mask,
) -> Tuple[Any, Any]:
    out = transform(image=image, mask=mask)
    return out["image"], out["mask"]
