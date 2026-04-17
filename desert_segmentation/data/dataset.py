"""Image / mask dataset with optional rare-class biased cropping."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from desert_segmentation.data.mask_encoding import RawMaskCodec


def _list_images(dir_path: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        f for f in os.listdir(dir_path) if Path(f).suffix.lower() in exts
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        codec: RawMaskCodec,
        transform: Optional[Callable] = None,
        mode: str = "train",
        crop_size: int = 512,
        rare_class_crop_prob: float = 0.35,
        ignore_index: int = 255,
        seed: int = 42,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.codec = codec
        self.transform = transform
        self.mode = mode
        self.crop_size = crop_size
        self.rare_class_crop_prob = rare_class_crop_prob if mode == "train" else 0.0
        self.ignore_index = ignore_index
        self._rng = random.Random(seed)

        names = _list_images(self.images_dir)
        self._pairs: List[Tuple[str, str]] = []
        for n in names:
            mp = self.masks_dir / n
            if not mp.is_file():
                raise FileNotFoundError(f"Missing mask for {n}: {mp}")
            self._pairs.append((str(self.images_dir / n), str(mp)))

        if not self._pairs:
            raise RuntimeError(f"No images in {self.images_dir}")

    def __len__(self) -> int:
        return len(self._pairs)

    @property
    def image_names(self) -> List[str]:
        """Basenames aligned with dataset indices (for weighted sampling)."""
        return [Path(p[0]).name for p in self._pairs]

    def _load_pair(self, ip: str, mp: str) -> Tuple[np.ndarray, np.ndarray]:
        image = np.array(Image.open(ip).convert("RGB"))
        raw_mask = np.array(Image.open(mp))
        if raw_mask.ndim == 2:
            enc, _ = self.codec.encode_mask(raw_mask.astype(np.uint16))
        else:
            raise ValueError(f"Expected single-channel mask, got shape {raw_mask.shape}")
        return image, enc

    def _random_crop_bias_rare(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        ch, cw = self.crop_size, self.crop_size
        if h < ch or w < cw:
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=self.ignore_index)
            h, w = image.shape[:2]

        if self.mode == "train" and self._rng.random() < self.rare_class_crop_prob:
            hist, _ = np.histogram(mask.flatten(), bins=self.codec.num_classes, range=(0, self.codec.num_classes))
            rare = int(np.argmin(hist))
            ys, xs = np.where(mask == rare)
            if len(xs) > 0:
                idx = self._rng.randrange(len(xs))
                cx, cy = int(xs[idx]), int(ys[idx])
            else:
                cx, cy = w // 2, h // 2
        else:
            cx, cy = self._rng.randrange(w), self._rng.randrange(h)

        x0 = np.clip(cx - cw // 2, 0, w - cw)
        y0 = np.clip(cy - ch // 2, 0, h - ch)
        return image[y0 : y0 + ch, x0 : x0 + cw], mask[y0 : y0 + ch, x0 : x0 + cw]

    def __getitem__(self, idx: int) -> dict:
        ip, mp = self._pairs[idx]
        image, mask = self._load_pair(ip, mp)

        if self.mode == "train":
            image, mask = self._random_crop_bias_rare(image, mask)

        if self.transform is not None:
            t = self.transform(image=image, mask=mask)
            image = t["image"]
            mask = t["mask"]

        if isinstance(mask, torch.Tensor):
            mask_t = mask
        else:
            mask_t = torch.from_numpy(np.asarray(mask))
        if mask_t.dtype in (torch.float32, torch.float16):
            mask_t = (mask_t * 255.0).round().clamp(0, 255).long()
        else:
            mask_t = mask_t.long()

        return {
            "image": image,
            "mask": mask_t,
            "path": ip,
        }
