"""Load YAML config and resolve paths relative to workspace root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path | str, root: Path | None = None) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if root is None:
        root = Path(cfg.get("root", ".")).resolve()
    else:
        root = Path(root).resolve()
    cfg["root"] = str(root)
    return cfg


def resolve_path(root: Path, *parts: str) -> Path:
    return (root / Path(*parts)).resolve()


def get_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(cfg["root"])
    d = cfg["data"]
    return {
        "train_images": resolve_path(root, d["train_images"]),
        "train_masks": resolve_path(root, d["train_masks"]),
        "val_images": resolve_path(root, d["val_images"]),
        "val_masks": resolve_path(root, d["val_masks"]),
        "test_images": resolve_path(root, d["test_images"]),
    }
