#!/usr/bin/env python3
"""Run inference on testing/Color_Images; optional ONNX export."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desert_segmentation.infer.predict import export_onnx, predict_folder
from desert_segmentation.utils.config import get_paths, load_config
from desert_segmentation.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "desert_segmentation" / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="infer_outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--onnx", type=str, default=None, help="If set, export ONNX to this path and exit")
    args = parser.parse_args()

    root = Path(args.root or ROOT).resolve()
    cfg = load_config(args.config, root=root)
    setup_logging()

    if args.onnx:
        export_onnx(Path(args.checkpoint), Path(args.onnx))
        logger.info("exported ONNX to %s", args.onnx)
        return

    paths = get_paths(cfg)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    predict_folder(Path(args.checkpoint), paths["test_images"], out_dir, limit=args.limit)


if __name__ == "__main__":
    main()
