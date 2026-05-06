#!/usr/bin/env python3
"""Hugging Face Spaces entrypoint for the Gradio segmentation demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_gradio import main as demo_main


def _resolve_checkpoint_path(root: Path) -> Path:
    """
    Resolve checkpoint path for deployment.

    Priority:
    1) CHECKPOINT_PATH points to an existing local file.
    2) <root>/checkpoints/best.pt exists.
    3) Download from HF Hub if HF_MODEL_REPO is configured.
    """
    checkpoint_env = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_env:
        candidate = Path(checkpoint_env)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if candidate.is_file():
            return candidate

    local_default = root / "checkpoints" / "best.pt"
    if local_default.is_file():
        return local_default

    model_repo = os.environ.get("HF_MODEL_REPO")
    if not model_repo:
        raise SystemExit(
            "Checkpoint not found. Set CHECKPOINT_PATH, add checkpoints/best.pt, "
            "or configure HF_MODEL_REPO (+ optional HF_MODEL_FILENAME)."
        )

    model_filename = os.environ.get("HF_MODEL_FILENAME", "best.pt")
    downloaded = hf_hub_download(
        repo_id=model_repo,
        filename=model_filename,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(downloaded)


if __name__ == "__main__":
    ckpt = _resolve_checkpoint_path(ROOT)
    os.environ.setdefault("ROOT", str(ROOT))
    os.environ["CHECKPOINT_PATH"] = str(ckpt)

    # Force Space-friendly host/port while preserving demo_gradio behavior.
    sys.argv = [
        "scripts/demo_gradio.py",
        "--root",
        str(ROOT),
        "--checkpoint",
        str(ckpt),
        "--host",
        "0.0.0.0",
        "--port",
        str(int(os.environ.get("PORT", "7860"))),
    ]
    demo_main()
