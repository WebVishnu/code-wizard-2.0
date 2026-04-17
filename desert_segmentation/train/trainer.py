"""Training loop with AMP, cosine+warmup, EMA, best-mIoU checkpointing."""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from desert_segmentation.train.evaluate import evaluate
from desert_segmentation.train.optimizer_utils import build_adamw_groups

logger = logging.getLogger(__name__)


def _checkpoint_score(val_metrics: Dict[str, Any], metric_name: str) -> float:
    key = {
        "miou": "miou",
        "pixel_accuracy": "global_pixel_accuracy",
        "global_pixel_accuracy": "global_pixel_accuracy",
        "mean_class_accuracy": "mean_class_accuracy",
    }.get(metric_name.lower().strip(), "miou")
    return float(val_metrics[key])


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._collect(model)

    @torch.no_grad()
    def _collect(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

def _warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 0.01,
) -> Any:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
    num_classes: int,
    ignore_index: int,
    checkpoint_dir: Path,
    class_names: tuple[str, ...],
    max_train_batches: Optional[int] = None,
) -> Dict[str, Any]:
    tcfg = cfg["train"]
    epochs = int(tcfg["epochs"])
    lr = float(tcfg["lr"])
    wd = float(tcfg["weight_decay"])
    amp_enabled = bool(tcfg.get("amp", True)) and torch.cuda.is_available()
    clip = float(tcfg.get("gradient_clip", 0.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.08))
    patience = int(tcfg.get("early_stop_patience", 20))
    log_interval = int(tcfg.get("log_interval", 20))
    checkpoint_metric = str(tcfg.get("checkpoint_metric", "miou"))
    backbone_lr_mult = float(tcfg.get("backbone_lr_mult", 1.0))

    ema_cfg = cfg.get("ema") or {}
    use_ema = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema: Optional[ModelEMA] = ModelEMA(model, decay=ema_decay) if use_ema else None

    opt = build_adamw_groups(model, lr=lr, weight_decay=wd, backbone_lr_mult=backbone_lr_mult)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = LambdaLR(opt, _warmup_cosine_lambda(total_steps, warmup_steps))
    scaler: Optional[GradScaler] = GradScaler() if amp_enabled else None

    best_score = -1.0
    bad_epochs = 0
    history: list = []

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_log = 0
        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss, _ = criterion(logits, masks)
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            sched.step()
            global_step += 1
            if ema is not None:
                ema.update(model)
            running += float(loss.detach().cpu())
            n_log += 1
            if global_step % log_interval == 0:
                pbar.set_postfix(loss=f"{running / max(n_log, 1):.4f}")
                running = 0.0
                n_log = 0
            if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                break

        backup = copy.deepcopy(model.state_dict())
        if ema is not None:
            ema.copy_to(model)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            desc=f"val {epoch}",
        )
        model.load_state_dict(backup)

        miou = float(val_metrics["miou"])
        gpa = float(val_metrics["global_pixel_accuracy"])
        mca = float(val_metrics["mean_class_accuracy"])
        score = _checkpoint_score(val_metrics, checkpoint_metric)
        row = {
            "epoch": epoch,
            "miou": miou,
            "fw_iou": float(val_metrics["fw_iou"]),
            "global_pixel_accuracy": gpa,
            "mean_class_accuracy": mca,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_score": score,
        }
        history.append(row)
        logger.info(
            "epoch %s | val mIoU=%.4f fwIoU=%.4f pixelAcc=%.4f meanClsAcc=%.4f [%s=%.4f]",
            epoch,
            miou,
            row["fw_iou"],
            gpa,
            mca,
            checkpoint_metric,
            score,
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "optimizer": opt.state_dict(),
                "config": cfg,
                "class_names": class_names,
            },
            last_path,
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            save_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "miou": miou,
                "global_pixel_accuracy": gpa,
                "mean_class_accuracy": mca,
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_score": score,
                "per_class_iou": val_metrics["per_class_iou"].tolist(),
                "config": cfg,
                "class_names": class_names,
            }
            torch.save(save_payload, best_path)
            logger.info(
                "saved new best checkpoint %s=%.4f -> %s",
                checkpoint_metric,
                score,
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stopping at epoch %s (no improvement %s epochs)", epoch, patience)
                break

        with (checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return {
        "best_miou": float(max(h["miou"] for h in history)) if history else -1.0,
        "best_score": best_score,
        "best_path": str(best_path),
        "history": history,
    }