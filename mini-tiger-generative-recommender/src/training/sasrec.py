"""独立 SASRec baseline 的训练循环。"""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.evaluate import _device, evaluate_sasrec


def train_sasrec_baseline(
    model,
    train_dataset,
    validation_dataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ks: list[int],
    monitor: str = "hitrate@20",
    patience: int | None = 2,
    device: str | torch.device = "cpu",
    weight_decay: float = 0.01,
    gradient_clip_norm: float | None = 1.0,
    checkpoint_path: str | Path | None = None,
    exclude_seen: bool = True,
) -> tuple[object, list[dict]]:
    """使用全目录交叉熵训练 SASRec，并按验证 HitRate 早停。"""
    runtime_device = _device(device)
    model = model.to(runtime_device)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=runtime_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    best_score = float("-inf")
    best_state = None
    stale = 0
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

    for epoch in range(epochs):
        model.train()
        total_loss, samples = 0.0, 0
        for history_tokens, targets, _ in loader:
            history_tokens = history_tokens.to(runtime_device)
            targets = targets.to(runtime_device)
            logits = model(history_tokens)
            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            samples += len(targets)

        validation = evaluate_sasrec(
            model,
            validation_dataset,
            ks,
            batch_size=batch_size,
            device=runtime_device,
            exclude_seen=exclude_seen,
        )
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(samples, 1),
            "validation": validation,
        }
        history.append(record)
        score = float(validation[monitor])
        if score > best_score:
            best_score = score
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "best_state": best_state,
                    "best_score": best_score,
                    "training_history": history,
                    "monitor": monitor,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        print(
            f"sasrec_epoch={epoch + 1} loss={record['loss']:.6f} "
            f"{monitor}={score:.6f}",
            flush=True,
        )
        if patience is not None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_masked_sasrec(
    model,
    train_dataset,
    validation_dataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ks: list[int],
    monitor: str = "hitrate@20",
    patience: int | None = 3,
    device: str | torch.device = "cpu",
    weight_decay: float = 0.01,
    gradient_clip_norm: float | None = 1.0,
    checkpoint_path: str | Path | None = None,
    mixed_precision: bool = False,
    resume: bool = False,
    warmup_epochs: int = 2,
    min_learning_rate: float = 1e-5,
) -> tuple[object, list[dict]]:
    """完整序列并行训练，只在正反馈目标位置计算全目录 CE。"""
    runtime_device = _device(device)
    model = model.to(runtime_device)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=runtime_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=runtime_device.type == "cuda",
    )
    total_steps = max(epochs * len(loader), 1)
    warmup_steps = min(warmup_epochs * len(loader), total_steps - 1)
    minimum_ratio = min_learning_rate / learning_rate

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    amp_enabled = bool(mixed_precision and runtime_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history, best_score, best_state, stale = [], float("-inf"), None, 0
    start_epoch = 0
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=runtime_device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if saved.get("scheduler_state"):
            scheduler.load_state_dict(saved["scheduler_state"])
        if saved.get("scaler_state"):
            scaler.load_state_dict(saved["scaler_state"])
        history = saved["training_history"]
        best_score = saved["best_score"]
        best_state = saved["best_state"]
        stale = saved["stale"]
        start_epoch = saved["next_epoch"]
        print(f"resumed masked SASRec at epoch={start_epoch + 1}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, positive_targets = 0.0, 0
        for item_tokens, feedback_types, labels in loader:
            item_tokens = item_tokens.to(runtime_device, non_blocking=True)
            feedback_types = feedback_types.to(runtime_device, non_blocking=True)
            labels = labels.to(runtime_device, non_blocking=True)
            count = int(labels.ne(-100).sum())
            if count == 0:
                continue
            with torch.autocast(
                device_type=runtime_device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                # 完整序列先通过 Transformer；只对有正目标的位置计算全目录
                # logits。与 CE(ignore_index=-100) 数学等价，但避免为负目标和
                # padding 生成昂贵的 [num_items] 分数。
                states = model.encode_sequence(item_tokens, feedback_types)
                positive = labels.ne(-100)
                logits = model.score_all_items(states[positive])
                loss = nn.functional.cross_entropy(
                    logits,
                    labels[positive],
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach()) * count
            positive_targets += count

        validation = evaluate_sasrec(
            model,
            validation_dataset,
            ks,
            batch_size=batch_size,
            device=runtime_device,
            exclude_seen=False,
        )
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / max(positive_targets, 1),
            "positive_targets": positive_targets,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(record)
        score = float(validation[monitor])
        if score > best_score:
            best_score, stale = score, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "next_epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_state": best_state,
                    "best_score": best_score,
                    "training_history": history,
                    "monitor": monitor,
                    "stale": stale,
                    "protocol": "full-events-positive-target-mask-v2",
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        print(
            f"masked_sasrec_epoch={epoch + 1} loss={record['loss']:.6f} "
            f"positive_targets={positive_targets} {monitor}={score:.6f}",
            flush=True,
        )
        if patience is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
