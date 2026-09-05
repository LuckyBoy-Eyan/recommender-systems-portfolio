"""生成式推荐模型的最小训练循环。

scripts/run_demo.py 调用 train_model；train_model 通过 DataLoader 间接调用
NextItemDataset.__getitem__，再调用 SemanticIDTransformer.forward。
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def _device(value: str | torch.device) -> torch.device:
    """选择 CUDA、Apple MPS 或 CPU。"""
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def train_model(
    model,
    dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    *,
    device: str | torch.device = "cpu",
    weight_decay: float = 0.01,
    gradient_clip_norm: float | None = 1.0,
    num_workers: int = 0,
    validation_fn=None,
    monitor_metric: str | None = None,
    early_stopping_patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    mixed_precision: bool = False,
    amp_dtype: str = "float16",
    warmup_epochs: int = 2,
    min_learning_rate: float = 1e-5,
):
    """使用各级 token 交叉熵之和训练一个 SemanticIDTransformer。

    参数:
        model: 要训练的 SemanticIDTransformer。函数会原地更新其参数。
        dataset: NextItemDataset，迭代后返回 history、target_codes、target_item。
        epochs: 完整遍历训练数据的次数。
        batch_size: 每次参数更新使用的样本数量。
        learning_rate: AdamW 优化器学习率。
        device: ``auto``、``cpu``、``mps`` 或 ``cuda``。
        weight_decay: AdamW 权重衰减系数。
        gradient_clip_norm: 梯度裁剪阈值；None 表示不裁剪。
        num_workers: DataLoader 后台进程数。macOS 默认 0 更稳定。
        validation_fn: 可选回调，每轮结束后接收 model 并返回指标字典。
        monitor_metric: 用于选择最佳 epoch 的指标，例如 ``hitrate@20``。
        early_stopping_patience: 验证指标连续多少轮不提升后停止。
        checkpoint_path: 可选检查点路径。每轮完成后原子写入，可恢复长训练。
        resume: 路径存在时是否恢复模型、优化器、早停状态和训练曲线。
        mixed_precision: CUDA 上是否启用自动混合精度。
        amp_dtype: ``float16`` 或 ``bfloat16``；前者同时使用 GradScaler。

    返回:
        训练后的同一个 model 对象，便于链式使用。当前入口虽然不接收返回值，
        但 model 已经被原地修改。

    调用:
        scripts/run_demo.py 对 Semantic ID 模型和 Random ID 模型各调用一次。

    损失:
        ``loss = CE(level_0) + CE(level_1) + ... + CE(tail)``。
        每一级的标签都是目标商品对应位置的 token。
    """
    # DataLoader 会把很多条训练样本拼成一个 batch，方便模型一次训练多条数据。
    runtime_device = _device(device)
    model.to(runtime_device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=runtime_device.type == "cuda",
    )
    # AdamW 是优化器，负责根据 loss 的反向传播结果更新模型参数。
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
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
    amp_enabled = bool(mixed_precision and runtime_device.type == "cuda")
    autocast_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and autocast_dtype == torch.float16
    )
    # 切到训练模式，启用 dropout 等只在训练时生效的模块。
    model.train()
    training_history = []
    best_score = float("-inf")
    best_state = None
    stale_epochs = 0
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
        training_history = saved["training_history"]
        best_score = saved["best_score"]
        best_state = saved["best_state"]
        stale_epochs = saved["stale_epochs"]
        start_epoch = saved["next_epoch"]
        print(f"resumed checkpoint={checkpoint} next_epoch={start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        epoch_loss, sample_count = 0.0, 0
        level_loss_totals = None
        hard_loss_total = 0.0
        for batch in loader:
            if len(batch) == 5:
                history, feedback_types, target_codes, _, _ = batch
            else:
                history, target_codes, _, _ = batch
                feedback_types = None
            # DataLoader 拼接后的形状：
            # history      [batch_size, max_history, num_levels]
            # target_codes [batch_size, num_levels]
            # 后两个返回值是原始 target_item 与 history_items；
            # 训练 loss 不需要它们，所以写成 _。
            # history 是用户历史物品的 Semantic ID 序列。
            # target_codes 是下一个真实物品的 Semantic ID，比如 [c1, c2, tail]。
            # 训练时把 target_codes 传给模型，是为了使用 teacher forcing：
            # 预测下一层 token 时，喂入真实的上一层 token，让训练更稳定。
            history = history.to(runtime_device, non_blocking=True)
            target_codes = target_codes.to(runtime_device, non_blocking=True)
            if feedback_types is not None:
                feedback_types = feedback_types.to(
                    runtime_device, non_blocking=True
                )
            with torch.autocast(
                device_type=runtime_device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits = model(history, target_codes, feedback_types)
                # logits 里面有多组预测结果：
                # logits[0] 预测第 1 级 token，logits[1] 预测第 2 级 token，依此类推。
                # 每一级 Semantic ID 都单独算分类损失，最后相加成一个总 loss。
                level_losses = [
                    nn.functional.cross_entropy(
                        level_logits, target_codes[:, level]
                    )
                    for level, level_logits in enumerate(logits)
                ]
                loss = sum(level_losses)
            # 清空上一轮残留的梯度，避免新旧梯度混在一起。
            optimizer.zero_grad()
            # 根据 loss 反向传播，计算每个模型参数应该往哪个方向调整。
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            # 优化器真正更新参数，让模型下次更相信正确的 Semantic ID token。
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(history)
            if level_loss_totals is None:
                level_loss_totals = [0.0] * len(level_losses)
            for level, level_loss in enumerate(level_losses):
                level_loss_totals[level] += float(level_loss.detach()) * len(history)
            hard_loss_total += float(loss.detach()) * len(history)
            sample_count += len(history)
        mean_loss = epoch_loss / max(sample_count, 1)
        record = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "hard_loss": hard_loss_total / max(sample_count, 1),
            "level_losses": [
                value / max(sample_count, 1)
                for value in (level_loss_totals or [])
            ],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if validation_fn is not None:
            validation_metrics = validation_fn(model)
            record["validation"] = validation_metrics
            metric_name = monitor_metric or next(
                key for key in validation_metrics if key.startswith("hitrate@")
            )
            score = float(validation_metrics[metric_name])
            if score > best_score:
                best_score = score
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
            model.train()
        training_history.append(record)
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
                    "training_history": training_history,
                    "best_score": best_score,
                    "best_state": best_state,
                    "stale_epochs": stale_epochs,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        message = f"epoch={epoch + 1} loss={mean_loss:.6f}"
        if validation_fn is not None:
            message += f" {metric_name}={score:.6f}"
        print(message)
        if (
            validation_fn is not None
            and early_stopping_patience is not None
            and stale_epochs >= early_stopping_patience
        ):
            print(f"early_stopping epoch={epoch + 1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # 不改变旧 API 的返回类型；训练曲线挂在模型上供入口保存。
    model.training_history = training_history
    return model
