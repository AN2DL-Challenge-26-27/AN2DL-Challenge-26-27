"""Training mechanics: optimizer (optionally with layer-wise LR decay), learning-rate schedule,
one training epoch (optionally with MixUp, gradient accumulation, and weight EMA), and
prediction with 8-way TTA. Model creation is in ``models.py``; settings in ``config.py``.

Documentation: ``docs/08_TRAINING_AND_RESULTS.md``.
"""

import math

import numpy as np
import torch
from torch import nn
from timm.optim import param_groups_layer_decay
from torch.utils.data import DataLoader

from .models import create_model  # noqa: F401  (re-exported for older imports)


def build_optimizer(model: nn.Module, lr: float, head_lr_multiplier: float, weight_decay: float,
                    layer_decay: float = 0.0) -> torch.optim.Optimizer:
    """AdamW without weight decay on norms and biases.

    * ``layer_decay == 0``: the backbone uses ``lr`` and the new head ``lr * head_lr_multiplier``.
    * ``layer_decay > 0``: layer-wise LR decay; the head uses ``lr`` and each earlier block is
      scaled by another factor ``layer_decay`` (timm's grouping), which protects pretrained filters.
    """
    if layer_decay > 0:
        no_decay = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
        param_groups = param_groups_layer_decay(model, weight_decay=weight_decay, layer_decay=layer_decay,
                                                no_weight_decay_list=no_decay)
        for group in param_groups:
            group["lr"] = group["base_lr"] = lr * group.pop("lr_scale", 1.0)
        return torch.optim.AdamW(param_groups)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups: dict[tuple[bool, bool], list] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        key = (id(parameter) in head_ids, parameter.ndim <= 1)
        groups.setdefault(key, []).append(parameter)
    param_groups = [
        {"params": params, "lr": lr * (head_lr_multiplier if is_head else 1.0),
         "weight_decay": 0.0 if no_decay else weight_decay}
        for (is_head, no_decay), params in groups.items()
    ]
    for group in param_groups:
        group["base_lr"] = group["lr"]
    return torch.optim.AdamW(param_groups)


def set_learning_rate(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int,
                      min_ratio: float = 0.01) -> None:
    """Linear warm-up followed by cosine decay to ``min_ratio`` of the base learning rate."""
    if step < warmup_steps:
        factor = (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy against probability targets (MixUp), with optional per-class weights."""
    log_probs = logits.float().log_softmax(1)
    if weight is not None:
        log_probs = log_probs * weight
    return -(targets * log_probs).sum(1).mean()


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, *, step: int, total_steps: int,
                    warmup_steps: int, amp_dtype, mixup=None, class_weight=None, ema=None,
                    accumulation_steps: int = 1) -> tuple[float, float, int]:
    """Train for one pass over ``loader``; return (mean loss, accuracy, next optimizer step).

    ``mixup`` (a ``timm.data.Mixup``) turns the labels into soft targets for this epoch.
    ``accumulation_steps`` > 1 sums gradients over several batches per optimizer step.
    ``ema`` (a ``timm.utils.ModelEmaV3``) is updated after every optimizer step.
    """
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    batches = len(loader)
    for index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = None
        if mixup is not None:
            images, targets = mixup(images, labels)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = criterion(logits, labels) if targets is None else soft_cross_entropy(logits, targets, class_weight)
        (loss / accumulation_steps).backward()
        if (index + 1) % accumulation_steps == 0 or index + 1 == batches:
            set_learning_rate(optimizer, step, total_steps, warmup_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                ema.update(model, step=step)
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return loss_sum / seen, correct / seen, step


def _dihedral_views(images: torch.Tensor):
    """The 8 rotations/reflections of a batch (4 rotations x optional horizontal flip)."""
    for flip in (False, True):
        flipped = images.flip(-1) if flip else images
        for k in range(4):
            yield torch.rot90(flipped, k, dims=(-2, -1))


@torch.no_grad()
def predict(model, loader: DataLoader, device, *, tta: bool = False, amp_dtype=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (softmax probabilities, labels, mean loss on labelled data) for every item in ``loader``."""
    model.eval()
    probabilities, targets, loss_sum, labelled = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if tta:  # average the probabilities of the 8 views; no loss is reported
                probs = torch.stack([model(view).float().softmax(1) for view in _dihedral_views(images)]).mean(0)
            else:
                logits = model(images).float()
                probs = logits.softmax(1)
        mask = labels >= 0  # test images have label -1
        if not tta and mask.any():
            loss_sum += criterion(logits[mask.to(device)], labels[mask].to(device)).item()
            labelled += int(mask.sum())
        probabilities.append(probs.cpu().numpy())
        targets.append(labels.numpy())
    mean_loss = loss_sum / labelled if labelled else float("nan")
    return np.concatenate(probabilities), np.concatenate(targets), mean_loss
