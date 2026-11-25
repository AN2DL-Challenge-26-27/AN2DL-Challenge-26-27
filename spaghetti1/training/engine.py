"""Model creation, one training epoch, and prediction with optional 8-way TTA.

Documentation: ``docs/08_TRAINING_AND_RESULTS.md``.
"""

import math

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import NUM_CLASSES


def create_model(name: str, drop_path_rate: float = 0.1, pretrained: bool = True) -> nn.Module:
    """A ``timm`` backbone with a fresh 10-class head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES, drop_path_rate=drop_path_rate)


def build_optimizer(model: nn.Module, lr: float, head_lr_multiplier: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with a higher learning rate for the new head; no weight decay on norms and biases."""
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


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, *, step: int, total_steps: int,
                    warmup_steps: int, amp_dtype) -> tuple[float, float, int]:
    """Train for one pass over ``loader``; return (mean loss, accuracy, next global step)."""
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        set_learning_rate(optimizer, step, total_steps, warmup_steps)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
        step += 1
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
