"""Model registry: short preset names for the backbones we use, and model creation.

Documentation: ``docs/08_TRAINING_AND_RESULTS.md`` (section "Configuration") and ``configs/README.md``.

A *preset* maps a short name (``MODEL=dinov2_base`` in a config file) to a ``timm`` checkpoint
and to settings that suit that backbone on an 8 GB GPU (input size, batch size, gradient
accumulation, ...). Config files and command-line flags still override every preset value.

Any other ``timm`` model name can be used directly (``MODEL=resnet50.a1_in1k``); it then
gets no preset defaults. To add a preset, add one entry to ``MODEL_PRESETS``. Only
general-purpose pretrained weights are allowed by the competition rules (no space-specific
models such as Zoobot; the DINOv3 ``sat493m`` satellite-image checkpoints are not used either).
"""

from dataclasses import dataclass, field

import timm
from torch import nn

from ..config import NUM_CLASSES


@dataclass(frozen=True)
class ModelPreset:
    timm_name: str
    description: str
    defaults: dict = field(default_factory=dict)  # TrainConfig fields suited to this backbone
    tested: bool = False  # True when a run with this preset is reported in docs/08


MODEL_PRESETS: dict[str, ModelPreset] = {
    "convnext_tiny": ModelPreset(
        "convnext_tiny.fb_in22k_ft_in1k", "ConvNeXt-Tiny, ImageNet-22k supervised (28 M parameters)",
        {"image_size": 224, "batch_size": 32}, tested=True),
    "convnext_small": ModelPreset(
        "convnext_small.fb_in22k_ft_in1k", "ConvNeXt-Small, ImageNet-22k supervised (50 M)",
        {"image_size": 288, "batch_size": 24}, tested=True),
    "convnextv2_tiny": ModelPreset(
        "convnextv2_tiny.fcmae_ft_in22k_in1k", "ConvNeXt-V2-Tiny, FCMAE self-supervised + ImageNet-22k (29 M)",
        {"image_size": 288, "batch_size": 32}),
    "efficientnetv2_s": ModelPreset(
        "tf_efficientnetv2_s.in21k_ft_in1k", "EfficientNetV2-S, ImageNet-21k supervised (22 M)",
        {"image_size": 288, "batch_size": 32}),
    "swin_base_384": ModelPreset(
        "swin_base_patch4_window12_384.ms_in22k_ft_in1k", "Swin-Base, ImageNet-22k supervised, 384 px (88 M)",
        {"image_size": 384, "batch_size": 8, "accumulation_steps": 4, "grad_checkpointing": True}),
    "dinov2_base": ModelPreset(
        "vit_base_patch14_reg4_dinov2.lvd142m", "DINOv2 ViT-B/14 with registers, self-supervised LVD-142M (86 M)",
        {"image_size": 336, "batch_size": 8, "accumulation_steps": 4, "grad_checkpointing": True,
         "lr": 2e-4, "layer_decay": 0.75}, tested=True),
    "dinov3_base": ModelPreset(
        "vit_base_patch16_dinov3.lvd1689m", "DINOv3 ViT-B/16, self-supervised LVD-1689M web images, 2025 (86 M)",
        {"image_size": 384, "batch_size": 8, "accumulation_steps": 4, "grad_checkpointing": True,
         "lr": 2e-4, "layer_decay": 0.75}),
    "dinov3_convnext_tiny": ModelPreset(
        "convnext_tiny.dinov3_lvd1689m", "ConvNeXt-Tiny distilled from DINOv3 (LVD-1689M web images), 2025 (28 M)",
        {"image_size": 288, "batch_size": 32}),
    "eva02_base": ModelPreset(
        "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "EVA-02 ViT-B/14, MIM + ImageNet-22k (87 M)",
        {"image_size": 336, "batch_size": 8, "accumulation_steps": 4, "grad_checkpointing": True,
         "lr": 2e-4, "layer_decay": 0.75}),
}

# Backbones with a fixed patch or window grid; timm adapts them when ``img_size`` is passed
# (e.g. DINOv2 at 336 px = 24 x 24 patches of 14 px).
FIXED_SIZE_FAMILIES = ("vit_", "eva", "deit", "beit", "swin")


def resolve_model(name: str) -> tuple[str, str, dict]:
    """Return (timm model name, preset name or "", preset defaults) for a preset or timm name."""
    preset = MODEL_PRESETS.get(name)
    if preset is not None:
        return preset.timm_name, name, dict(preset.defaults)
    return name, "", {}


def create_model(name: str, drop_path_rate: float = 0.1, pretrained: bool = True,
                 image_size: int | None = None) -> nn.Module:
    """A ``timm`` backbone (preset or timm name) with a fresh 10-class head."""
    timm_name, _, _ = resolve_model(name)
    kwargs = {}
    if image_size and timm_name.startswith(FIXED_SIZE_FAMILIES):
        kwargs["img_size"] = image_size
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=NUM_CLASSES,
                             drop_path_rate=drop_path_rate, **kwargs)


def describe_presets() -> str:
    """Human-readable table of the presets (``python -m spaghetti1 list-models``)."""
    lines = [f"{'preset':18s} {'tested':7s} {'timm checkpoint':52s} defaults"]
    for name, preset in MODEL_PRESETS.items():
        defaults = ", ".join(f"{key}={value}" for key, value in preset.defaults.items())
        lines.append(f"{name:18s} {'yes' if preset.tested else 'no':7s} {preset.timm_name:52s} {defaults}")
        lines.append(f"{'':27s}{preset.description}")
    return "\n".join(lines)
