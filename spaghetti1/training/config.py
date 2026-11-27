"""Training settings: one dataclass, experiment config files in ``.env`` format, and traceability.

Documentation: ``configs/README.md`` and ``docs/08_TRAINING_AND_RESULTS.md`` (section "Configuration").

Where a setting comes from (lowest to highest priority):

1. the defaults of ``TrainConfig`` below;
2. machine settings from the project ``.env`` (only ``SPAGHETTI_NUM_WORKERS``);
3. the model preset chosen with ``MODEL=<preset>`` (see ``models.py``);
4. the experiment config file(s): ``configs/experiments/<name>.env``, which may inherit from
   another file with ``BASE=<relative path>``;
5. command-line flags (``--epochs 5``).

Keys in config files are the field names in upper case (``IMAGE_SIZE=336``). Unknown keys stop
the run with an error, so typos never pass silently. Each run saves the fully resolved settings
as ``config.env`` (re-runnable with ``--config``) and ``config.json`` (with the source of every
value), so it is always clear which parameters produced a result.
"""

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path

from dotenv import dotenv_values

from ..config import PROJECT_DIR
from .models import resolve_model


@dataclass
class TrainConfig:
    """All settings of one experiment. Defaults reproduce the first baseline (E0)."""

    # ---- identity
    run_name: str = "e0_convnext_tiny_fold0"
    description: str = ""            # free text, shown in config.env
    model: str = "convnext_tiny"     # preset name (models.py) or any timm model name
    preset: str = ""                 # filled in automatically when ``model`` is a preset
    fold: int = 0                    # validation fold 0-4; -1 trains on all kept images
    seed: int = 42
    # ---- optimisation
    epochs: int = 25
    batch_size: int = 32
    accumulation_steps: int = 1      # batches per optimizer step (effective batch = batch_size x this)
    lr: float = 1e-4                 # backbone LR; with layer_decay > 0 it is the head's LR
    head_lr_multiplier: float = 10.0  # head LR = lr x this (only when layer_decay == 0)
    layer_decay: float = 0.0         # > 0: layer-wise LR decay (each earlier block x this)
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    drop_path_rate: float = 0.1
    grad_checkpointing: bool = False  # trade compute for memory (large ViTs on 8 GB)
    ema_decay: float = 0.0           # > 0: evaluate and save an exponential moving average of the weights
    # ---- loss and labels
    label_smoothing: float = 0.1
    class_weight_power: float = 0.5  # 0 = unweighted loss, 0.5 = 1/sqrt(n_c), 1 = inverse frequency
    label_policy: str = "original"   # original | drop_copies | as_given (docs/03)
    balance_power: float = 0.0       # > 0: over-sample rare classes instead
    mixup_alpha: float = 0.0         # > 0: MixUp (no CutMix) with mixing weights from Beta(alpha, alpha)
    mixup_prob: float = 0.5          # share of batches that are mixed
    mixup_off_epochs: int = 5        # train the last N epochs without MixUp
    # ---- data
    image_size: int = 224
    augment: bool = True
    background_normalize: bool = False
    denoise: str = "none"            # none | bilateral | nlm
    num_workers: int = 4
    precision: str = "auto"          # auto (bf16 if the GPU supports it, else fp16) | bf16 | fp16 | fp32
    # ---- evaluation and outputs
    tta: bool = True                 # also evaluate validation with 8-way TTA
    predict_test: bool = False
    test_tta: bool = True            # use TTA for the test predictions (when tta is also on)
    resume: bool = False             # continue from outputs/runs/<run_name>/last.pt if it exists
    limit_batches: int = 0           # > 0: only this many batches per epoch (quick smoke tests)


FIELD_TYPES = {item.name: item.type for item in fields(TrainConfig)}
TRUE_WORDS, FALSE_WORDS = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


def parse_value(name: str, text) -> object:
    """Convert a config-file string to the type of ``TrainConfig.<name>``."""
    kind = FIELD_TYPES[name]
    if not isinstance(text, str):
        return kind(text)
    text = text.strip()
    if kind is bool:
        if text.lower() in TRUE_WORDS:
            return True
        if text.lower() in FALSE_WORDS:
            return False
        raise ValueError(f"{name.upper()}: expected true/false, got {text!r}")
    return kind(text)


def read_config_file(path: Path, _seen: tuple = ()) -> list[tuple[str, object, str]]:
    """Read a config file (and its ``BASE`` chain) as ordered (field, value, source) entries."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"BASE loop: {' -> '.join(str(p) for p in (*_seen, path))}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = dotenv_values(path, encoding="utf-8-sig", interpolate=False)
    entries: list[tuple[str, object, str]] = []
    base = raw.pop("BASE", None)
    if base:
        entries += read_config_file(path.parent / base, (*_seen, path))
    source = _relative(path)
    unknown = [key for key in raw if key.lower() not in FIELD_TYPES or key.lower() == "preset"]
    if unknown:
        valid = ", ".join(sorted(name.upper() for name in FIELD_TYPES if name != "preset"))
        raise ValueError(f"{source}: unknown key(s) {', '.join(unknown)}. Valid keys: BASE, {valid}")
    for key, value in raw.items():
        entries.append((key.lower(), parse_value(key.lower(), value), source))
    return entries


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_config(config_file: Path | None = None, overrides: dict | None = None) -> tuple[TrainConfig, dict]:
    """Resolve the settings of a run; return (config, source of every field)."""
    values = asdict(TrainConfig())
    sources = {name: "default" for name in values}

    def apply(name: str, value, source: str) -> None:
        values[name] = value
        sources[name] = source

    workers = os.getenv("SPAGHETTI_NUM_WORKERS")
    if workers:
        apply("num_workers", int(workers), ".env (SPAGHETTI_NUM_WORKERS)")

    file_entries = read_config_file(config_file) if config_file else []
    overrides = {key: parse_value(key, value) for key, value in (overrides or {}).items()}
    model_name = overrides.get("model") or next(
        (value for name, value, _ in reversed(file_entries) if name == "model"), values["model"])
    timm_name, preset, preset_defaults = resolve_model(model_name)
    for name, value in preset_defaults.items():
        apply(name, value, f"preset {preset}")
    for name, value, source in file_entries:
        apply(name, value, source)
    for name, value in overrides.items():
        apply(name, value, "command line")
    # Store the exact timm checkpoint, plus the preset it came from.
    values["model"], values["preset"] = timm_name, preset
    sources["preset"] = sources["model"]
    if config_file and sources["run_name"] == "default":  # name runs after their config file
        apply("run_name", Path(config_file).stem, "config file name")
    return TrainConfig(**values), sources


def format_config(config: TrainConfig, sources: dict) -> str:
    """The resolved settings as a re-runnable ``.env`` file, with the source of every value."""
    lines = ["# Resolved training settings (written by spaghetti1). Re-run with:",
             f"#   python -m spaghetti1 train --config <this file>", ""]
    for name, value in asdict(config).items():
        if name == "preset":
            continue
        written = config.preset if name == "model" and config.preset else value
        if isinstance(written, bool):
            written = str(written).lower()
        lines.append(f"{name.upper()}={written}  # {sources.get(name, 'default')}")
    return "\n".join(lines) + "\n"
