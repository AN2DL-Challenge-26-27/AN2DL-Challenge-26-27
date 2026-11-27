# Experiment configuration files

Every training run is described by one small text file in `.env` format (`KEY=VALUE`). The files are tracked in git, so it is always visible which parameters produced which result. Full explanation: [docs/08 · Training and results](../docs/08_TRAINING_AND_RESULTS.md#configuration-files-and-model-presets).

```text
configs/
├── recipes/
│   └── research_v2.env        # shared training recipe (loss, MixUp, EMA, drop path, ...)
└── experiments/
    ├── _template.env          # copy this to start a new experiment
    ├── e0_..._fold0.env ...   # the six runs reported in docs/08 (re-runnable)
    ├── r1_convnext_tiny_224_recipe_fold0.env
    └── r2_dinov2_base_336_recipe_fold0.env
```

## Use

```bash
python -m spaghetti1 list-models                                   # model presets for MODEL=
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env --dry-run
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env --fold 1 --run-name r2_dinov2_base_336_recipe_fold1
```

`--dry-run` prints every resolved setting and where it came from, without training.

## Rules

* **Keys** are the `TrainConfig` field names in upper case (`IMAGE_SIZE`, `LAYER_DECAY`, ...); see `spaghetti1/training/config.py`. An unknown key (a typo) stops the run with the list of valid keys.
* **`BASE=<path>`** inherits another file (path relative to this file), for example `BASE=../recipes/research_v2.env`. Inheritance can be chained; later files win.
* **`MODEL=`** is a preset (`convnext_tiny`, `convnext_small`, `convnextv2_tiny`, `efficientnetv2_s`, `swin_base_384`, `dinov2_base`, `eva02_base`) or any `timm` model name. A preset also sets suitable defaults for that backbone (input size, batch size, ...).
* **The run is named after the file** unless `RUN_NAME=` is set. Results go to `outputs/runs/<run name>/`.
* **Precedence**, lowest to highest: `TrainConfig` defaults → project `.env` (`SPAGHETTI_NUM_WORKERS`) → model preset → `BASE` file(s) → this file → command-line flags.
* Comments (`#`) are allowed on their own line and after a value.

## Traceability

Each run writes `config.env` (all resolved settings, with the source of each value as a comment) and `config.json` (the same plus a `sources` map) into its output folder. Re-run a finished experiment exactly with `python -m spaghetti1 train --config outputs/runs/<run>/config.env`.

Machine-specific settings (paths, number of workers, where runs are written) belong in the project `.env` (copy `.env.example`), not in experiment files.
