# 10 · Research review and the next experiments

Previous: [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)

After the first experiments (see [08](08_TRAINING_AND_RESULTS.md)) we asked an external deep-research tool what architectures, recipes, and combinations work best for this task. We gave it our models, scores, weak classes, and constraints (general-purpose pretrained weights only, 8 GB laptop GPU or Colab). The full answer is kept in [research/2026-11-27_deep_research_galaxy10.md](research/2026-11-27_deep_research_galaxy10.md). This page records what we take from it and in which order we test it.

## What the report says, in one paragraph

Single supervised ImageNet ConvNets plateau around 0.845 on this kind of data, and our weak classes (irregular, tight spiral, loose spiral) are the known hard classes of Galaxy10 DECaLS. The largest remaining gains are expected from four sources:

* **a better training recipe:** no class weights, MixUp instead of label smoothing, EMA weights, layer-wise learning-rate decay, more stochastic depth;
* **diverse model families,** especially patch-14 vision transformers at 336–448 px (DINOv2, EVA-02), whose errors differ from ConvNeXt's;
* **5-fold training,** for reliable scores and fold ensembles;
* **soft pseudo-labelling of the 14,736 test images,** which contain most of the unique galaxies.

It estimates a realistic ceiling of about 0.90–0.91 without astronomy-specific pretraining.

## Checked against our own data

| Claim in the report | Our evidence | Verdict |
| --- | --- | --- |
| The test set holds most of the unique galaxies | 2,999 unique training skies (originals and unique images) + 14,736 test images = 17,735, the size of Galaxy10 DECaLS (17,736) | **Consistent**; makes pseudo-labelling attractive |
| 599 validation images give about ±1.5 points of noise | Same estimate as ours (08) | **Confirmed** |
| The same-family ConvNeXt models make the same mistakes | 75 of 107 errors shared (08) | **Confirmed** |
| Irregular, tight spiral, and loose spiral are the hard classes | Our F1: 0.35–0.60, 0.73–0.80, 0.67–0.72 | **Confirmed** |
| Avoid CutMix and aggressive crops | We never used CutMix; our scale jitter is 0.9–1.1 with ≤ 4% shift | Already in place |
| `1/√n` class weights hurt weighted F1 | Plausible, not measured | **Test (R1)** |
| EMA, MixUp, layer decay, drop path fix the over-fitting | ConvNeXt-Small reached 0.97 training accuracy | **Test (R1)** |
| DINOv2 at 336 px adds the most diversity | Not measured | **Test (R2)** |
| Published Galaxy10 scores quoted in the report | Not verified by us | Treated as indications only |

## Chosen approach and why

We follow the report's own "quick sanity check" first. It is the cheapest test of the two largest claims (recipe and model diversity) and is decided on fold 0 before investing in 5-fold training.

| Run | What changes | Question it answers | Compared with |
| --- | --- | --- | --- |
| **R1** | ConvNeXt-Tiny 224 with the new recipe: no class weights, no label smoothing, MixUp α 0.3 (off in the last 5 epochs), EMA 0.998, drop path 0.25, layer decay 0.85, head learning rate 3e-4, 30 epochs | Does the recipe alone beat the old one? | E1a (0.843, same model and policy) |
| **R2** | DINOv2-Base (`vit_base_patch14_reg4_dinov2.lvd142m`) at 336 px with the same recipe, layer decay 0.75, head learning rate 2e-4, batch 8 × 4 accumulation, gradient checkpointing, 25 epochs | Does a different family score as well and make different errors? | E2b/E3 (0.847) and the 3-model ensemble (0.856) |

Both runs use the `drop_copies` label policy so that they compare directly with E1a–E3.

**Layer-decay values.** timm splits ConvNeXt-Tiny into 20 learning-rate levels and ViT-B into 14. The report's suggested 0.75 for ConvNeXt would leave the first layers at 0.4% of the head's learning rate, so R1 uses 0.85 (lowest level about 4.5%). R2 uses 0.75 (lowest level about 2.4%).

**Success criteria.**

* R1 is adopted as the new default recipe if it is at least as good as E1a (0.843) and its training accuracy stays clearly below 0.97.
* R2 is worth scaling to 5 folds if it reaches about 0.84 or better on its own, **and** either it shares clearly fewer errors with the ConvNeXt runs (well below 70%) or adding it to the ensemble raises the fold-0 ensemble above 0.856.

## Later steps (only after R1/R2)

| Order | Step | Condition |
| ---: | --- | --- |
| 1 | 5-fold training of the best recipe for 2–3 families (ConvNeXt, DINOv2, and EVA-02 or Swin), so that scores use about 3,000 out-of-fold images | R1/R2 succeed |
| 2 | Temperature scaling, out-of-fold ensemble weights, regularised per-class multipliers for weighted F1 | 5-fold predictions available |
| 3 | One round of soft pseudo-labelling (class-stratified, temperature-sharpened) with a leakage-free fold-0 check | Diverse 5-fold teacher available |
| 4 | Polar-coordinate auxiliary model, `asinh` stretch, symmetric cross-entropy | If time remains |

Not planned: training rotation-equivariant networks from scratch (no pretrained weights) and astronomy-pretrained models (forbidden by the rules).

## How to run R1 and R2

The two runs are config files that inherit the shared recipe:

* [configs/recipes/research_v2.env](../configs/recipes/research_v2.env): the recipe;
* [configs/experiments/r1_convnext_tiny_224_recipe_fold0.env](../configs/experiments/r1_convnext_tiny_224_recipe_fold0.env);
* [configs/experiments/r2_dinov2_base_336_recipe_fold0.env](../configs/experiments/r2_dinov2_base_336_recipe_fold0.env), which uses the `dinov2_base` preset for size, batch, and learning rates.

[experiments/research_round1.ps1](../experiments/research_round1.ps1) runs both. That takes about 40–60 minutes on the laptop GPU:

```powershell
powershell -ExecutionPolicy Bypass -File experiments/research_round1.ps1
python -m spaghetti1 train --config configs/experiments/r2_dinov2_base_336_recipe_fold0.env --dry-run   # inspect settings
```

All options were tested on the CPU (2026-11-27) for both models, including DINOv2 at 336 px: layer-decay groups, MixUp soft targets, gradient accumulation, EMA updates, gradient checkpointing, prediction, and resuming. An end-to-end R1 smoke run with `--limit-batches 2` also passed. The GPU runs themselves are pending: Windows blocked GPU access after the previous training session, and a restart is needed.

## Status

| Run | Status | Weighted F1 | + TTA |
| --- | --- | ---: | ---: |
| R1 | Waiting for the GPU | | |
| R2 | Waiting for the GPU | | |
