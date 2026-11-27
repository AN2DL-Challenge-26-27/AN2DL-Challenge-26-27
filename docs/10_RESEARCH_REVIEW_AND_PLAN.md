# 10 · Research review and the next experiments

Previous: [09 · Model weights and reproduction](09_MODEL_WEIGHTS_AND_REPRODUCTION.md) · Next: [11 · Training in the cloud](11_CLOUD_TRAINING.md)

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

## Results of round 1 (2026-11-27, fold 0, 599 validation images)

| Run | Weighted F1 | + TTA | Irregular (0) | Tight (6) | Loose (7) | Final train acc. | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1a, old recipe (reference) | 0.843 | 0.844 | 0.51 | 0.78 | 0.70 | 0.92 | 8 min |
| **R1**, ConvNeXt-Tiny 224, research recipe | 0.831 | 0.828 | 0.39 | 0.75 | 0.70 | 0.86 | 11 min |
| **R2**, DINOv2-Base 336, research recipe | **0.852** | 0.839 | 0.38 | **0.81** | 0.72 | 0.86 | 70 min |

Times are with the GPU temperature guard (pauses at 82 °C, see below), which roughly doubles the wall time.

| Ensemble (average of TTA probabilities) | Weighted F1 |
| --- | ---: |
| Earlier best: Tiny-224 + Tiny-384 + Small-288 | 0.856 |
| Tiny-384 + DINOv2 | 0.857 |
| Tiny-384 + Small-288 + DINOv2 | 0.858 |
| **Tiny-224 + Tiny-384 + Small-288 + DINOv2** | **0.859** |

Share of errors also made by the other model, as a fraction of the smaller error set:

| Pair | Shared |
| --- | ---: |
| ConvNeXt-Tiny-384 vs ConvNeXt-Small-288 | 82% |
| DINOv2 vs each ConvNeXt run | 74–77% |
| R1 vs R2 | 85% |

### Verdict against the success criteria

* **R1 is not adopted.** The research recipe reduced over-fitting (training accuracy 0.86 instead of 0.92) but scored 1.2 points lower than the old recipe on ConvNeXt-Tiny. That difference is within noise, so it is not a clear loss either. The old recipe stays the default for ConvNeXt.
* **R2 passes.** It is the best single model so far (0.852). Its errors overlap less with ConvNeXt (74–77%) than ConvNeXt models overlap with each other (82%), though not "well below 70%". Adding it raises the fold-0 ensemble from 0.856 to 0.859. That gain is small (within the ±1.5-point noise), but it is consistent across combinations.
* **TTA remains unreliable:** it lowered R2 from 0.852 to 0.839, as it lowered Tiny-288 before.
* **The irregular class is still the weak point** (0.38–0.59). DINOv2 clearly improved tight spirals (0.81, the best so far) but not irregular galaxies. The 4-model ensemble reaches 0.59 on irregular galaxies.

### Recommended next steps

1. A 4-model submission (Tiny-224 + Tiny-384 + Small-288 + DINOv2). It needs DINOv2 test predictions: about 40 minutes with TTA under the temperature guard, or about 5 minutes without TTA.
2. 5-fold training of DINOv2 and ConvNeXt-Tiny-384, about 8 hours with the guard (overnight). It gives reliable out-of-fold scores on ~3,000 images and fold ensembles, which matter more than any further fold-0 comparison.
3. Then calibration and per-class multipliers on the out-of-fold predictions, and one round of soft pseudo-labelling (see "Later steps").

## GPU temperature guard

`experiments/gpu_guard.py` runs next to any training. It pauses the `spaghetti1` processes at 82 °C and resumes them at 72 °C after at least 45 s. During this round it paused about 75 times, and the GPU never exceeded 84 °C. Without it, the GPU sat at 85–86 °C. If the guard is killed during a pause, `python experiments/gpu_guard.py --resume-all` resumes the training.

## Status

| Run | Status | Weighted F1 | + TTA |
| --- | --- | ---: | ---: |
| R1 | Done | 0.831 | 0.828 |
| R2 | Done | 0.852 | 0.839 |

## Leaderboard round 1 (2026-11-27)

The first leaderboard score (3-model ConvNeXt ensemble: 0.8371, validation 0.856) showed a gap of about 2 points (see 08). Three more files test what DINOv2 adds on the real test set. DINOv2 is used **without TTA**: TTA lowered its own score (0.839 vs 0.851), and the ensembles score the same either way (4 models: 0.8593 without vs 0.8594 with). The files are built from saved probabilities by [experiments/combine_submissions.py](../experiments/combine_submissions.py) and stored in `results/submissions/`.

| File | Models | Val wF1 | Differs from the 0.8371 file | Kaggle public |
| --- | --- | ---: | ---: | ---: |
| `ensemble_tiny224_tiny384_small288` (reference) | ConvNeXt Tiny-224 + Tiny-384 + Small-288 | 0.856 | — | 0.8371 |
| `lb1_a_dinov2_alone` | DINOv2-Base 336 | 0.851 | 10.6% of test images | 0.8321 |
| `lb1_b_4models_convnext3_dinov2` | the three ConvNeXt + DINOv2 | 0.859 | 2.4% | 0.8410 |
| `lb1_c_tiny384_small288_dinov2` | Tiny-384 + Small-288 + DINOv2 | 0.860 | 3.4% | **0.8424** |

File A answers whether one DINOv2 model generalises better than the ConvNeXt ensemble. B and C answer whether adding DINOv2 to ConvNeXt lifts the leaderboard. B and C change only 2–3% of the predictions, so their scores will move by tenths of a point at most.

### What the leaderboard told us

* **Validation ranks submissions correctly.** All four files have the same order on the leaderboard as on fold-0 validation, with a nearly constant gap of 1.7–1.9 points. Local validation can therefore be used to choose between models; the leaderboard is only needed to confirm.
* **Model diversity pays off on the test set.** Adding DINOv2 to ConvNeXt raised the leaderboard from 0.8371 to 0.8424 (+0.5), although DINOv2 alone (0.8321) is weaker than the ConvNeXt ensemble.
* **The constant gap of about 1.8 points is structural:** it does not depend on the model. Likely causes are training on 80% of the data (fold-0 models) and a mild test-domain shift.
* **Best file so far:** `lb1_c_tiny384_small288_dinov2`, 0.8424.

**Next:** 5-fold training of DINOv2 and ConvNeXt-Tiny-384 (models that together see all training data, fold ensembles, out-of-fold scores), then a third family (EVA-02 or ConvNeXt-V2), then pseudo-labelling to attack the gap.

## Newer models: DINOv3 (2025), fold 0

Both runs use the same recipe as their counterparts, so the comparison is direct. Configs: `n1_dinov3_convnext_tiny_288_fold0.env` and `n2_dinov3_base_384_recipe_fold0.env`.

| Run | Weighted F1 | + TTA | Compared with | Shared errors | Effect on the best ensemble |
| --- | ---: | ---: | --- | ---: | --- |
| N1: DINOv3 ConvNeXt-Tiny 288 | 0.840 | 0.833 | ImageNet ConvNeXt-Tiny 288: 0.846 | 79–83% with the other models | none (best combos do not include it) |
| N2: DINOv3 ViT-B/16 384 | 0.832 | 0.843 | DINOv2-B 336: 0.851 | 84% with DINOv2 | lowers it: 0.860 → 0.852–0.858 |

**Verdict:** with the same recipe, DINOv3 is slightly weaker than DINOv2 on this task and adds no diversity. It also trains slower (95 vs 70 min on the laptop). DINOv2 remains the model for 5-fold training (`dinov2_5fold` in the Colab/Kaggle notebooks). A DINOv3 learning-rate search could change this, but it is not a priority.

### DINOv3 on the leaderboard, and robustness to corruptions

DINOv3 alone (no TTA) scored **0.8253** on Kaggle, only 0.7 points below its validation score (0.832). DINOv2 lost 1.9 points (0.851 → 0.8321). Scoring the fold-0 validation images with the test-like corruptions applied explains the difference:

| Drop from clean validation (points) | DINOv2 | DINOv3 | ConvNeXt-Tiny-384 | ConvNeXt-Small-288 |
| --- | ---: | ---: | ---: | ---: |
| Colour, brightness, noise, blur | −1.0 | **−0.1** | −0.8 | −1.3 |
| Same + an overlay on every image | −2.6 | **−0.9** | −2.8 | −1.3 |
| Colour casts only | −2.0 | **−0.3** | −1.0 | −1.1 |

DINOv3 is clearly more robust as a single model. On "test-like" validation (default corruption probabilities), DINOv3 is only 0.4 points behind DINOv2 (0.838 vs 0.842), matching the leaderboard far better than clean validation. **From now on, report test-like validation next to clean validation.**

In ensembles the robustness is already covered by ConvNeXt + DINOv2:

| Ensemble | Clean | Test-like | Heavy | Kaggle |
| --- | ---: | ---: | ---: | ---: |
| Tiny-384 + Small-288 + DINOv2 | 0.8596 | 0.8597 | 0.8410 | 0.8424 |
| 3 ConvNeXt + DINOv2 | 0.8593 | 0.8627 | 0.8405 | 0.8410 |
| 3 ConvNeXt + DINOv2 + DINOv3 (`lb1_d_...`) | 0.8547 | 0.8609 | 0.8431 | |

Adding DINOv3 is expected to be neutral on the leaderboard. The plan stays 5-fold DINOv2.
