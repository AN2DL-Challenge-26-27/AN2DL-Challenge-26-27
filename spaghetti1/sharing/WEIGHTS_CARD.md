# SPAGHETTI-1 model weights (`spaghetti1_weights_v2`)

Trained models from the SPAGHETTI-1 experiments, for evaluating and predicting **without retraining**. Full instructions: `docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md` in the team repository; results: `docs/08_TRAINING_AND_RESULTS.md` and `docs/10_RESEARCH_REVIEW_AND_PLAN.md`.

## Models (all trained on CV fold 0; validation = 599 originals and unique images)

| Run folder | Backbone (`timm`, general-purpose pretrained) | Input | Recipe | Val wF1 | + TTA |
| --- | --- | ---: | --- | ---: | ---: |
| `e0_convnext_tiny_original_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | label policy original | 0.836 | 0.835 |
| `e1_convnext_tiny_drop_copies_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | drop_copies | 0.843 | 0.844 |
| `e1_convnext_tiny_as_given_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | as_given | 0.836 | 0.829 |
| `e2_convnext_tiny_288_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 288 | drop_copies | 0.846 | 0.833 |
| `e2_convnext_tiny_384_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 384 | drop_copies | 0.847 | 0.846 |
| `e3_convnext_small_288_fold0` | `convnext_small.fb_in22k_ft_in1k` | 288 | drop_copies | 0.841 | 0.847 |
| `r1_convnext_tiny_224_recipe_fold0` | `convnext_tiny.fb_in22k_ft_in1k` | 224 | research recipe | 0.831 | 0.828 |
| `r2_dinov2_base_336_recipe_fold0` | `vit_base_patch14_reg4_dinov2.lvd142m` | 336 | research recipe | **0.852** | 0.839 |
| `n1_dinov3_convnext_tiny_288_fold0` | `convnext_tiny.dinov3_lvd1689m` | 288 | drop_copies | 0.840 | 0.833 |
| `n2_dinov3_base_384_recipe_fold0` | `vit_base_patch16_dinov3.lvd1689m` | 384 | research recipe | 0.832 | 0.843 |

DINOv3 (`n2_...`) scores lower on clean validation but is the most robust to the test-like corruptions (docs/10).

## Ensembles with Kaggle scores

Members use the same TTA choice as the submissions: ConvNeXt with 8-way TTA, DINOv2 and DINOv3 without TTA.

| Members | Val wF1 | Test-like val | Kaggle public |
| --- | ---: | ---: | ---: |
| e1 + e2_384 + e3 + r2 + n2 | 0.855 | 0.861 | **0.8441** |
| e2_384 + e3 + r2 | 0.860 | 0.860 | 0.8424 |
| e1 + e2_384 + e3 | 0.856 | | 0.8371 |

## Files per run

* `best.pt`: PyTorch `state_dict` of the best validation epoch (SHA-256 in `weights_info.json`)
* `config.json` (and `config.env` for newer runs): every training setting; the loader rebuilds the network and preprocessing from it
* `metrics.json`, `history.csv`, `val_predictions.csv`: scores, learning curves, validation probabilities
* `val_probabilities*.npy`, `test_probabilities.npy` (where saved): raw probabilities for building ensembles without a GPU

## Use

```bash
python -m spaghetti1 fetch-weights --source "<Drive link or path to spaghetti1_weights_v2.zip>"
python -m spaghetti1 evaluate --run-dir data/weights/spaghetti1_weights_v2/runs/r2_dinov2_base_336_recipe_fold0
python -m spaghetti1 predict --output-dir outputs/submissions/my_ensemble --run-dirs \
    data/weights/spaghetti1_weights_v2/runs/e2_convnext_tiny_384_fold0 \
    data/weights/spaghetti1_weights_v2/runs/r2_dinov2_base_336_recipe_fold0
```
