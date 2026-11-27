# SPAGHETTI-1 | Image Classification: shared files

This is the team's Google Drive folder. Code and documentation are in the GitHub repository; start with `docs/00_START_HERE.md` there.

| Item | Contents |
| --- | --- |
| `spaghetti1_clean_v1.zip` | Cleaned dataset (3,995 training images, 14,736 test images, manifests and folds); used by the Colab notebooks |
| `spaghetti1_weights_v2.zip` | All 10 trained laptop models (fold 0) with checksums and saved predictions; replaces `spaghetti1_weights_v1.zip` (6 models). Load with `python -m spaghetti1 fetch-weights --source <zip>` |
| `submissions/` | Every submission file made so far, each with `submission.csv`, `ensemble_info.json` (models and validation score), and the members' probabilities (`.npy`) |
| `colab_runs/` | Created by `notebooks/03_colab_training.ipynb` (5-fold training results) |

## Submissions and Kaggle public scores

| Folder in `submissions/` | Models | Validation wF1 | Kaggle public |
| --- | --- | ---: | ---: |
| `lb1_d_5models_convnext3_dinov2_dinov3` | ConvNeXt Tiny-224 + Tiny-384 + Small-288 + DINOv2 + DINOv3 | 0.855 (test-like 0.861) | **0.8441 (best)** |
| `lb1_c_tiny384_small288_dinov2` | ConvNeXt Tiny-384 + Small-288 + DINOv2 | 0.860 | 0.8424 |
| `lb1_b_4models_convnext3_dinov2` | 3 ConvNeXt + DINOv2 | 0.859 | 0.8410 |
| `ensemble_tiny224_tiny384_small288` | 3 ConvNeXt | 0.856 | 0.8371 |
| `lb1_a_dinov2_alone` (= `dinov2_base_336_notta`) | DINOv2-Base 336 | 0.851 | 0.8321 |
| `n2_dinov3_base_384_recipe_fold0` | DINOv3-Base 384 | 0.832 | 0.8253 |
| `ensemble_tiny224_tiny384` | ConvNeXt Tiny-224 + Tiny-384 | 0.846 | not submitted |
| `e0_convnext_tiny_original_fold0` | ConvNeXt-Tiny 224 baseline | 0.836 | not submitted |

ConvNeXt members use 8-way TTA probabilities; DINOv2 and DINOv3 use probabilities without TTA. "Test-like" validation scores the validation images with test-like corruptions; it tracks the leaderboard better than clean validation (`docs/10`).

## Next

Run `notebooks/03_colab_training.ipynb` with `PLAN = "dinov2_5fold"`, then with `PLAN = "dinov3_5fold"`. The results appear in `colab_runs/`; the small `spaghetti1_results_*.zip` files are what is needed to build the next ensemble.
