# Deep-research report: improving beyond 0.856 weighted F1 (received 2026-11-27)

> **Source and status.** This is the answer of an external deep-research tool to the prompt summarised in [docs/10](../10_RESEARCH_REVIEW_AND_PLAN.md). It is kept unedited for reference. Published scores and paper claims quoted here were **not verified** by the team; [docs/10](../10_RESEARCH_REVIEW_AND_PLAN.md) lists which points our own data confirms, which we test, and which we postpone.

Your plateau at **0.845 single-model / 0.856 same-family ensemble** is not a sign of a weak ConvNeXt recipe—it is the expected ceiling of supervised ImageNet-22k ConvNets trained on ~3,000 unique skies. On the *full* Galaxy10 DECaLS dataset (17,736 images; ~14.2k training images, roughly 4× your labelled set), reproducible multi-seed benchmarks of general-purpose backbones land at **0.84–0.88**, with the exact same weak classes you observed: Disturbed/Irregular (~0.54 F1), Unbarred Tight Spiral (~0.75 F1), and Unbarred Loose Spiral (~0.76 F1).

Two structural facts in your setup represent your biggest untapped gains:

1. **Your unlabelled test set (14,736 images) holds ~80% of the unique galaxies** (3,000 unique train skies + 14,736 test ≈ 17,736 full Galaxy10 DECaLS) *plus* the exact target corruptions. Semi-supervised pseudo-labelling is therefore far stronger here than on standard Kaggle tasks.
2. **Your current loss (`1/√n` class weights) and single-fold validation (599 images) work against you.** `1/√n` optimizes for Macro F1 rather than Weighted F1, and 599 validation samples carry a binomial 1σ standard error of √(0.85 × 0.15 / 599) ≈ **0.015** (1.5 F1 points), explaining why fold-0 deltas of ±0.6 points are mostly noise.

---

## 1. Backbones & Input Resolutions (General-Purpose Weights)

Published Galaxy10 DECaLS results (general-purpose pretrained backbones, full ~14.2k–15.9k training splits unless noted):

| Backbone Family | Representative Checkpoint (`timm`) | Reported Galaxy10 DECaLS Score | Sweet-Spot Resolution | 8 GB VRAM Fit (`bf16` + grad ckpt) | Role for Hard Classes |
| --- | --- | --- | --- | --- | --- |
| **SSL Plain ViT (DINOv2)** | `vit_base_patch14_reg4_dinov2.lvd142m` | **0.8789 F1 / 0.8786 acc** (GalaxAlign baseline); **85.19% acc** (Large, 3-seed 224 px); **84.31% wF1** (Small) | **336–448 px** (must be multiple of 14; 336 = 24×24 tokens) | Base fits easily (`bs=8`, accum 4); Large better on Colab | Best single general-purpose family for fine-grained outer texture & low-surface-brightness **Irregular** asymmetry |
| **MIM-CLIP ViT (EVA-02)** | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | Strongest general-purpose fine-grained ViT family; outperforms IN-22k ViT/Swin on small noisy datasets | **336–448 px** (native 448 = 32×32 tokens) | Fits 8 GB at 336 px (`bs=8`); 448 px fits with grad ckpt (`bs=4`, accum 8) | Dense 14×14 patching + RoPE resolves fine arm pitch angle (**Tight vs. Loose**) |
| **Hybrid CNN-Transformer** | `caformer_s36.sail_in22k_ft_in1k_384` / `maxvit_tiny_tf_384.in1k` | MaxViT reaches top-tier galaxy morphology accuracy; Astroformer (CoAtNet/MaxViT-style) claims 94.86%* | **288–384 px** (MaxViT requires exact divisible sizes: 224 or 384) | `caformer_s36` & `maxvit_small` fit 8 GB at 384 px | Local conv stems capture central bars/bulges while global attention compares arm spacing across the disc |
| **Windowed Transformer (Swin)** | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | Consistently matches or edges out ConvNeXt on fine-grained small/faint galaxies | **384 px** (window 12) or **256 px** (`swinv2_base_window16_256`) | Fits 8 GB (`bs=8`, grad ckpt) | High orthogonality to ConvNeXt; shifted windows capture global spiral symmetry |
| **FCMAE ConvNet (ConvNeXt-V2)** | `convnextv2_base.fcmae_ft_in22k_in1k_384` (or `tiny`) | `convnextv2-atto-1k` alone hits **83.95% wF1** (3-seed, 224 px); ConvNeXt-B hits **84.31% acc** | **288–384 px** | `tiny` native; `base` @ 384 fits with grad ckpt | Global Response Normalization (GRN) suppresses background noise/colour casts better than ConvNeXt-V1 |
| **EfficientNetV2 / B7** | `tf_efficientnetv2_m.in21k_ft_in1k` | `efficientnet_b7`: **84.93 ± 0.50% wF1** (3-seed, 224 px) | **288–384 px** | Fits 8 GB comfortably (`bs=16`) | Diverse inductive bias (MBConv + SE blocks) vs. ConvNeXt's 7×7 depthwise kernels; fast training |
| **BEiT / BEiT-v2** | `beitv2_base_patch16_224.in1k_ft_in22k_in1k` | Evaluated in 2026 Galaxy10 foundation benchmark | **224–384 px** (patch-16) | Fits 8 GB easily | Generally superseded by **EVA-02** (which uses CLIP-distilled MIM + SwiGLU + RoPE) |

**Caveat on >94% outlier papers:* Astroformer-4 (271M params) reports 94.86% and a 2025 GCNN+VAE paper reports 98.6% (and a 2024 ViT paper claims 99.85%), but independent, audited multi-seed benchmarks on clean Galaxy10 DECaLS splits consistently plateau at **84–88%** for general-purpose backbones and **~92.9%** even for astronomy-pretrained Zoobot. Treat 94%+ figures as non-reproducible/split-artifact outliers rather than realistic targets.

### Why resolution behaved the way it did—and when upscaling pays off

* **Why ConvNeXt saturated between 224 and 384 px:** Your source cutouts are natively 256×256. ConvNeXt uses a stride-4 stem, so at 288 px its first feature map is already 72×72—upscaling further adds interpolated blur rather than new spatial information.
* **Why ViTs (DINOv2, EVA-02) *do* gain from 336–448 px:** Patch-14/16 ViTs downsample 14×–16× immediately. At 224 px, a patch-16 ViT sees only a 14×14 grid—barely 4–6 tokens across a galaxy's disc, smearing tight vs. loose spiral arms together. At **336 px (24×24 tokens)** or **448 px (32×32 tokens)**, patch-14 ViTs resolve individual arm pitch angles and faint irregular clumps. Use **bicubic interpolation** and prefer **patch-14** (`dinov2`, `eva02`) over patch-16/32.

---

## 2. Rotation-Equivariant & Rotation-Invariant Networks

* **Pure E(2)-Equivariant GCNNs (`escnn`):** Pandya et al. (2023) tested cyclic (C_N) and dihedral (D_N) group CNNs on Galaxy10 DECaLS and showed higher clean accuracy than matched vanilla CNNs plus **<6% accuracy loss under 50% Poisson noise**, with D_4/D_8 outperforming heavier D_16 models—though all models still struggled most on **Disturbed/Irregular** galaxies.
* **Why not train `escnn` from scratch here:** E(2) networks lack ImageNet-22k/LVD-142M pretrained weights. On 3,196 noisy training images per fold, losing large-scale pretraining costs far more (~5–8 F1 points) than exact equivariance recovers.
* **Two pretrained-compatible alternatives that *do* work:**
  1. **Train-time + Test-time D_4 Embedding Pooling (Dieleman-style):** Pass 4 (or 8) dihedral views (0°, 90°, 180°, 270° + flips) of each image through a *shared pretrained backbone*, average (or concat mean + max) the penultimate embeddings, and feed the pooled vector into one linear head. Unlike post-hoc TTA, this backpropagates through the invariant representation, stabilizes predictions under synthetic overlays/noise, and fits an 8 GB GPU cleanly with `convnext_tiny` or `vit_small_patch14_dinov2` at `bs=4` (effective 16–32 views) + grad accumulation.
  2. **Polar-Coordinate Unwrapping (Auxiliary Diversity Model):** Remap each 256×256 cutout from (x, y) centered on the galaxy to (θ, log r) polar coordinates (inspired by polar-representation galaxy pipelines). In polar space:
     * **Tight vs. Loose spirals** become parallel diagonal stripes whose **slope is literally the spiral pitch angle** (shallow slope = tight; steep slope = loose).
     * **Round smooth** galaxies become flat horizontal bands; **Barred spirals** show a strong 180°-periodic brightness ridge near the core; **Irregular/Disturbed** galaxies exhibit broken, non-periodic vertical asymmetry.
     * Even if a polar-input ConvNeXt/Swin scores 1–2 points lower standalone (~0.83), its error correlation with Cartesian models drops sharply—giving outsized ensemble gains specifically on **Tight ↔ Loose** and **Irregular ↔ Loose**.

---

## 3. Regularisation (Fixing the 0.97 Train Accuracy Overfitting)

Your ConvNeXt-Small hitting 0.97 train accuracy while plateauing at 0.841 val F1 is classic memorization of Galaxy Zoo label noise. Four changes directly fix this:

1. **Layer-Wise Learning Rate Decay (LLRD):** Scale learning rates by ω^(L−l) across stages/blocks (ω = 0.70–0.80 for ConvNeXt/Swin; ω = 0.65–0.75 for DINOv2/EVA-02, with base backbone LR `5e-5` to `1e-4` and head LR `5e-4` to `1e-3`). LLRD preserves pretrained low-level filters, cuts across-seed variance, and stops small-dataset overfitting.
2. **Stochastic Depth (`drop_path_rate`) + Weight Decay:** Ensure `drop_path_rate` is explicitly set (in `timm`, defaults are often `0.0`! Notice Hugging Face's default ConvNeXtV2 config uses `0.0`). Use **`0.2` for Tiny/Small** and **`0.3–0.4` for Base** models, paired with `weight_decay=0.05` (up to `0.1` for Base ViTs).
3. **Model EMA (`timm.utils.ModelEmaV2`):** Exponential moving average with `decay=0.999` (or `0.998` given short ~100-step epochs on 3.2k images). EMA acts as free checkpoint ensembling along the trajectory and dampens late-epoch noisy-label spikes (+0.3–0.7 F1 stability).
4. **MixUp YES, CutMix NO:**
   * **Avoid CutMix** on galaxies: pasting a square patch onto a radially symmetric galaxy creates physically contradictory cues (e.g., grafting a spiral arm onto a smooth core mimics Merging/Irregular).
   * **Use gentle MixUp** (`alpha=0.2–0.4`, `p=0.5`, disabled in the final 5 epochs) *instead of* hard `label_smoothing=0.1`. MixUp prevents overconfident memorization of boundary cases without corrupting spatial structure.
   * **Augmentation audit for Irregular/Loose:** Check your blur/darkening/noise strength. Aggressive darkening or heavy Gaussian blur can erase low-surface-brightness outer spiral arms, turning a Loose Spiral into a Smooth galaxy or an Irregular galaxy into a blurred blob. Keep heavy corruptions probabilistic (p ≈ 0.3–0.5) and include a **mild `asinh` (arcsinh) luminance stretch** test (see Section 7).

---

## 4. Learning with Noisy Labels (LNL)

Galaxy10 DECaLS labels derive from discrete thresholds on Galaxy Zoo citizen-science vote fractions. Many Tight↔Loose and Irregular↔Loose galaxies sit near a 50/50 volunteer split—meaning the true target is soft, not one-hot.

* **Replace uniform `label_smoothing=0.1` with OOF Soft-Target Self-Distillation:** Uniform smoothing dumps probability mass onto impossible classes (e.g., assigning 1% probability to Edge-on when the galaxy is a face-on spiral). Instead, once you have a diverse 5-fold OOF ensemble p_OOF:

  y_target = (1 − β) · y_hard + β · p_OOF   (β ∈ [0.20, 0.35])

  This puts soft mass *only* on the real visual neighbour (e.g., 70% Loose / 30% Tight), directly matching the underlying Galaxy Zoo vote distribution.
* **Early-Learning Regularization (ELR) or Symmetric Cross-Entropy (SCE):** Both are standard, drop-in LNL objectives for fine-grained noisy datasets:
  * **SCE:** L_SCE = α · CE(y, p) + β · CE(p, y) (with α = 1.0, β = 0.1–0.3, clamping y_min = 10⁻⁴). The reverse-CE term bounds gradients from badly mislabelled Irregular/Spiral outliers.
  * **ELR:** Adds λ · log(1 − ⟨p, t⟩) where t is an EMA of each sample's early-epoch predictions, stopping the model from memorizing noisy labels after epoch ~10 without slowing early learning.
* **Handling your 719 multi-copy groups:** Because the noisy copies consistently carry the neighbouring label (Smooth↔In-between, Barred↔Tight) and already appear brighter/noisier, **do not discard them**—treat them as free domain-augmented training samples using the **original image's clean label** (or better, the **OOF ensemble soft target** computed on the clean original). Keeping their original-anchored soft label eliminates the contradictory supervision while preserving their brightness/noise regularization.

---

## 5. Semi-Supervised Learning & Pseudo-Labelling on the 14,736 Test Images

With ~3,000 unique labelled skies and **14,736 unlabelled target-domain skies**, semi-supervised learning is your highest-leverage path from ~0.86 to ~0.90. Semi-supervised galaxy studies (e.g., FixMatch-style consistency and uncertainty filtering) consistently report major gains when unlabelled DECaLS pools outnumber labelled samples.

**Recommended protocol (Iterative Soft Noisy-Student / Weak-to-Strong Consistency):**

1. **Build a diverse 5-fold Teacher Ensemble first** (at least 3 distinct families: e.g., ConvNeXt-V2 + DINOv2 + EVA-02/Swin). Never pseudo-label from a single ConvNeXt family—that just amplifies its existing 70% shared errors.
2. **Generate clean Teacher probabilities on the 14,736 test images** using **8-way D_4 TTA** (and optionally test-time per-image channel-balance/whitening *only* if severe colour-cast test images show degraded teacher entropy).
3. **Avoid global hard confidence thresholds (e.g., `max_prob > 0.90`):** A global threshold selects almost exclusively Round Smooth and Edge-on galaxies while starving **Irregular (F1 0.50)**, **Tight Spiral**, and **Loose Spiral**. Instead:
   * Use **temperature-sharpened soft targets** (p_i^(1/T) / Σ_j p_j^(1/T) with T ≈ 0.75–0.85) so uncertain Tight/Loose galaxies teach calibrated distributions rather than wrong hard labels.
   * Filter by **class-stratified top-K%** (e.g., keep the top 60–75% lowest-entropy test samples *within each predicted class* in Round 1, expanding to 85–90% in Round 2), ensuring Irregular and Cigar-shaped maintain healthy representation.
4. **Weak-to-Strong Student training:** Train each student fold on `(labelled_fold_train + selected_pseudo_test)`, applying **strong augmentations** (360° rotation + extra colour/noise/blur + MixUp) to the pseudo-labelled test images while keeping their loss weight at λ_u ∈ [0.5, 1.0] (or sampling 50% labelled / 50% pseudo-labelled per batch).
5. **Leakage-free local CV check:** To verify that pseudo-labelling truly helps before submitting, run one sanity check on Fold 0 where pseudo-labels are generated *using only Fold-0's teacher models* (never touching Fold-0 validation images). Once validated, generate final test pseudo-labels from the full 5-fold multi-family teacher.

---

## 6. Losses, Calibration & Post-Hoc Tuning for Weighted F1

* **Drop `1/√n` class weights in the training loss.** Weighted F1 is Σ_c (n_c / N) · F1_c. Rare classes like Cigar-shaped (n=56, ~1.4% support) carry almost zero weight in the global score, whereas frequent classes (Round Smooth, In-between, Loose Spiral, Tight Spiral) dominate. Upweighting rare classes shifts decision boundaries outward, lowering rare-class precision and stealing true positives from major classes.
* **Train with unweighted calibrated loss** (standard CE + gentle MixUp, or SCE / OOF soft targets) so predicted probabilities approximate true posteriors P(y | x).
* **Per-model Temperature Scaling:** Before ensembling, fit a single scalar temperature T_m per model on 5-fold OOF logits to minimize NLL. Overconfident models (like your ConvNeXt-Small at 0.97 train acc) otherwise hijack probability averages.
* **Post-Hoc Class Multipliers for Weighted F1:** Optimize 10 positive multipliers w = (w_0, …, w_9) predicting ŷ = argmax_c (w_c · p_c) directly for Weighted F1 on **full 5-fold OOF** (~3,000 samples) using coordinate ascent or Powell/Nelder-Mead optimization:
  * **Guard against overfitting:** Evaluate with **nested OOF** (fit w on 4 folds, measure Weighted F1 on the held-out 5th fold) and shrink toward uniform: w_final = 1 + 0.6 · (w_opt − 1).
  * Expect **+0.3 to +0.8 Weighted F1**, primarily by pulling back over-predicted Irregular/Cigar false positives and sharpening the Tight ↔ Loose decision boundary.

---

## 7. Specifically Separating Irregular, Tight Spiral & Loose Spiral

Why these three collide—and targeted fixes for each boundary:

* **Unbarred Tight Spiral ↔ Unbarred Loose Spiral (pitch angle & arm-to-interarm contrast):**
  * **What separates them:** Tight spirals have closely wound arms with small pitch angles and thin dark gaps; loose spirals have wide, open arms with large pitch angles and lower surface brightness.
  * **What helps:** (1) **Patch-14 ViTs at 336–448 px** (DINOv2, EVA-02) to preserve arm gap geometry; (2) **Polar-coordinate auxiliary model**, where pitch angle becomes linear slope; (3) **Mild `asinh` stretch** on a dedicated auxiliary branch (or 4-channel / blended stretch) to lift faint outer arms without saturating the core (keep stretch gentle so test noise isn't amplified).
* **Irregular/Disturbed ↔ Unbarred Loose Spiral (low-surface-brightness asymmetry & chaotic clumping):**
  * **What separates them:** Loose spirals maintain a coherent 2-arm or grand-design symmetry around a central core; Disturbed/Irregular galaxies show asymmetric outer tidal debris, off-center clumps, or disrupted discs.
  * **What helps:** (1) **Global self-attention (DINOv2 / EVA-02 / Swin)**—pure CNNs focus on local star-forming knots that look identical in both classes, whereas global attention measures *180° rotational asymmetry* across the whole disc; (2) **Avoid aggressive cropping (`RandomResizedCrop` scale < 0.85)**—galaxy cutouts are already centered, and cropping off outer tidal tails turns Disturbed into Loose Spiral; use tight scale jitter (`0.90–1.05`) + small shift (`±5%`) + full 360° rotation; (3) **Soft OOF targets + post-hoc multipliers**, since Disturbed often suffers low recall (~0.44) or erratic precision under hard-label noise.

---

## 8. Ensembling Across Architectures & Folds

Your 70% error overlap (75/107 shared errors) happened because Tiny-224, Tiny-384, and Small-288 share the same inductive bias, pretraining objective (supervised IN-22k), and coordinate system.

* **Target 4 orthogonal axes of diversity:**
  1. **Self-Supervised Plain ViT:** `vit_base_patch14_reg4_dinov2.lvd142m` (336 px)
  2. **MIM-CLIP Plain ViT:** `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` (336 or 448 px)
  3. **Hierarchical Hybrid / Windowed Transformer:** `caformer_s36.sail_in22k_ft_in1k_384` or `swin_base_patch4_window12_384.ms_in22k_ft_in1k` (384 px)
  4. **Modern ConvNet:** `convnextv2_base.fcmae_ft_in22k_in1k_384` (or your `convnext_small` with fixed regularization) + optional **Polar-input `convnext_tiny` / `swin_small`**.
* **Blending mechanics:**
  * Always train **full 5-fold CV** for your top 3–4 architectures (averaging 5 folds per architecture alone typically adds **+0.5 to +1.0 F1** over single-fold inference on the test set).
  * Compare **temperature-scaled probability average** vs. **calibrated logit average (geometric mean)** on 5-fold OOF; geometric mean often wins on noisy fine-grained classes because one confident model cannot override three moderate disagreements.
  * Optimize non-negative ensemble weights on **5-fold OOF log-loss** (via SciPy SLSQP), *then* tune regularized class multipliers on Weighted F1. Optimizing blend weights directly on F1 overfits.

---

## 9. Realistic Weighted-F1 Ceiling Estimate

Ground your expectations against two hard constraints: (a) general-purpose weights only (no Zoobot), and (b) Galaxy Zoo volunteer vote ambiguity (~6–9% irreducible Bayes error on Galaxy10 DECaLS, consistent with your 0.90 6-model oracle).

| Milestone | Expected 5-Fold OOF / Test Weighted F1 | Primary Driver |
| --- | --- | --- |
| Current baseline (ConvNeXt V1 Tiny/Small, fold-0 ensemble) | **0.856** | Single architecture family, ~3.2k train, `1/√n` weights |
| Fixed recipe + full 5-fold OOF + unweighted loss + LLRD/EMA/MixUp | **0.862 – 0.870** | Removing overfitting & loss/metric mismatch |
| Diverse 4-family supervised 5-fold ensemble (+ polar model + multipliers) | **0.875 – 0.888** | Breaking the 70% error correlation (DINOv2 + EVA-02 + Swin/CAFormer + ConvNeXt-V2) |
| **+ Iterative Soft Pseudo-Labelling on 14,736 test images (1–2 rounds)** | **0.890 – 0.905** | Unlocking 5× more unique galaxies + direct test-corruption adaptation |
| **Hard practical ceiling (no astronomical pretraining)** | **~0.905 – 0.915** | Bounded by Galaxy Zoo label noise on Irregular / Tight / Loose boundaries |

Reaching **~0.89–0.90** is genuinely achievable here—specifically *because* the 14,736 unlabelled test set lets you recover the missing ~80% of Galaxy10 DECaLS.

---

## 10. Prioritised, Compute-Aware Plan (RTX 4060 8 GB + Colab)

All settings below assume `bf16` (RTX 4060 / Colab L4/A100) or `fp16` (Colab T4), `model.set_grad_checkpointing(True)`, effective batch size 32 via gradient accumulation, and **full 5-fold OOF tracking** (~3,000 clean/corrupted unique skies).

| Priority | Action | Hardware & Est. Time (5 Folds) | Expected Gain (Δ wF1) |
| --- | --- | --- | --- |
| **P1 (Immediate, 0 training)** | Switch evaluation to **full 5-fold OOF**; drop `1/√n` class weights; add **EMA (`0.9985`)**, **`drop_path_rate=0.25–0.35`**, **LLRD (`0.75`)**, **MixUp (`alpha=0.3`, off last 5 ep)**, tight crop (`0.92–1.0`); keep 719 copy groups with the original's label. | RTX 4060 (~20–30 min on 1 fold ablation, then 5 folds) | **+0.5 to +1.0** |
| **P2 (Highest single-model diversity)** | Train 5-fold **`vit_base_patch14_reg4_dinov2.lvd142m`** at **336 px** (`bs=8`, accum 4, LLRD `0.70`, `lr=5e-5`, head ×10, `drop_path=0.3`, 20–25 epochs). | RTX 4060 (~1.5–2 hrs for 5 folds) | **+1.0 to +1.8** when blended with ConvNeXt |
| **P3 (Second orthogonal transformer)** | Train 5-fold **`eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`** at **336 px** (local 4060, `bs=8`) or **448 px** (Colab T4/L4, `bs=4–8`, accum 4–8). | Colab or RTX 4060 (~2–3 hrs for 5 folds) | **+0.6 to +1.2** to ensemble |
| **P4 (Hybrid + Modern ConvNet)** | Train 5-fold **`convnextv2_base.fcmae_ft_in22k_in1k_384`** (288–384 px) *or* **`caformer_s36.sail_in22k_ft_in1k_384`** / **`swin_base_patch4_window12_384`** (384 px) to replace ConvNeXt-V1. | Colab L4/T4 or RTX 4060 (~2–3 hrs for 5 folds) | **+0.4 to +0.8** to ensemble |
| **P5 (Domain-specific diversity)** | Train 5-fold **Polar-unwrapped** `convnext_tiny` or `swin_small_patch4_window7_224` (256 px polar map (θ, log r)) + optionally test a mild `asinh`-stretched DINOv2 fold-0 ablation. | RTX 4060 (~45–60 min for 5 folds) | **+0.3 to +0.7** (targeted at Tight↔Loose & Irregular) |
| **P6 (The push toward 0.90)** | **Round-1 Soft Pseudo-Labelling:** Use your 4-family 5-fold ensemble + 8-way D_4 TTA to generate temperature-sharpened (T=0.8) soft labels on the 14,736 test images (class-stratified top 70% confidence) + OOF soft-target blending (β=0.25) on train. Retrain your top 3 families (**DINOv2-B, EVA-02-B, ConvNeXt-V2/CAFormer**) as strong-aug students (~15 epochs each). | Colab L4/A100 (~3–4 hrs total for 3 student families, 5 folds) | **+1.2 to +2.2** |
| **P7 (Final polish)** | **Temperature-scaled logit blend** (weights fit on OOF log-loss) + **nested-CV regularized class multipliers** for Weighted F1 + 8-way D_4 TTA at final test inference. | CPU (~5 min) | **+0.4 to +0.8** |

**Quick execution sanity check:** Run P1 + P2 on Fold 0 *first* to confirm DINOv2-Base (336 px) breaks the ConvNeXt error overlap (check shared error count—it should drop well below 70%). Once confirmed, scale P2–P4 to 5 folds so your Round-1 pseudo-labelling teacher (P6) is maximally diverse.
