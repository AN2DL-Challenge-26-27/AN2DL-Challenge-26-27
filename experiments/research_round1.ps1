# Research round 1 (docs/10): R1 = new recipe on ConvNeXt-Tiny, R2 = DINOv2-Base at 336 px.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File experiments/research_round1.ps1
# Optional: $env:SPAGHETTI_PYTHON = "path\to\python.exe" (default: python on PATH).
# Both runs use --resume, so running the script again continues an interrupted run.

$ErrorActionPreference = "Stop"
$python = if ($env:SPAGHETTI_PYTHON) { $env:SPAGHETTI_PYTHON } else { "python" }

# Shared recipe from the research review: no class weights, MixUp instead of label smoothing, EMA,
# stronger stochastic depth; label policy drop_copies to compare with E1a-E3.
$recipe = @(
    "--label-policy", "drop_copies", "--class-weight-power", "0", "--label-smoothing", "0",
    "--mixup-alpha", "0.3", "--mixup-prob", "0.5", "--mixup-off-epochs", "5",
    "--ema-decay", "0.998", "--drop-path-rate", "0.25", "--resume"
)

# R1: ConvNeXt-Tiny 224 px, compare with e1_convnext_tiny_drop_copies_fold0 (0.843).
& $python -m spaghetti1 train --run-name r1_convnext_tiny_224_recipe_fold0 `
    --epochs 30 --lr 3e-4 --layer-decay 0.85 @recipe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# R2: DINOv2-Base (patch 14) at 336 px = 24 x 24 patches; batch 8 x 4 accumulation = effective 32.
& $python -m spaghetti1 train --run-name r2_dinov2_base_336_recipe_fold0 `
    --model vit_base_patch14_reg4_dinov2.lvd142m --image-size 336 --batch-size 8 --accumulation-steps 4 `
    --grad-checkpointing --epochs 25 --lr 2e-4 --layer-decay 0.75 @recipe
exit $LASTEXITCODE
