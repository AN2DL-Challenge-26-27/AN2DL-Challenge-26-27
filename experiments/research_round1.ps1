# Research round 1 (docs/10): R1 = research recipe on ConvNeXt-Tiny, R2 = DINOv2-Base at 336 px.
# All settings are in the config files (configs/experiments/r1_*.env, r2_*.env), which inherit
# the shared recipe configs/recipes/research_v2.env. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File experiments/research_round1.ps1
# Optional: $env:SPAGHETTI_PYTHON = "path\to\python.exe" (default: python on PATH).
# The recipe sets RESUME=true, so running the script again continues an interrupted run.

$ErrorActionPreference = "Stop"
$python = if ($env:SPAGHETTI_PYTHON) { $env:SPAGHETTI_PYTHON } else { "python" }

foreach ($config in @(
    "configs/experiments/r1_convnext_tiny_224_recipe_fold0.env",
    "configs/experiments/r2_dinov2_base_336_recipe_fold0.env"
)) {
    & $python -m spaghetti1 train --config $config
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
