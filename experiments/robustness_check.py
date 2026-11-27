"""Ensemble scores on clean and test-like corrupted fold-0 validation images (docs/10). Run from the repo root with PYTHONPATH=. ."""
import itertools
import numpy as np, torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from spaghetti1.config import DATA_DIR
from spaghetti1.preprocessing.dataset import GalaxyDataset, load_train_manifest, split_fold
from spaghetti1.preprocessing.transforms import AugmentConfig, GalaxyTransform, PreprocessConfig
from spaghetti1.training.engine import predict
from spaghetti1.training.predict import load_run

if __name__ == "__main__":
    device, amp = torch.device("cuda"), torch.bfloat16
    _, val = split_fold(load_train_manifest(), 0); y = val["label"].to_numpy()
    still = dict(rotate=False, flip=False, scale_range=(1, 1), max_shift=0)
    # "test-like": the default photometric corruptions + overlays at their training probability (15%)
    settings = {"clean": None, "test-like": AugmentConfig(**still), "heavy": AugmentConfig(**still, p_artifact=1.0)}
    runs = {"t224": ("e1_convnext_tiny_drop_copies_fold0", True), "t384": ("e2_convnext_tiny_384_fold0", True),
            "s288": ("e3_convnext_small_288_fold0", True), "dino2": ("r2_dinov2_base_336_recipe_fold0", False),
            "dino3": ("n2_dinov3_base_384_recipe_fold0", False)}
    P = {s: {} for s in settings}
    for key, (run, tta) in runs.items():  # same TTA choice as in the submissions
        model, config = load_run(f"outputs/runs/{run}", device)
        for s, augment in settings.items():
            torch.manual_seed(0)
            loader = DataLoader(GalaxyDataset(val, DATA_DIR, GalaxyTransform(PreprocessConfig(image_size=config["image_size"]), augment)),
                                batch_size=32, num_workers=0)
            P[s][key] = predict(model, loader, device, tta=tta, amp_dtype=amp)[0]
        del model; torch.cuda.empty_cache(); print("done", key, flush=True)
    f = lambda s, c: f1_score(y, np.mean([P[s][k] for k in c], 0).argmax(1), average="weighted")
    rows = []
    for n in range(1, 6):
        for c in itertools.combinations(runs, n):
            rows.append((f("clean", c), f("test-like", c), f("heavy", c), c))
    rows.sort(key=lambda r: -(r[0] + r[1] + r[2]))
    print(f"{'models':38s} clean  test-like  heavy   mean")
    for r in rows[:12]:
        print(f"{'+'.join(r[3]):38s} {r[0]:.4f}  {r[1]:.4f}   {r[2]:.4f}  {np.mean(r[:3]):.4f}")
    for c in [("t384", "s288", "dino2"), ("dino3",), ("dino2",)]:
        print("ref", "+".join(c), [round(f(s, c), 4) for s in settings])
