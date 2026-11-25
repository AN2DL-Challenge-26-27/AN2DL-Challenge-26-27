# `spaghetti1/preprocessing`: cleaning, manifests, transforms, and data loading

Full explanation: [docs/03_PREPROCESSING.md](../../docs/03_PREPROCESSING.md) to [docs/06_SHARING_AND_LOADING_DATA.md](../../docs/06_SHARING_AND_LOADING_DATA.md).

Read the modules in this order:

| # | Module | Role |
| --- | --- | --- |
| 1 | `images.py` | `read_rgb_png`: decode and validate a 256 × 256 RGB PNG (used everywhere) |
| 2 | `quality.py` | Per-image flags: alien overlay, blank frame, noise level, background colour |
| 3 | `duplicates.py` | Star-field fingerprints, duplicate matching over 8 orientations, union-find groups |
| 4 | `manifest.py` | `python -m spaghetti1 preprocess`: runs 2 and 3, marks originals and copies, assigns folds, writes `manifests/` and review sheets |
| 5 | `review.py` | Contact sheets used by `manifest.py` |
| 6 | `transforms.py` | `GalaxyTransform`: augmentation (training) and deterministic preprocessing (all splits) |
| 7 | `preview.py` | `python -m spaghetti1 augment-preview` |
| 8 | `dataset.py` | `GalaxyDataset`, `split_fold`, label policies, class weights, balanced sampler |
| 9 | `loaders.py` | `build_dataloaders`: one call from dataset folder to train/validation/test DataLoaders |
