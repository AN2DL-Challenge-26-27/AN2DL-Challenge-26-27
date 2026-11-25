# `spaghetti1/sharing`: the cleaned dataset and model weights on Google Drive

Full explanation: [docs/06_SHARING_AND_LOADING_DATA.md](../../docs/06_SHARING_AND_LOADING_DATA.md) and [docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md](../../docs/09_MODEL_WEIGHTS_AND_REPRODUCTION.md).

| # | File | Role |
| --- | --- | --- |
| 1 | `export.py` | `python -m spaghetti1 export`: pack kept training images, all test images, and the manifests into `spaghetti1_clean_v1.zip` |
| 2 | `DATASET_CARD.md` | Description placed inside the zip |
| 3 | `fetch.py` | `prepare_dataset(source)` / `python -m spaghetti1 fetch`: download from a Drive link (or copy from a path), extract, verify, and return the dataset folder |
| 4 | `weights.py` | `export-weights` / `fetch-weights` / `export-results`: the model-weights zip (with SHA-256 checks) and the `results/` snapshot |
| 5 | `WEIGHTS_CARD.md` | Description placed inside the weights zip |
