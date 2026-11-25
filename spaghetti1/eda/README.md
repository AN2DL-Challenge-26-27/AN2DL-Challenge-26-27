# Exploratory data analysis (EDA)

- `image_stats.py` reads images through the preprocessing package and records decoding errors, file hashes, and pixel statistics.
- `report.py` builds tables for class distribution, train and test differences, low-contrast images, and exact duplicates.
- `plots.py` saves plots and sample image galleries.

From the project root, run `python -m spaghetti1 eda` with the virtual environment described in the main guide. The default input is `data/raw/` and the reports are saved to `outputs/eda/`. Documentation: [docs/02_EDA_FINDINGS.md](../../docs/02_EDA_FINDINGS.md) · [Main README](../../README.md)
