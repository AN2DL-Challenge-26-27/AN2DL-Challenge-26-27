# SPAGHETTI-1 project status and EDA report

[Download the progress report (PDF)](SPAGHETTI-1-Progress-Report.pdf)

As of November 25, 2026, the official dataset has been verified and all 19,064 images have been scanned. The main findings are a roughly 9-to-1 class imbalance and lower average brightness and contrast in the test set. An alien overlay was also observed in a training image and needs a separate validation check.

## Work completed

- Downloaded the official Kaggle dataset to `data/raw/` and verified that 4,328 training images and 14,736 test images match their CSV identifiers.
- Set up separate commands for download, verification, and exploratory analysis in a Python 3.12 virtual environment. Pinned the dependencies and verified them with `pip check`.
- Divided the workflow into data ingestion, image preprocessing, and exploratory analysis. Preprocessing decodes and validates 256 × 256 RGB PNGs. A full rerun after reorganizing the code reproduced all 12 EDA outputs byte for byte.

## Measured results

| Check | Result |
| --- | --- |
| Images with decoding or format errors | 0 of 19,064 |
| Images with a pixel standard deviation below 2 | 0 |
| Byte-identical duplicate groups, including across splits | 0 |
| Most frequent training class | Gnocchi, 744 images (17.19%) |
| Least frequent training class | Penne, 83 images (1.92%) |
| Largest-to-smallest class count ratio | 8.96 to 1 |

| Mean per-image statistic | Training | Test |
| --- | ---: | ---: |
| Brightness on the 0–255 intensity scale | 53.55 | 41.03 |
| Contrast measured as pixel intensity standard deviation | 35.90 | 28.31 |
| Mean absolute intensity difference between neighboring pixels | 12.23 | 8.13 |

On average, test images are darker and have lower contrast and smaller intensity changes between adjacent pixels. The effect on model performance will be measured during validation. The aggregate measurements come from the local EDA files `summary.json`, `class_distribution.csv`, and `domain_shift.csv`.

## Alien overlay observed in training

One reviewed training image contains an alien graphic that covers part of the galaxy scene. Its assigned class is Lasagna, and it passed the format and decoding checks. The prevalence of overlays by training class and in the test set is still unknown, as is their effect on validation performance.

An overlay can hide features needed to identify a galaxy. If it appears disproportionately in certain classes, a model may also use it as a shortcut for predicting the label. The current finding comes from visual inspection of this example. [Shortcut learning in deep neural networks](https://doi.org/10.1038/s42256-020-00257-z)

## Next validation steps

1. Create a reproducible stratified validation split from `train.csv` and train the first baseline on the full training portion, including flagged images.
2. Tag overlays in reviewed images, count them by training class, and estimate their frequency in a reviewed sample of test images.
3. Record overall validation weighted F1. For the reviewed validation images, report weighted F1 and sample counts for images with and without overlays. Use the same validation split for any subsequent masking or augmentation experiment.

The competition deliverables are a prediction CSV with the header `id,label`, a PDF report using the provided template, and executed notebooks with visible outputs. [Competition page](https://www.kaggle.com/competitions/an2dl2627)
