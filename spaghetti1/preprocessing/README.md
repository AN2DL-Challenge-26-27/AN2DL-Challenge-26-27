# Image preprocessing

The `read_rgb_png` function in `images.py` opens an image, validates that it is a 256 × 256 RGB PNG, and returns a `uint8` array of shape `(256, 256, 3)`. The array stays in memory and preserves the original pixel values.

The `python -m spaghetti1 eda` command uses this function to scan and display images. Training-specific normalization and augmentation are planned for the model development stage. [Setup and usage](../../docs/SETUP.md)
