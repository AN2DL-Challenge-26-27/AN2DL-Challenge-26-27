"""Per-image quality checks: alien overlays, blank frames, and noise level.

All thresholds were calibrated on the full dataset (19,064 images) and cross-checked
against a DINOv2 linear probe and visual review. Documentation: ``docs/03_PREPROCESSING.md``.
"""

import cv2
import numpy as np

# Alien overlay: a pale beige "skin" region with dark eyes inside its outline.
# OpenCV HSV ranges (H in 0-179, S and V in 0-255).
ALIEN_SKIN_HUE = (20, 30)
ALIEN_SKIN_SATURATION = (30, 62)
ALIEN_SKIN_MIN_VALUE = 150
ALIEN_EYE_MAX_VALUE = 90
# Measured: aliens have >= 2,698 skin pixels and >= 342 eye pixels; every other image
# has 0 eye pixels. Bright yellow galaxies can reach 4,680 skin pixels but have no eyes.
ALIEN_MIN_SKIN_PIXELS = 1000
ALIEN_MIN_EYE_PIXELS = 200

# Blank frame: flat grey noise with no object. Blank frames measure ~7, real images >= 26.
BLANK_MAX_CONTRAST = 15.0

# Noise: robust sigma of the residual after a 3x3 median filter (0-255 scale).
# The test set's 99th percentile is ~13, so higher values are heavier than test noise.
NOISY_MIN_SIGMA = 12.0

_CLOSE_KERNEL = np.ones((5, 5), np.uint8)


def _robust_sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def alien_features(pixels: np.ndarray) -> tuple[int, int]:
    """Return (largest skin-region size, dark pixels inside that region's convex hull)."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., channel] for channel in range(3))
    skin = (
        (hue >= ALIEN_SKIN_HUE[0]) & (hue <= ALIEN_SKIN_HUE[1])
        & (saturation >= ALIEN_SKIN_SATURATION[0]) & (saturation <= ALIEN_SKIN_SATURATION[1])
        & (value > ALIEN_SKIN_MIN_VALUE)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    if count < 2:
        return 0, 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    skin_pixels = int(stats[largest, cv2.CC_STAT_AREA])
    if skin_pixels < ALIEN_MIN_SKIN_PIXELS:
        return skin_pixels, 0

    region = cv2.morphologyEx((labels == largest).astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(region)
    cv2.drawContours(hull, [cv2.convexHull(np.vstack(contours))], -1, 1, thickness=-1)
    eyes = (hull == 1) & (region == 0) & (value < ALIEN_EYE_MAX_VALUE)
    return skin_pixels, int(eyes.sum())


def measure_quality(pixels: np.ndarray) -> dict:
    """Compute quality flags for one uint8 RGB image of shape (H, W, 3)."""
    skin_pixels, eye_pixels = alien_features(pixels)
    values = pixels.astype(np.float32)
    background = np.median(values.reshape(-1, 3), axis=0)
    contrast = float(np.percentile(values, 99) - background.mean())
    residual = (values - cv2.medianBlur(pixels, 3).astype(np.float32)).mean(axis=2)
    noise_sigma = _robust_sigma(residual)
    return {
        "alien_skin_px": skin_pixels,
        "alien_eye_px": eye_pixels,
        "is_alien": skin_pixels >= ALIEN_MIN_SKIN_PIXELS and eye_pixels >= ALIEN_MIN_EYE_PIXELS,
        "contrast": contrast,
        "is_blank": contrast < BLANK_MAX_CONTRAST,
        "noise_sigma": noise_sigma,
        "is_noisy": noise_sigma >= NOISY_MIN_SIGMA,
        "background_r": float(background[0]),
        "background_g": float(background[1]),
        "background_b": float(background[2]),
    }
