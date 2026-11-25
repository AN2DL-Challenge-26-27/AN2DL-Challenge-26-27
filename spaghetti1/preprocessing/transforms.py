"""Image transforms for training and inference.

Documentation: ``docs/05_AUGMENTATION_AND_TRANSFORMS.md``.

Pipeline for one image (uint8 RGB, 256 x 256):

1. Training only, geometric: random rotation by any angle (reflection padding, so no black
   corners), mirror, small scale and shift. Galaxy morphology does not depend on orientation.
2. Training only, photometric corruptions that reproduce what the test set shows. Fitting
   train/test duplicate pairs gave global gains of 0.6-1.2 with offsets, per-channel gains,
   and red/blue channel mixing. The test set is darker and also contains colour floods,
   coloured stripes, line streaks, noise, and blur.
3. Train and test, deterministic: optional denoising and optional per-image background
   normalisation (both off by default; switch them on as experiments).
4. Train and test: resize (optional) and normalise with the backbone's mean and std.

Every random choice uses ``torch`` RNG, so ``torch.manual_seed`` and DataLoader worker
seeding make augmentation reproducible.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic steps applied identically to training, validation, and test images."""

    image_size: int = 256
    denoise: str = "none"  # "none" | "bilateral" | "nlm"
    background_normalize: bool = False
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation. Probabilities are per image."""

    rotate: bool = True
    flip: bool = True
    scale_range: tuple[float, float] = (0.9, 1.1)
    max_shift: float = 0.04  # fraction of image size
    p_intensity: float = 0.8
    gain_range: tuple[float, float] = (0.6, 1.25)
    offset_range: tuple[float, float] = (-0.04, 0.05)
    p_channel_gain: float = 0.5
    channel_gain_range: tuple[float, float] = (0.75, 1.25)
    p_channel_mix: float = 0.25
    max_channel_mix: float = 0.35
    p_gamma: float = 0.3
    gamma_range: tuple[float, float] = (0.8, 1.25)
    p_noise: float = 0.3
    noise_sigma_range: tuple[float, float] = (0.005, 0.04)
    p_blur: float = 0.2
    blur_sigma_range: tuple[float, float] = (0.3, 1.2)
    p_artifact: float = 0.15


def _uniform(low: float, high: float, size: tuple[int, ...] = ()) -> torch.Tensor:
    return low + (high - low) * torch.rand(size)


def _chance(probability: float) -> bool:
    return probability > 0 and torch.rand(()).item() < probability


# --------------------------------------------------------------------------- geometric

def random_geometry(image: torch.Tensor, config: AugmentConfig) -> torch.Tensor:
    """Rotate/scale/shift a (3, H, W) float image, filling borders by reflection."""
    if config.flip and _chance(0.5):
        image = image.flip(-1)
    angle = _uniform(0, 360).item() if config.rotate else 0.0
    scale = _uniform(*config.scale_range).item()
    size = image.shape[-1]
    shift = [round(_uniform(-config.max_shift, config.max_shift).item() * size) for _ in range(2)]
    if angle == 0 and scale == 1 and shift == [0, 0]:
        return image
    # The crop's half-diagonal (size * sqrt(2) / 2) must stay inside the scaled, shifted padding.
    zoom = min(scale, 1.0)
    pad = math.ceil((size * math.sqrt(2) / zoom - size) / 2 + max(map(abs, shift)) / zoom) + 1
    pad = min(pad, size - 1)  # reflection padding must be smaller than the image
    padded = F.pad(image[None], (pad, pad, pad, pad), mode="reflect")[0]
    warped = TF.affine(padded, angle=angle, translate=shift, scale=scale, shear=[0.0, 0.0],
                       interpolation=TF.InterpolationMode.BILINEAR)
    return TF.center_crop(warped, [size, size])


# ------------------------------------------------------------------------ photometric

def random_photometric(image: torch.Tensor, config: AugmentConfig) -> torch.Tensor:
    """Simulate the brightness, colour, noise, and blur changes seen in the test set."""
    if _chance(config.p_channel_mix):
        mix = torch.eye(3)
        for target, source in ((0, 2), (2, 0), (0, 1), (2, 1)):  # red/blue mixing dominates in test
            mix[target, source] = _uniform(0, config.max_channel_mix).item() * (1.0 if source != 1 else 0.3)
        mix = mix / mix.sum(dim=1, keepdim=True)
        image = torch.einsum("ij,jhw->ihw", mix, image)
    if _chance(config.p_channel_gain):
        image = image * _uniform(*config.channel_gain_range, (3, 1, 1))
    if _chance(config.p_intensity):
        image = image * _uniform(*config.gain_range).item() + _uniform(*config.offset_range).item()
    if _chance(config.p_gamma):
        image = image.clamp(0, 1) ** _uniform(*config.gamma_range).item()
    if _chance(config.p_blur):
        sigma = _uniform(*config.blur_sigma_range).item()
        kernel = 2 * math.ceil(3 * sigma) + 1
        image = TF.gaussian_blur(image, [kernel, kernel], [sigma, sigma])
    if _chance(config.p_noise):
        sigma = _uniform(*config.noise_sigma_range).item()
        image = image + sigma * torch.randn_like(image)
    return image.clamp(0, 1)


def _random_colour() -> torch.Tensor:
    colour = torch.rand(3)
    colour[torch.randint(0, 3, ()).item()] = _uniform(0.7, 1.0).item()  # saturated, like the test artifacts
    return colour[:, None, None]


def random_artifact(image: torch.Tensor) -> torch.Tensor:
    """Add one synthetic overlay of a kind seen in the test set (never alien graphics)."""
    _, height, width = image.shape
    kind = torch.randint(0, 4, ()).item()
    image = image.clone()
    if kind == 0:  # semi-transparent coloured rectangle
        h, w = (torch.randint(16, 96, ()).item() for _ in range(2))
        y, x = torch.randint(0, height - h, ()).item(), torch.randint(0, width - w, ()).item()
        alpha = _uniform(0.4, 0.9).item()
        image[:, y:y + h, x:x + w] = (1 - alpha) * image[:, y:y + h, x:x + w] + alpha * _random_colour()
    elif kind == 1:  # thin straight streak
        canvas = np.zeros((height, width), np.uint8)
        start = (torch.randint(0, width, ()).item(), torch.randint(0, height, ()).item())
        angle = _uniform(0, 2 * math.pi).item()
        length = torch.randint(20, 140, ()).item()
        end = (int(start[0] + length * math.cos(angle)), int(start[1] + length * math.sin(angle)))
        cv2.line(canvas, start, end, 1, thickness=int(torch.randint(1, 4, ()).item()))
        mask = torch.from_numpy(canvas).bool()
        image[:, mask] = (0.3 * image[:, mask] + 0.7 * _random_colour()[:, :, 0]).clamp(0, 1)
    elif kind == 2:  # colour flood over one side of the image
        size = torch.randint(height // 5, height // 2, ()).item()
        alpha = _uniform(0.3, 0.7).item()
        side = torch.randint(0, 4, ()).item()
        region = [slice(None), slice(None), slice(None)]
        region[1 + side % 2] = slice(0, size) if side < 2 else slice(-size, None)
        image[tuple(region)] = (1 - alpha) * image[tuple(region)] + alpha * _random_colour()
    else:  # horizontal band with shifted brightness (read-out stripe)
        h = torch.randint(3, 20, ()).item()
        y = torch.randint(0, height - h, ()).item()
        image[:, y:y + h, :] = (image[:, y:y + h, :] * _uniform(0.3, 1.6).item()).clamp(0, 1)
    return image


# ----------------------------------------------------------------------- deterministic

def denoise(pixels: np.ndarray, method: str) -> np.ndarray:
    """Edge-preserving denoising of a uint8 RGB image."""
    if method == "none":
        return pixels
    if method == "bilateral":
        return cv2.bilateralFilter(pixels, d=5, sigmaColor=25, sigmaSpace=5)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(pixels, None, h=5, hColor=5, templateWindowSize=5, searchWindowSize=15)
    raise ValueError(f"Unknown denoise method: {method}")


def background_normalize(image: torch.Tensor) -> torch.Tensor:
    """Map each channel's background (median) to 0.1 and the brightest 0.5% to 0.9.

    The same scale is used for all channels, so the galaxy's colour ratios survive while
    global brightness differences and background colour casts are removed.
    """
    background = image.flatten(1).median(dim=1).values[:, None, None]
    luminance = image.mean(dim=0)
    bright = torch.quantile(luminance.flatten(), 0.995)
    scale = (bright - background.mean()).clamp_min(0.02)
    return (0.1 + 0.8 * (image - background) / scale).clamp(0, 1)


class GalaxyTransform:
    """Callable mapping a uint8 (H, W, 3) array to a normalised float tensor (3, S, S)."""

    def __init__(self, preprocess: PreprocessConfig = PreprocessConfig(), augment: AugmentConfig | None = None):
        self.preprocess = preprocess
        self.augment = augment
        self.mean = torch.tensor(preprocess.mean)[:, None, None]
        self.std = torch.tensor(preprocess.std)[:, None, None]

    def to_unit(self, pixels: np.ndarray) -> torch.Tensor:
        """Augment and preprocess, returning a [0, 1] image before mean/std normalisation."""
        image = torch.tensor(pixels).permute(2, 0, 1).float() / 255.0  # copies, so cached arrays stay intact
        if self.augment is not None:
            image = random_geometry(image, self.augment)
            image = random_photometric(image, self.augment)
            if _chance(self.augment.p_artifact):
                image = random_artifact(image)
        if self.preprocess.denoise != "none":
            pixels = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            image = torch.from_numpy(denoise(pixels, self.preprocess.denoise)).permute(2, 0, 1).float() / 255.0
        if self.preprocess.background_normalize:
            image = background_normalize(image)
        if image.shape[-1] != self.preprocess.image_size:
            image = TF.resize(image, [self.preprocess.image_size] * 2, antialias=True)
        return image

    def __call__(self, pixels: np.ndarray) -> torch.Tensor:
        return (self.to_unit(pixels) - self.mean) / self.std


def build_transforms(
    preprocess: PreprocessConfig = PreprocessConfig(), augment: AugmentConfig = AugmentConfig()
) -> tuple[GalaxyTransform, GalaxyTransform]:
    """Return (training transform, evaluation transform) sharing the same preprocessing."""
    return GalaxyTransform(preprocess, augment), GalaxyTransform(preprocess, None)
