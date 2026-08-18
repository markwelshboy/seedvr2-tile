from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessOptions:
    megapixels: float | None = None
    resample: str = "lanczos"
    noise: float = 0.0
    noise_seed: int | None = None


_RESAMPLE = {
    "nearest-exact": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "area": Image.Resampling.BOX,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def target_size_for_megapixels(width: int, height: int, megapixels: float) -> tuple[int, int]:
    """Match ComfyUI Scale Image to Total Pixels semantics (MP = 1024 * 1024 pixels)."""
    if megapixels <= 0:
        raise ValueError("megapixels must be > 0")
    total = megapixels * 1024 * 1024
    scale = math.sqrt(total / (width * height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def add_gaussian_noise(image: Image.Image, amount: float, seed: int) -> Image.Image:
    """Match ComfyUI ImageAddNoise: clip(image + strength * N(0, 1), 0, 1)."""
    if amount < 0 or amount > 1:
        raise ValueError("noise amount must be in the range [0, 1]")
    if amount == 0:
        return image.copy()
    arr = np.asarray(image).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noisy = np.clip(arr + amount * rng.standard_normal(arr.shape, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray(np.round(noisy * 255.0).astype(np.uint8), mode=image.mode)


def preprocess_image(
    image: Image.Image,
    alpha: Image.Image | None,
    options: PreprocessOptions,
    *,
    image_index: int = 0,
    base_seed: int = 0,
) -> tuple[Image.Image, Image.Image | None, list[str]]:
    """Resize first, then add noise to RGB only. Alpha follows resize but never receives noise."""
    rgb = image.copy()
    a = alpha.copy() if alpha is not None else None
    messages: list[str] = []

    if options.megapixels is not None:
        target_w, target_h = target_size_for_megapixels(rgb.width, rgb.height, options.megapixels)
        if (target_w, target_h) != rgb.size:
            filt = _RESAMPLE[options.resample]
            rgb = rgb.resize((target_w, target_h), filt)
            if a is not None:
                a = a.resize((target_w, target_h), Image.Resampling.LANCZOS)
        messages.append(f"pre={target_w}x{target_h} ({options.megapixels:.3f} MP target)")

    if options.noise:
        seed = (options.noise_seed if options.noise_seed is not None else base_seed) + image_index
        rgb = add_gaussian_noise(rgb, options.noise, seed)
        messages.append(f"noise={options.noise:.4f} seed={seed}")

    return rgb, a, messages
