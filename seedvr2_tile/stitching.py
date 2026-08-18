from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .tiling import TileSpec


def _resize_float(image: Image.Image, width: int, height: int) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float32) / 255.0


def _edge_mask(
    width: int,
    height: int,
    *,
    fade_left: int,
    fade_top: int,
    fade_right: int,
    fade_bottom: int,
) -> np.ndarray:
    x = np.ones(width, dtype=np.float32)
    y = np.ones(height, dtype=np.float32)

    def ramp(n: int) -> np.ndarray:
        if n <= 0:
            return np.ones(0, dtype=np.float32)
        t = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        return 0.5 - 0.5 * np.cos(np.pi * t)

    if fade_left:
        x[:fade_left] = ramp(fade_left)
    if fade_right:
        x[-fade_right:] = ramp(fade_right)[::-1]
    if fade_top:
        y[:fade_top] = ramp(fade_top)
    if fade_bottom:
        y[-fade_bottom:] = ramp(fade_bottom)[::-1]
    return y[:, None] * x[None, :]


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gp = [img]
    for _ in range(levels):
        if min(gp[-1].shape[:2]) < 8:
            break
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gp = _gaussian_pyramid(img, levels)
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        up = cv2.pyrUp(gp[i], dstsize=(gp[i - 1].shape[1], gp[i - 1].shape[0]))
        lp.append(gp[i - 1] - up)
    return lp


def _laplacian_blend(a: np.ndarray, b: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    la = _laplacian_pyramid(a, levels)
    lb = _laplacian_pyramid(b, levels)
    gm = _gaussian_pyramid(mask.astype(np.float32), len(la) - 1)
    gm = list(reversed(gm))

    blended: list[np.ndarray] = []
    for aa, bb, mm in zip(la, lb, gm):
        if mm.ndim == 2:
            mm = mm[..., None]
        if mm.shape[:2] != aa.shape[:2]:
            mm = cv2.resize(mm, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_LINEAR)
            if mm.ndim == 2:
                mm = mm[..., None]
        blended.append(aa * (1.0 - mm) + bb * mm)

    out = blended[0]
    for band in blended[1:]:
        out = cv2.pyrUp(out, dstsize=(band.shape[1], band.shape[0])) + band
    return np.clip(out, 0.0, 1.0)


def _content_mask(existing: np.ndarray, incoming: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    def sharpness(x: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(np.clip(x * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 1.0)

    ea = sharpness(existing)
    ib = sharpness(incoming)
    preference = ib / (ea + ib + 1e-6)
    return np.clip(base_mask * 0.7 + preference * 0.3, 0.0, 1.0)


class Stitcher:
    def __init__(self, width: int, height: int, method: str = "multiband") -> None:
        self.width = width
        self.height = height
        self.method = method
        self.canvas = np.zeros((height, width, 3), dtype=np.float32)
        self.weights = np.zeros((height, width), dtype=np.float32)

    def add(self, spec: TileSpec, processed: Image.Image, *, scale: float) -> None:
        process_w, process_h = spec.process_size
        out_process_w = max(1, round(process_w * scale))
        out_process_h = max(1, round(process_h * scale))
        tile = _resize_float(processed, out_process_w, out_process_h)

        pad_l, pad_t, _, _ = spec.synthetic_padding
        sx0, sy0, sx1, sy1 = spec.source_box
        desired_x0 = spec.core_box[0] - (spec.core_box[0] - sx0) - pad_l
        desired_y0 = spec.core_box[1] - (spec.core_box[1] - sy0) - pad_t

        crop_x0 = round((sx0 - desired_x0) * scale)
        crop_y0 = round((sy0 - desired_y0) * scale)
        crop_x1 = round((sx1 - desired_x0) * scale)
        crop_y1 = round((sy1 - desired_y0) * scale)
        tile = tile[crop_y0:crop_y1, crop_x0:crop_x1]

        ox0, oy0 = round(sx0 * scale), round(sy0 * scale)
        ox1, oy1 = round(sx1 * scale), round(sy1 * scale)
        ox1 = min(ox1, self.width)
        oy1 = min(oy1, self.height)
        target_w, target_h = ox1 - ox0, oy1 - oy0
        if target_w <= 0 or target_h <= 0:
            return
        if tile.shape[1] != target_w or tile.shape[0] != target_h:
            tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        cx0, cy0, cx1, cy1 = spec.core_box
        fade_left = max(0, round((cx0 - sx0) * scale))
        fade_top = max(0, round((cy0 - sy0) * scale))
        fade_right = max(0, round((sx1 - cx1) * scale))
        fade_bottom = max(0, round((sy1 - cy1) * scale))
        mask = _edge_mask(
            target_w,
            target_h,
            fade_left=fade_left,
            fade_top=fade_top,
            fade_right=fade_right,
            fade_bottom=fade_bottom,
        )

        roi = self.canvas[oy0:oy1, ox0:ox1]
        old_w = self.weights[oy0:oy1, ox0:ox1]

        if self.method in {"linear", "simple"}:
            weight = np.ones_like(mask) if self.method == "simple" else mask
            roi += tile * weight[..., None]
            old_w += weight
            return

        covered = old_w > 1e-6
        if not np.any(covered):
            roi[:] = tile
            old_w[:] = np.maximum(mask, 1e-6)
            return

        existing = roi / np.maximum(old_w[..., None], 1e-6)
        blend_mask = mask.copy()
        blend_mask[~covered] = 1.0
        if self.method == "content-aware":
            blend_mask = _content_mask(existing, tile, blend_mask)
            blend_mask[~covered] = 1.0

        blended = _laplacian_blend(existing, tile, blend_mask, levels=4)
        if self.method == "bilateral":
            smooth = cv2.bilateralFilter(
                np.clip(blended * 255.0, 0, 255).astype(np.uint8), 5, 20, 5
            ).astype(np.float32) / 255.0
            seam = ((blend_mask > 0.02) & (blend_mask < 0.98))[..., None]
            blended = np.where(seam, smooth, blended)

        roi[:] = blended
        old_w[:] = 1.0

    def finish(self) -> Image.Image:
        out = self.canvas / np.maximum(self.weights[..., None], 1e-6)
        return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
