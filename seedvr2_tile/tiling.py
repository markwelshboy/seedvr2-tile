from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileSpec:
    image_index: int
    tile_index: int
    core_box: tuple[int, int, int, int]
    source_box: tuple[int, int, int, int]
    process_size: tuple[int, int]
    synthetic_padding: tuple[int, int, int, int]
    filename: str


def _positions(width: int, height: int, tile_w: int, tile_h: int, strategy: str) -> Iterable[tuple[int, int]]:
    coords = [(x, y) for y in range(0, height, tile_h) for x in range(0, width, tile_w)]
    if strategy == "chess":
        even, odd = [], []
        for x, y in coords:
            cell = (x // tile_w) + (y // tile_h)
            (even if cell % 2 == 0 else odd).append((x, y))
        return [*even, *odd]
    return coords


def make_tiles(
    image: Image.Image,
    *,
    image_index: int,
    tile_width: int,
    tile_height: int,
    padding: int,
    strategy: str,
) -> list[tuple[TileSpec, Image.Image]]:
    """Create fixed-size reflection-padded processing tiles.

    The core grid itself does not overlap. ``padding`` adds real neighboring image
    context around each core tile; the padded contexts overlap during stitching.
    Synthetic reflection is used only where the requested processing rectangle
    extends beyond the source image or past a partial final core tile.
    """
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if padding < 0:
        raise ValueError("padding must be >= 0")

    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    process_w = tile_width + 2 * padding
    process_h = tile_height + 2 * padding
    result: list[tuple[TileSpec, Image.Image]] = []

    for tile_index, (x, y) in enumerate(_positions(w, h, tile_width, tile_height, strategy)):
        core_w = min(tile_width, w - x)
        core_h = min(tile_height, h - y)
        core_box = (x, y, x + core_w, y + core_h)

        desired_x0 = x - padding
        desired_y0 = y - padding
        desired_x1 = x + tile_width + padding
        desired_y1 = y + tile_height + padding

        sx0 = max(0, desired_x0)
        sy0 = max(0, desired_y0)
        sx1 = min(w, desired_x1)
        sy1 = min(h, desired_y1)
        source_box = (sx0, sy0, sx1, sy1)

        crop = arr[sy0:sy1, sx0:sx1]
        pad_left = sx0 - desired_x0
        pad_top = sy0 - desired_y0
        pad_right = desired_x1 - sx1
        pad_bottom = desired_y1 - sy1

        # Partial right/bottom core tiles are intentionally padded out to the
        # normal processing footprint so every tile has identical dimensions.
        pad_mode = "reflect" if crop.shape[0] > 1 and crop.shape[1] > 1 else "edge"
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=pad_mode,
        )
        if crop.shape[1] != process_w or crop.shape[0] != process_h:
            raise RuntimeError(
                f"internal tile size error: got {crop.shape[1]}x{crop.shape[0]}, "
                f"expected {process_w}x{process_h}"
            )

        filename = f"i{image_index:06d}_t{tile_index:06d}.png"
        spec = TileSpec(
            image_index=image_index,
            tile_index=tile_index,
            core_box=core_box,
            source_box=source_box,
            process_size=(process_w, process_h),
            synthetic_padding=(pad_left, pad_top, pad_right, pad_bottom),
            filename=filename,
        )
        result.append((spec, Image.fromarray(crop, mode="RGB")))

    return result


def save_tiles(tiles: list[tuple[TileSpec, Image.Image]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for spec, image in tiles:
        image.save(directory / spec.filename, format="PNG")
