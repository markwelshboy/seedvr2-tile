import numpy as np
from PIL import Image

from seedvr2_tile.probe_scene_sweep import _crop_normalized, _tiles_for_window
from seedvr2_tile.stitching import Stitcher
from seedvr2_tile.tiling import make_tiles


def _gradient(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = x
    arr[..., 1] = y
    arr[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_scene_window_crossing_core_boundary_selects_both_tiles():
    image = _gradient(2048, 1024)
    pairs = make_tiles(
        image,
        image_index=0,
        tile_width=1024,
        tile_height=1024,
        padding=64,
        strategy="linear",
    )
    window = (0.47, 0.35, 0.53, 0.65)
    selected = _tiles_for_window(pairs, window, image.width, image.height)
    assert {spec.tile_index for spec, _ in selected} == {0, 1}


def test_scene_window_inside_core_avoids_unneeded_neighbor():
    image = _gradient(2048, 1024)
    pairs = make_tiles(
        image,
        image_index=0,
        tile_width=1024,
        tile_height=1024,
        padding=64,
        strategy="linear",
    )
    window = (0.15, 0.35, 0.25, 0.65)
    selected = _tiles_for_window(pairs, window, image.width, image.height)
    assert [spec.tile_index for spec, _ in selected] == [0]


def test_partial_stitch_reconstructs_boundary_scene_from_identity_tiles():
    image = _gradient(2048, 1024)
    pairs = make_tiles(
        image,
        image_index=0,
        tile_width=1024,
        tile_height=1024,
        padding=64,
        strategy="linear",
    )
    window = (0.47, 0.35, 0.53, 0.65)
    selected = _tiles_for_window(pairs, window, image.width, image.height)

    stitcher = Stitcher(image.width, image.height, method="multiband")
    for spec, tile in selected:
        stitcher.add(spec, tile, scale=1.0)
    reconstructed = _crop_normalized(stitcher.finish(), window)
    expected = _crop_normalized(image, window)

    a = np.asarray(reconstructed, dtype=np.int16)
    b = np.asarray(expected, dtype=np.int16)
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 1.0
