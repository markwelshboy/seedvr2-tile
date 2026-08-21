import numpy as np
from PIL import Image

from seedvr2_tile.stitching import Stitcher
from seedvr2_tile.tiling import make_tiles


def test_identity_tiles_stitch_back_to_source_shape():
    yy, xx = np.mgrid[0:300, 0:500]
    arr = np.stack([(xx % 256), (yy % 256), ((xx + yy) % 256)], axis=-1).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")
    pairs = make_tiles(image, image_index=0, tile_width=192, tile_height=192, padding=32, strategy="chess")
    stitcher = Stitcher(500, 300, method="linear")
    for spec, tile in pairs:
        stitcher.add(spec, tile, scale=1.0)
    out = np.asarray(stitcher.finish()).astype(np.int16)
    assert out.shape == arr.shape
    assert np.abs(out - arr.astype(np.int16)).mean() < 1.0
