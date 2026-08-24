import numpy as np
from PIL import Image

from seedvr2_tile.probe_sweep_report import _detail_fit_cell


def test_detail_fit_cell_center_crops_and_fills_square():
    # Tall image with distinct top/center/bottom bands. A 50% center crop should
    # exclude the extremes and fill the output square without white letterboxing.
    arr = np.zeros((200, 100, 3), dtype=np.uint8)
    arr[:50] = (255, 0, 0)
    arr[50:150] = (0, 255, 0)
    arr[150:] = (0, 0, 255)
    image = Image.fromarray(arr, mode="RGB")

    rendered = _detail_fit_cell(image, 120)
    out = np.asarray(rendered)

    assert rendered.size == (120, 120)
    assert out[..., 1].mean() > 240
    assert out[..., 0].mean() < 10
    assert out[..., 2].mean() < 10
