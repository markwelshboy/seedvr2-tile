import json

import numpy as np
from PIL import Image

from seedvr2_tile.config import load_config
from seedvr2_tile.preprocess import PreprocessOptions, preprocess_image, target_size_for_megapixels


def test_preprocess_resize_and_noise_preserve_alpha_geometry():
    rgb = Image.new("RGB", (400, 200), (120, 120, 120))
    alpha = Image.new("L", (400, 200), 200)
    out_rgb, out_alpha, messages = preprocess_image(
        rgb,
        alpha,
        PreprocessOptions(megapixels=0.02, noise=0.05, noise_seed=123),
    )
    assert out_rgb.size == out_alpha.size
    assert out_rgb.size != rgb.size
    assert any(msg.startswith("pre=") for msg in messages)
    assert any(msg.startswith("noise=") for msg in messages)
    assert np.asarray(out_rgb).std() > 0


def test_target_megapixels_uses_comfy_1024_squared_definition():
    width, height = target_size_for_megapixels(2048, 1024, 1.0)
    assert width * height == 1024 * 1024 or abs(width * height - 1024 * 1024) < max(width, height)
    assert width / height == 2.0


def test_load_config_flattens_sections(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "io": {"format": "webp", "recursive": True},
                "preprocess": {"megapixels": 1.2, "noise": 0.03},
                "upscale": {"scale": 3},
                "tiling": {"tile": 768},
                "backend": {"attention-mode": "sageattn_2", "vae-tiled": True},
            }
        ),
        encoding="utf-8",
    )
    defaults = load_config(config_path)
    assert defaults["scale"] == 3
    assert defaults["tile"] == 768
    assert defaults["pre_megapixels"] == 1.2
    assert defaults["noise"] == 0.03
    assert defaults["attention_mode"] == "sageattn_2"
    assert defaults["vae_tiled"] is True
    assert defaults["format"] == "webp"
    assert defaults["recursive"] is True
