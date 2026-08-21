from types import SimpleNamespace

import pytest

from seedvr2_tile.naming import option_values, render_output_stem, validate_output_template


def _args(**overrides):
    values = dict(
        scale=2.0, long_edge=None, short_edge=None,
        fbcnn_enabled=False, fbcnn_quality="auto", fbcnn_device="auto",
        pre_megapixels=None, pre_resample="lanczos", noise=0.0, noise_seed=None,
        tile=1024, tile_width=None, tile_height=None, overlap=64,
        tile_upscale_resolution=2048, strategy="chess", blend="multiband",
        format="png", quality=95, seed=42, cuda_device=None,
        attention_mode="sdpa", color_correction="lab", blocks_to_swap=0,
        swap_io_components=False, dit_offload_device="none", vae_offload_device="none",
        tensor_offload_device="cpu", vae_tiled=False, vae_tile_size=1024,
        vae_tile_overlap=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_configured_template_uses_option_names():
    values = option_values(_args(pre_megapixels=1.0), model_label="3b-fp16")
    stem = render_output_stem(
        basename="portrait",
        mode="on",
        template="#basename#_#model#_#scale#_#pre-megapixels#",
        values=values,
    )
    assert stem == "portrait_3b-fp16_2_1"


def test_unknown_template_field_is_rejected():
    with pytest.raises(ValueError, match="unknown output-template"):
        validate_output_template("#basename#_#made-up#")


def test_delta_only_appends_non_default_processing_values():
    values = option_values(_args(pre_megapixels=1.0, fbcnn_enabled=True), model_label="3b")
    stem = render_output_stem(basename="portrait", mode="delta", template=None, values=values)
    assert stem.startswith("portrait_")
    assert "pre-megapixels-1" in stem
    assert "fbcnn-on" in stem
    assert "scale-2" not in stem
    assert "model-3b" not in stem


def test_all_appends_defaults_too():
    values = option_values(_args(), model_label="3b")
    stem = render_output_stem(basename="portrait", mode="all", template=None, values=values)
    assert "m3b" in stem
    assert "s2" in stem
    assert "t1024" in stem
    assert "vto128" in stem
    assert len(stem) <= 220
