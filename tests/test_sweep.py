from pathlib import Path

from seedvr2_tile.sweep import (
    SourceImage,
    Variant,
    _parse_pre_values,
    _spread_indices,
    bucket_for_megapixels,
    build_variants,
    select_spread,
    variant_applies,
    variant_id,
)


def _source(source_id: str, mp: float, bucket: str = "medium") -> SourceImage:
    return SourceImage(
        source_id=source_id,
        source_path=Path(f"/tmp/{source_id}.png"),
        relative=Path(f"{source_id}.png"),
        width=1024,
        height=max(1, round(mp * 1024)),
        megapixels=mp,
        bucket=bucket,
        selected=True,
    )


def test_bucket_boundaries():
    assert bucket_for_megapixels(0.9, 1.25, 4.0) == "small"
    assert bucket_for_megapixels(1.25, 1.25, 4.0) == "medium"
    assert bucket_for_megapixels(4.0, 1.25, 4.0) == "medium"
    assert bucket_for_megapixels(4.01, 1.25, 4.0) == "large"


def test_spread_indices_cover_range():
    assert _spread_indices(7, 3) == [0, 3, 6]
    assert _spread_indices(3, 5) == [0, 1, 2]
    assert _spread_indices(7, 1) == [3]


def test_select_spread_uses_megapixel_order():
    items = [_source("c", 3.0), _source("a", 1.0), _source("b", 2.0)]
    assert select_spread(items, 2) == {"a", "c"}


def test_parse_pre_values_accepts_native():
    assert _parse_pre_values("native,0.5,1.0") == (None, 0.5, 1.0)


def test_variant_id_is_stable_and_readable():
    assert variant_id("large", 1.5, 2.0, 0.005) == "large__pre-1p5mp__scale-2x__noise-0p005"


def test_build_variants_is_cartesian_product():
    variants = build_variants("small", (None, 0.5), (1.5, 2.0), (0.0, 0.01))
    assert len(variants) == 8
    assert variants[0] == Variant(
        variant_id="small__pre-native__scale-1p5x__noise-0",
        bucket="small",
        pre_megapixels=None,
        scale=1.5,
        noise=0.0,
    )


def test_variant_skips_pre_upscale_and_excessive_output():
    source = _source("x", 0.6, bucket="small")
    pre_upscale = Variant("v1", "small", 0.75, 2.0, 0.0)
    applies, reason = variant_applies(source, pre_upscale, 20.0)
    assert not applies
    assert "would not downscale" in (reason or "")

    huge = Variant("v2", "small", None, 6.0, 0.0)
    applies, reason = variant_applies(source, huge, 20.0)
    assert not applies
    assert "exceeds cap" in (reason or "")

    good = Variant("v3", "small", 0.5, 2.0, 0.0)
    applies, reason = variant_applies(source, good, 20.0)
    assert applies
    assert reason is None
