from pathlib import Path

import numpy as np
from PIL import Image

from seedvr2_tile.probe_sweep import (
    Probe,
    ProbeCandidate,
    _choose_distinct_probe_candidates,
    _inference_filename,
    _nearest_tile,
    _reference_pre,
    extract_core,
)
from seedvr2_tile.sweep import SourceImage
from seedvr2_tile.tiling import TileSpec


def _spec(index: int, core_box=(0, 0, 10, 10), source_box=(0, 0, 10, 10), padding=(1, 1, 1, 1)) -> TileSpec:
    return TileSpec(
        image_index=0,
        tile_index=index,
        core_box=core_box,
        source_box=source_box,
        process_size=(12, 12),
        synthetic_padding=padding,
        filename=f"tile-{index}.png",
    )


def _source(mp: float) -> SourceImage:
    return SourceImage(
        source_id="source",
        source_path=Path("source.png"),
        relative=Path("source.png"),
        width=3000,
        height=4000,
        megapixels=mp,
        bucket="large",
        selected=True,
    )


def test_probe_selection_prefers_distinct_detail_dark_center_tiles():
    candidates = [
        ProbeCandidate(_spec(0), 0.25, 0.25, brightness=0.50, detail=0.20),
        ProbeCandidate(_spec(1), 0.75, 0.25, brightness=0.10, detail=0.10),
        ProbeCandidate(_spec(2), 0.25, 0.75, brightness=0.60, detail=0.90),
        ProbeCandidate(_spec(3), 0.75, 0.75, brightness=0.40, detail=0.30),
    ]
    selected = _choose_distinct_probe_candidates(candidates, 3)
    assert [(label, item.spec.tile_index) for label, item in selected] == [
        ("detail", 2),
        ("dark", 1),
        ("center", 0),
    ]


def test_probe_count_collapses_to_available_tiles():
    candidates = [ProbeCandidate(_spec(0), 0.5, 0.5, brightness=0.5, detail=0.5)]
    selected = _choose_distinct_probe_candidates(candidates, 3)
    assert len(selected) == 1
    assert selected[0][1].spec.tile_index == 0


def test_reference_pre_uses_highest_resolution_valid_candidate():
    source = _source(11.4)
    assert _reference_pre(source, (1.0, 1.5, 2.0)) == 2.0
    assert _reference_pre(source, (None, 1.0, 2.0)) is None


def test_nearest_tile_maps_normalized_probe_to_candidate_grid():
    image = Image.new("RGB", (200, 100))
    left = _spec(0, core_box=(0, 0, 100, 100), source_box=(0, 0, 100, 100), padding=(0, 0, 0, 0))
    right = _spec(1, core_box=(100, 0, 200, 100), source_box=(100, 0, 200, 100), padding=(0, 0, 0, 0))
    pairs = [(left, image.crop(left.core_box)), (right, image.crop(right.core_box))]
    probe = Probe("p1-detail", "detail", 0.8, 0.5, None, 1, 0.5, 0.5)
    selected, _ = _nearest_tile(probe, image, pairs)
    assert selected.tile_index == 1


def test_extract_core_removes_padding_at_requested_scale():
    spec = _spec(0)
    processed = Image.fromarray(np.full((24, 24, 3), 128, dtype=np.uint8))
    core = extract_core(spec, processed, scale=2.0)
    assert core.size == (20, 20)


def test_inference_filename_dedup_key_ignores_output_scale_but_tracks_resolution():
    source = _source(11.4)
    a = _inference_filename(source=source, pre_mp=1.0, noise=0.0, tile_index=2, resolution=2048)
    b = _inference_filename(source=source, pre_mp=1.0, noise=0.0, tile_index=2, resolution=2048)
    c = _inference_filename(source=source, pre_mp=1.0, noise=0.0, tile_index=2, resolution=1728)
    assert a == b
    assert a != c
