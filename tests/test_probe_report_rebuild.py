import csv
import json
from pathlib import Path

from PIL import Image

from seedvr2_tile.probe_report_rebuild import main


def test_report_only_rebuild_uses_existing_probe_cores(tmp_path: Path):
    root = tmp_path / "sweep"
    case = root / "reports" / "medium" / "source" / "p1-detail" / "noise-0"
    input_core = case / "inputs" / "pre-native.png"
    output_core = case / "results" / "medium__pre-native__scale-2x__noise-0.png"
    input_core.parent.mkdir(parents=True)
    output_core.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (40, 80, 120)).save(input_core)
    Image.new("RGB", (200, 200), (60, 100, 140)).save(output_core)

    manifest = {
        "mode": "post-preprocess-probe-tiles",
        "settings": {
            "pre_by_bucket": {"small": ["native"], "medium": ["native"], "large": [1.0]},
            "scales": [2.0],
            "noise_values": [0.0],
            "max_output_mp": 20.0,
            "tile": 1024,
            "overlap": 64,
            "strategy": "chess",
        },
        "sources": [
            {
                "source_id": "source",
                "source": "source.png",
                "width": 100,
                "height": 100,
                "megapixels": 1.5,
                "bucket": "medium",
                "selected": True,
                "probes": [
                    {
                        "probe_id": "p1-detail",
                        "label": "detail",
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "reference_pre_megapixels": "native",
                        "reference_tile_index": 0,
                        "brightness": 0.4,
                        "detail": 0.2,
                    }
                ],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "source",
                "bucket",
                "source_mp",
                "probe_id",
                "probe_label",
                "probe_center_x",
                "probe_center_y",
                "pre_mp",
                "scale",
                "noise",
                "processed_width",
                "processed_height",
                "total_tiles",
                "selected_tile_index",
                "group_resolution",
                "predicted_full_output_mp",
                "input_core",
                "output_core",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "source",
                "source": "source.png",
                "bucket": "medium",
                "source_mp": "1.5",
                "probe_id": "p1-detail",
                "probe_label": "detail",
                "probe_center_x": "0.5",
                "probe_center_y": "0.5",
                "pre_mp": "native",
                "scale": "2.0",
                "noise": "0.0",
                "processed_width": "100",
                "processed_height": "100",
                "total_tiles": "1",
                "selected_tile_index": "0",
                "group_resolution": "200",
                "predicted_full_output_mp": "6.0",
                "input_core": input_core.relative_to(root).as_posix(),
                "output_core": output_core.relative_to(root).as_posix(),
            }
        )

    assert main([str(root), "--comparison-crop-fraction", "0.4", "--cell-size", "160"]) == 0
    assert (case / "comparison.png").is_file()
    assert (case / "overview.png").is_file()

    crop_dir = case / "crops"
    input_crop = crop_dir / "01_pre-native__00-input.png"
    output_crop = crop_dir / "01_pre-native__01-scale-2x.png"
    assert input_crop.is_file()
    assert output_crop.is_file()
    with Image.open(input_crop) as opened:
        assert opened.size == (160, 160)
    with Image.open(output_crop) as opened:
        assert opened.size == (160, 160)

    scene = json.loads((case / "scene-window.json").read_text(encoding="utf-8"))
    assert scene["coordinate_space"] == "normalized-full-preprocessed-image"
    assert scene["requested_reference_core_fraction"] == 0.4
    assert scene["effective_reference_core_fraction"] <= 0.4
    assert scene["window"]["x0"] < 0.5 < scene["window"]["x1"]
    assert scene["window"]["y0"] < 0.5 < scene["window"]["y1"]

    html = (root / "index.html").read_text(encoding="utf-8")
    assert "Scene-registered detail view" in html
    assert "Whole probe-core overview" in html
    assert "Overlay crops" in html
