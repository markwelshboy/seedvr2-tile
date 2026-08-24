import csv
import json
from pathlib import Path

from PIL import Image

from seedvr2_tile.full_sweep_report import _compact_run, _is_singleton_recipe


def _manifest(*, scales=(3.0,), noises=(0.0,), pre=(1.0,)) -> dict:
    return {
        "seedvr2_tile_version": "test",
        "settings": {
            "scales": list(scales),
            "noise_values": list(noises),
            "pixel_noise_values": list(noises),
            "input_noise_scale": 0.0,
            "latent_noise_scale": 0.03,
            "pre_by_bucket": {"small": [], "medium": [], "large": list(pre)},
        },
        "sources": [
            {
                "source_id": "img0001",
                "source": "photo.png",
                "width": 3000,
                "height": 4000,
                "megapixels": 11.44,
                "bucket": "large",
                "selected": True,
            }
        ],
        "failures": [],
    }


def test_singleton_recipe_requires_one_value_on_each_active_axis():
    assert _is_singleton_recipe(_manifest())
    assert not _is_singleton_recipe(_manifest(scales=(2.0, 3.0)))
    assert not _is_singleton_recipe(_manifest(noises=(0.0, 0.01)))
    assert not _is_singleton_recipe(_manifest(pre=(1.0, 1.5)))


def test_compact_run_hides_comparison_arrays_in_details(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    manifest = _manifest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_rel = Path("results/large/recipe/img0001.png")
    output_path = root / output_rel
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (1200, 800), "white").save(output_path)

    fields = [
        "source_id", "source", "bucket", "source_mp", "pre_mp", "scale", "noise",
        "predicted_output_mp", "status", "reason", "output", "actual_output_mp",
        "pixel_noise", "input_noise_scale", "latent_noise_scale",
    ]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "img0001",
                "source": "photo.png",
                "bucket": "large",
                "source_mp": "11.44",
                "pre_mp": "1.000000",
                "scale": "3.000000",
                "noise": "0.000000",
                "predicted_output_mp": "9.0",
                "status": "complete",
                "reason": "",
                "output": output_rel.as_posix(),
                "actual_output_mp": "9.0",
                "pixel_noise": "0.000000",
                "input_noise_scale": "0.000000",
                "latent_noise_scale": "0.030000",
            }
        )

    report = root / "reports/large/img0001/noise-0/full.png"
    report.parent.mkdir(parents=True)
    Image.new("RGB", (320, 200), "gray").save(report)

    assert _compact_run(root)
    html = (root / "index.html").read_text(encoding="utf-8")
    assert "single-recipe production run" in html
    assert "report-previews/img0001.png" in html
    assert "Open full-resolution output" in html
    assert "<details><summary>Comparison views</summary>" in html
    assert "reports/large/img0001/noise-0/full.png" in html
    assert (root / "report-previews/img0001.png").is_file()


def test_compact_run_leaves_multi_variant_report_alone(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(_manifest(scales=(2.0, 3.0))), encoding="utf-8")
    sentinel = "ORIGINAL COMPARISON REPORT"
    (root / "index.html").write_text(sentinel, encoding="utf-8")
    assert not _compact_run(root)
    assert (root / "index.html").read_text(encoding="utf-8") == sentinel
