import csv
import json
from pathlib import Path

from PIL import Image

from seedvr2_tile.interactive_probe_report import write_interactive_report


def test_interactive_report_syncs_probe_selectors_by_source_recipe(tmp_path: Path):
    root = tmp_path / "sweep"
    root.mkdir()
    manifest = {
        "model": "3b",
        "mode": "post-preprocess-probe-scenes-v2",
        "settings": {
            "pre_by_bucket": {"small": [], "medium": [], "large": [1.0]},
            "scales": [3.0],
            "noise_values": [0.0],
            "pixel_noise_values": [0.0],
            "input_noise_scale": 0.0,
            "latent_noise_scale": 0.03,
            "seed": 42,
            "tile": 1024,
            "overlap": 64,
            "tile_upscale_resolution": 2048,
            "strategy": "chess",
            "attention_mode": "sdpa",
            "color_correction": "lab",
        },
        "sources": [
            {
                "source_id": "src1",
                "source": "photo.png",
                "width": 3000,
                "height": 4000,
                "megapixels": 11.44,
                "bucket": "large",
                "selected": True,
                "probes": [
                    {"probe_id": "p1-detail", "label": "detail", "center_x": 0.7, "center_y": 0.3},
                    {"probe_id": "p2-dark", "label": "dark", "center_x": 0.2, "center_y": 0.8},
                ],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "index.html").write_text("<html><body>old comparison</body></html>", encoding="utf-8")

    fieldnames = [
        "source_id", "source", "bucket", "source_mp", "probe_id", "probe_label",
        "pre_mp", "scale", "noise", "pixel_noise", "input_noise_scale", "latent_noise_scale",
        "output_scene",
    ]
    rows = []
    for probe_id, label in (("p1-detail", "detail"), ("p2-dark", "dark")):
        crop = root / "reports" / "large" / "src1" / probe_id / "noise-0" / "crops" / "01_pre-1mp__01-scale-3x.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "gray").save(crop)
        rows.append(
            {
                "source_id": "src1",
                "source": "photo.png",
                "bucket": "large",
                "source_mp": "11.44",
                "probe_id": probe_id,
                "probe_label": label,
                "pre_mp": "1.0",
                "scale": "3.0",
                "noise": "0.0",
                "pixel_noise": "0.0",
                "input_noise_scale": "0.0",
                "latent_noise_scale": "0.03",
                "output_scene": "",
            }
        )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    assert write_interactive_report(root)
    text = (root / "index.html").read_text(encoding="utf-8")
    assert "Download selected recipes" in text
    assert "seedvr2-selection-v1" in text
    assert "p1-detail" in text and "p2-dark" in text
    assert text.count("data-source-id='src1'") == 2
    assert "latent=0.03" in text
    assert (root / "comparison-index.html").read_text(encoding="utf-8") == "<html><body>old comparison</body></html>"
