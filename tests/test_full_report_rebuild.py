import csv
import json
from pathlib import Path

from PIL import Image

from seedvr2_tile.full_report_rebuild import main


def test_full_report_rebuild_compacts_singleton_without_inference(tmp_path: Path):
    root = tmp_path / "full"
    root.mkdir()
    output_rel = Path("results/large/large__pre-1mp__scale-3x__noise-0/src1.png")
    output = root / output_rel
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), "gray").save(output)

    manifest = {
        "seedvr2_tile_version": "test",
        "model": "3b",
        "settings": {
            "pre_by_bucket": {"small": [], "medium": [], "large": [1.0]},
            "scales": [3.0],
            "noise_values": [0.0],
            "pixel_noise_values": [0.0],
            "input_noise_scale": 0.0,
            "latent_noise_scale": 0.03,
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
            }
        ],
        "failures": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fields = [
        "source_id", "source", "bucket", "source_mp", "pre_mp", "scale", "noise",
        "predicted_output_mp", "status", "reason", "output", "actual_output_mp",
        "pixel_noise", "input_noise_scale", "latent_noise_scale",
    ]
    row = {
        "source_id": "src1",
        "source": "photo.png",
        "bucket": "large",
        "source_mp": "11.44",
        "pre_mp": "1.0",
        "scale": "3.0",
        "noise": "0.0",
        "predicted_output_mp": "9.0",
        "status": "complete",
        "reason": "",
        "output": output_rel.as_posix(),
        "actual_output_mp": "0.073",
        "pixel_noise": "0.0",
        "input_noise_scale": "0.0",
        "latent_noise_scale": "0.03",
    }
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    assert main([str(root)]) == 0
    text = (root / "index.html").read_text(encoding="utf-8")
    assert "single-recipe production run" in text
    assert "pre=1 MP" in text
    assert "latent noise=0.03" in text
    assert (root / "report-previews" / "src1.png").is_file()
