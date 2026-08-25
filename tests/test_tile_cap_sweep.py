import csv
import json
from pathlib import Path

from PIL import Image

from seedvr2_tile.tile_cap_sweep import _prepare_args, _write_report


def _make_leaf(root: Path, cap: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "settings": {
            "latent_noise_scale": 0.0,
            "input_noise_scale": 0.0,
            "tile_upscale_resolution": cap,
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    fields = [
        "source_id", "source", "bucket", "source_mp", "pre_mp", "scale", "noise",
        "predicted_output_mp", "status", "reason", "output", "actual_output_mp",
        "pixel_noise", "input_noise_scale", "latent_noise_scale",
    ]
    rows = []
    for scale in (2.0, 3.0):
        rel = Path("results/small") / f"cap-{cap}-scale-{scale:g}.png"
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 120), (cap // 16 % 255, int(scale * 50), 80)).save(target)
        rows.append(
            {
                "source_id": "src1",
                "source": "photo.jpeg",
                "bucket": "small",
                "source_mp": "1.11",
                "pre_mp": "native",
                "scale": f"{scale:.6f}",
                "noise": "0.0",
                "predicted_output_mp": f"{1.11 * scale * scale:.6f}",
                "status": "complete",
                "reason": "",
                "output": rel.as_posix(),
                "actual_output_mp": "1.0",
                "pixel_noise": "0.0",
                "input_noise_scale": "0.0",
                "latent_noise_scale": "0.0",
            }
        )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_args_extracts_cap_axis():
    argv, caps = _prepare_args([
        "input", "output", "--scales", "2,3", "--tile-upscale-resolution-values", "2048,3072"
    ])
    assert caps == (2048, 3072)
    assert "--tile-upscale-resolution-values" not in argv
    assert argv[:2] == ["input", "output"]


def test_report_groups_caps_as_rows_and_scales_as_columns(tmp_path: Path):
    root = tmp_path / "experiment"
    cap2048 = root / "tile-cap-2048"
    cap3072 = root / "tile-cap-3072"
    _make_leaf(cap2048, 2048)
    _make_leaf(cap3072, 3072)

    groups = _write_report(root, [(2048, cap2048), (3072, cap3072)])
    assert groups == 1

    text = (root / "index.html").read_text(encoding="utf-8")
    assert "photo.jpeg" in text
    assert "2048px" in text
    assert "3072px" in text
    assert "2×" in text
    assert "3×" in text
    assert "cap 2048 · 2×" in text
    assert "cap 3072 · 3×" in text
    assert len(list((root / "tile-cap-thumbs" / "src1").glob("*.jpg"))) == 4
