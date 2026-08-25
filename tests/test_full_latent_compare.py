import csv
from pathlib import Path

from PIL import Image

from seedvr2_tile.full_latent_compare import write_full_comparison_report


def _make_run(root: Path, latent: float, value: int) -> tuple[float, Path]:
    run = root / f"latent-noise-{str(latent).replace('.', 'p')}"
    output_rel = Path("results/small/small__pre-native__scale-2x__noise-0/src1.png")
    output = run / output_rel
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), (value, value, value)).save(output)

    fields = [
        "source_id", "source", "bucket", "source_mp", "pre_mp", "scale", "noise",
        "predicted_output_mp", "status", "reason", "output", "actual_output_mp",
        "pixel_noise", "input_noise_scale", "latent_noise_scale",
    ]
    row = {
        "source_id": "src1",
        "source": "photo.png",
        "bucket": "small",
        "source_mp": "0.75",
        "pre_mp": "native",
        "scale": "2.0",
        "noise": "0.0",
        "predicted_output_mp": "3.0",
        "status": "complete",
        "reason": "",
        "output": output_rel.as_posix(),
        "actual_output_mp": "3.0",
        "pixel_noise": "0.0",
        "input_noise_scale": "0.0",
        "latent_noise_scale": str(latent),
    }
    with (run / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    (run / "index.html").write_text("child", encoding="utf-8")
    return latent, run


def test_full_latent_report_groups_same_source_recipe(tmp_path: Path):
    root = tmp_path / "meta"
    root.mkdir()
    runs = [_make_run(root, 0.0, 40), _make_run(root, 0.03, 180)]

    groups = write_full_comparison_report(root, runs)

    assert len(groups) == 1
    group = groups[0]
    assert group.source_name == "photo.png"
    assert [item.latent for item in group.items] == [0.0, 0.03]

    overlay = root / "overlay-images" / group.group_rel
    assert (overlay / "latent-0.png").is_file()
    assert (overlay / "latent-0p03.png").is_file()
    assert (root / "comparison-thumbs" / group.group_rel / "latent-0.jpg").is_file()

    text = (root / "index.html").read_text(encoding="utf-8")
    assert "SeedVR2 latent-noise comparison" in text
    assert "photo.png" in text
    assert "latent=0" in text
    assert "latent=0.03" in text
    assert "overlay-images/" in text
