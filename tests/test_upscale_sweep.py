import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import seedvr2_tile.upscale_sweep as upscale_sweep


class FakeUpscaler:
    def __init__(self, model_name, **kwargs):
        self.info = SimpleNamespace(
            name=model_name,
            path=Path("fake-x2.pth"),
            architecture="FAKE",
            scale=2,
            device="cpu",
            dtype="float32",
        )

    def upscale(self, image: Image.Image) -> Image.Image:
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)


def test_generic_upscale_reuses_bucket_preprocess_and_report(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    Image.new("RGB", (160, 120), "gray").save(source_dir / "photo.jpg")
    output = tmp_path / "output"

    monkeypatch.setattr(upscale_sweep, "SpandrelUpscaler", FakeUpscaler)
    rc = upscale_sweep.main(
        [
            str(source_dir),
            str(output),
            "--model",
            "fake-x2",
            "--all-images",
            "--only-bucket",
            "small",
            "--pre-small",
            "native",
            "--scales",
            "2",
            "--pixel-noise-values",
            "0",
        ]
    )
    assert rc == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "generic-upscale-v1"
    assert manifest["backend"] == "spandrel"
    assert manifest["model"] == "fake-x2"
    assert manifest["model_info"]["native_scale"] == 2

    with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    result = output / rows[0]["output"]
    assert result.is_file()
    with Image.open(result) as image:
        assert image.size == (320, 240)

    html = (output / "index.html").read_text(encoding="utf-8")
    assert "fake-x2 upscale report" in html
    assert "architecture=FAKE" in html
