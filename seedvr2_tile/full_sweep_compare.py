from __future__ import annotations

import json
import sys
from pathlib import Path

from . import full_report_rebuild as _rebuild
from . import full_sweep_report as _full
from .full_latent_compare import write_full_comparison_report

_META_MODE = "latent-noise-meta-v1"


def _latent_runs(root: Path) -> list[tuple[float, Path]] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != _META_MODE or manifest.get("kind") != "full":
        return None
    return [
        (float(item.get("latent_noise_scale", 0.0)), root / str(item["path"]))
        for item in manifest.get("runs", [])
        if item.get("path")
    ]


def _refresh_comparison(root: Path) -> None:
    runs = _latent_runs(root)
    if not runs:
        return
    groups = write_full_comparison_report(root, runs)
    print(f"Full-image latent comparison groups: {len(groups)}")
    print(f"Butterfly full-image groups: {root / 'overlay-images'}")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_full.main(raw) or 0)
    if rc or not raw or "--help" in raw or "-h" in raw:
        return rc
    if len(raw) >= 2 and not raw[1].startswith("-"):
        _refresh_comparison(Path(raw[1]).expanduser().resolve())
    return 0


def report_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_rebuild.main(raw) or 0)
    if rc or not raw or raw[0] in {"-h", "--help"}:
        return rc
    if len(raw) == 1:
        _refresh_comparison(Path(raw[0]).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
