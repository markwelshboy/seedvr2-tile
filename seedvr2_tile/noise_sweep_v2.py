from __future__ import annotations

import json
import sys
from pathlib import Path

from . import noise_sweep as _base
from .full_latent_compare import write_full_comparison_report


def _full_meta_runs(root: Path) -> list[tuple[float, Path]] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != _base._META_MODE or manifest.get("kind") != "full":
        return None
    runs: list[tuple[float, Path]] = []
    for item in manifest.get("runs", []):
        runs.append((float(item["latent_noise_scale"]), root / item["path"]))
    return runs


def full_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_base.full_main(raw) or 0)
    if rc:
        return rc

    # The sweep wrapper contract requires INPUT OUTPUT before optional flags.
    if len(raw) >= 2 and not raw[0].startswith("-") and not raw[1].startswith("-"):
        root = Path(raw[1]).expanduser().resolve()
        runs = _full_meta_runs(root)
        if runs:
            groups = write_full_comparison_report(root, runs)
            print(f"Full-image latent comparison groups: {len(groups)}")
            print(f"Butterfly full-image groups: {root / 'overlay-images'}")
    return 0


def report_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and "--help" not in raw and "-h" not in raw:
        root = Path(raw[0]).expanduser().resolve()
        runs = _full_meta_runs(root)
        if runs is not None:
            groups = write_full_comparison_report(root, runs)
            print(f"Rebuilt {len(groups)} full-image latent comparison group(s); no SeedVR2 inference performed.")
            print(f"Report: {root / 'index.html'}")
            print(f"Butterfly full-image groups: {root / 'overlay-images'}")
            return 0
    return int(_base.report_main(raw) or 0)
