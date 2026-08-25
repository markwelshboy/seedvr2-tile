from __future__ import annotations

import json
import sys
from pathlib import Path

from . import full_sweep_report as _compact
from . import noise_sweep as _noise
from . import sweep as _full

_META_MODE = "latent-noise-meta-v1"


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pre_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"native", "none", "off"}:
        return None
    return float(value)


def _source(item: dict) -> _full.SourceImage:
    relative = Path(str(item.get("source", item.get("source_id", "image"))))
    return _full.SourceImage(
        source_id=str(item["source_id"]),
        source_path=relative,
        relative=relative,
        width=int(item.get("width", 0) or 0),
        height=int(item.get("height", 0) or 0),
        megapixels=float(item.get("megapixels", 0.0) or 0.0),
        bucket=str(item.get("bucket", "")),
        selected=bool(item.get("selected", False)),
    )


def _rebuild_one(root: Path) -> bool:
    manifest = _read_manifest(root)
    if manifest.get("mode") == _META_MODE:
        return False
    results = root / "results.csv"
    if not results.is_file():
        raise SystemExit(f"results not found: {results}")

    settings = manifest.get("settings") or {}
    pre_raw = settings.get("pre_by_bucket") or {}
    pre_by_bucket = {
        bucket: tuple(_pre_value(value) for value in pre_raw.get(bucket, []))
        for bucket in ("small", "medium", "large")
    }
    noise_values = tuple(
        float(value)
        for value in (settings.get("pixel_noise_values") or settings.get("noise_values") or [0.0])
    )
    sources = [_source(item) for item in manifest.get("sources", [])]
    if not sources:
        raise SystemExit(f"manifest has no sources: {root / 'manifest.json'}")

    _full._write_html(
        root,
        sources=sources,
        pre_by_bucket=pre_by_bucket,
        noise_values=noise_values,
        crop_names=list(_full.DEFAULT_CROPS),
        failures=manifest.get("failures") or [],
        plan_only=False,
    )
    compacted = _compact._compact_run(root)
    print(f"Rebuilt full-sweep report from existing artifacts: {root / 'index.html'}")
    if compacted:
        print("  single effective recipe: compact production layout enabled")
    return True


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print("usage: seedvr2-sweep-full-report SWEEP_DIR")
        print("Rebuild full-sweep HTML/previews from existing results; no SeedVR2 inference is performed.")
        return 0
    if len(raw) != 1:
        raise SystemExit("usage: seedvr2-sweep-full-report SWEEP_DIR")

    root = Path(raw[0]).expanduser().resolve()
    manifest = _read_manifest(root)
    if manifest.get("mode") == _META_MODE:
        if manifest.get("kind") != "full":
            raise SystemExit("this is a probe latent-noise report; use seedvr2-sweep-report instead")
        runs: list[tuple[float, Path]] = []
        count = 0
        for item in manifest.get("runs", []):
            run_path = item.get("path")
            if not run_path:
                continue
            child = root / str(run_path)
            if _rebuild_one(child):
                count += 1
            runs.append((float(item.get("latent_noise_scale", 0.0)), child))
        _noise._write_meta(root, kind="full", runs=runs, overlay_groups=[])
        print(f"Rebuilt {count} latent child report(s); no SeedVR2 inference performed.")
        print(f"Report: {root / 'index.html'}")
        return 0

    _rebuild_one(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
