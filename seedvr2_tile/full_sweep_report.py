from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from . import noise_sweep as _noise
from .sweep import DEFAULT_CROPS, _fmt_float, _pre_label

_META_MODE = "latent-noise-meta-v1"


def _read_manifest(root: Path) -> dict | None:
    path = root / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_buckets(manifest: dict) -> set[str]:
    return {
        str(item.get("bucket"))
        for item in manifest.get("sources", [])
        if item.get("selected") and item.get("bucket")
    }


def _is_singleton_recipe(manifest: dict) -> bool:
    settings = manifest.get("settings") or {}
    scales = tuple(settings.get("scales") or ())
    pixel_noises = tuple(settings.get("pixel_noise_values") or settings.get("noise_values") or ())
    pre_by_bucket = settings.get("pre_by_bucket") or {}
    buckets = _selected_buckets(manifest)
    if not buckets or len(scales) != 1 or len(pixel_noises) != 1:
        return False
    return all(len(tuple(pre_by_bucket.get(bucket) or ())) == 1 for bucket in buckets)


def _result_rows(root: Path) -> list[dict[str, str]]:
    path = root / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_preview(source: Path, target: Path, max_side: int = 960) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(target, format="PNG", compress_level=4)
    return True


def _recipe_text(row: dict[str, str], manifest: dict) -> str:
    settings = manifest.get("settings") or {}
    raw_pre = row.get("pre_mp", "native")
    pre = "native" if raw_pre.strip().lower() == "native" else _pre_label(float(raw_pre))
    scale = float(row.get("scale", "0") or 0)
    pixel = float(row.get("pixel_noise") or row.get("noise") or 0)
    latent = float(settings.get("latent_noise_scale", 0.0) or 0.0)
    input_noise = float(settings.get("input_noise_scale", 0.0) or 0.0)
    return (
        f"pre={pre} · scale={scale:g}x · pixel noise={pixel:g} · "
        f"latent noise={latent:g} · input noise={input_noise:g}"
    )


def _comparison_links(root: Path, source: dict, row: dict[str, str]) -> list[tuple[str, Path]]:
    pixel = float(row.get("pixel_noise") or row.get("noise") or 0)
    base = Path("reports") / str(source["bucket"]) / str(source["source_id"]) / f"noise-{_fmt_float(pixel)}"
    candidates = [("Full source/result", base / "full.png")]
    candidates.extend((f"{name.title()} crop", base / f"crop-{name}.png") for name in DEFAULT_CROPS)
    return [(label, rel) for label, rel in candidates if (root / rel).is_file()]


def _compact_run(root: Path) -> bool:
    manifest = _read_manifest(root)
    if not manifest or manifest.get("mode") == _META_MODE or not _is_singleton_recipe(manifest):
        return False

    rows = _result_rows(root)
    if not rows:
        return False
    rows_by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_source.setdefault(row.get("source_id", ""), []).append(row)
    if any(len(items) != 1 for items in rows_by_source.values() if items):
        return False

    version = html.escape(str(manifest.get("seedvr2_tile_version", "")))
    failures = manifest.get("failures") or []
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>SeedVR2 full-image report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 18px}",
        "h2{margin-top:38px}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}",
        ".preview{display:block;max-width:min(100%,960px);height:auto;border:1px solid #ddd;border-radius:6px}",
        ".actions{margin:10px 0 16px}details{margin-top:14px;border-top:1px solid #eee;padding-top:10px}",
        "summary{cursor:pointer;font-weight:600}.compare{margin:14px 0}.compare img{max-width:100%;height:auto;border:1px solid #ddd}",
        ".warn{background:#fff4df;padding:10px;border-radius:8px}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}",
        "</style></head><body>",
        "<h1>SeedVR2 full-image report</h1>",
        f"<p class='meta'>seedvr2-tile {version} · single-recipe production run</p>",
        "<p class='meta'>Only one effective parameter combination was requested, so comparison arrays are collapsed by default.</p>",
    ]
    if failures:
        parts.append(f"<p class='warn'>{len(failures)} variant run(s) reported errors. See <code>manifest.json</code>.</p>")

    for bucket in ("small", "medium", "large"):
        bucket_sources = [
            item for item in manifest.get("sources", [])
            if item.get("selected") and item.get("bucket") == bucket
        ]
        if not bucket_sources:
            continue
        parts.append(f"<h2>{bucket.title()} bucket</h2>")
        for source in bucket_sources:
            source_id = str(source["source_id"])
            source_rows = rows_by_source.get(source_id, [])
            row = source_rows[0] if source_rows else None
            parts.append("<div class='card'>")
            parts.append(
                f"<h3>{html.escape(str(source.get('source', source_id)))}</h3>"
                f"<p class='meta'>{int(source.get('width', 0))}×{int(source.get('height', 0))} · "
                f"{float(source.get('megapixels', 0)):.2f} MP</p>"
            )
            if row is None:
                parts.append("<p class='warn'>No result row was recorded for this image.</p></div>")
                continue

            parts.append(f"<p class='meta'>{html.escape(_recipe_text(row, manifest))}</p>")
            status = row.get("status", "unknown")
            output_rel = Path(row["output"]) if row.get("output") else None
            if status == "complete" and output_rel is not None and (root / output_rel).is_file():
                preview_rel = Path("report-previews") / f"{source_id}.png"
                if _make_preview(root / output_rel, root / preview_rel):
                    parts.append(
                        f"<a href='{html.escape(output_rel.as_posix())}'><img class='preview' "
                        f"src='{html.escape(preview_rel.as_posix())}' alt='restored output'></a>"
                    )
                parts.append(
                    f"<p class='actions'><a href='{html.escape(output_rel.as_posix())}'>Open full-resolution output</a></p>"
                )
            else:
                reason = row.get("reason") or status
                parts.append(f"<p class='warn'>Result: {html.escape(reason)}</p>")

            comparisons = _comparison_links(root, source, row)
            if comparisons:
                parts.append("<details><summary>Comparison views</summary>")
                parts.append("<p class='meta'>Legacy source/result contact sheets retained for inspection.</p>")
                for label, rel in comparisons:
                    safe = html.escape(rel.as_posix())
                    parts.append(
                        f"<div class='compare'><h4>{html.escape(label)}</h4>"
                        f"<a href='{safe}'><img src='{safe}' alt='comparison'></a></div>"
                    )
                parts.append("</details>")
            parts.append("</div>")

    parts.append("</body></html>")
    (root / "index.html").write_text("".join(parts), encoding="utf-8")
    print(f"Compacted singleton full-sweep report: {root / 'index.html'}")
    return True


def _compact_tree(root: Path) -> int:
    manifest = _read_manifest(root)
    if not manifest:
        return 0
    if manifest.get("mode") == _META_MODE:
        count = 0
        for item in manifest.get("runs", []):
            run_path = item.get("path")
            if run_path and _compact_run(root / str(run_path)):
                count += 1
        return count
    return 1 if _compact_run(root) else 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_noise.full_main(raw) or 0)
    if rc or not raw or "--help" in raw or "-h" in raw:
        return rc
    if len(raw) >= 2 and not raw[1].startswith("-"):
        _compact_tree(Path(raw[1]).expanduser().resolve())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
