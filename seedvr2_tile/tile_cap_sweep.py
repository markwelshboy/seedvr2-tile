from __future__ import annotations

import csv
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from . import full_sweep_compare as _base
from .noise_controls import extract_value_option

_META_MODE = "tile-upscale-resolution-meta-v1"
_LATENT_META_MODE = "latent-noise-meta-v1"


def _has_option(argv: Sequence[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in argv)


def _parse_caps(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        cap = int(token)
        if cap < 16:
            raise ValueError(f"tile upscale resolution must be >= 16, got {cap}")
        if cap not in result:
            result.append(cap)
    if not result:
        raise ValueError("at least one tile upscale resolution is required")
    return tuple(result)


def _prepare_args(raw: Sequence[str]) -> tuple[list[str], tuple[int, ...] | None]:
    argv = list(raw)
    if _has_option(argv, "--tile-upscale-resolution-values") and _has_option(argv, "--tile-upscale-resolution"):
        raise SystemExit(
            "use either --tile-upscale-resolution-values or --tile-upscale-resolution, not both"
        )
    argv, raw_values = extract_value_option(
        argv,
        {"--tile-upscale-resolution-values"},
        default=None,
    )
    if raw_values is None:
        return argv, None
    try:
        return argv, _parse_caps(raw_values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(root: Path) -> list[dict[str, str]]:
    path = root / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _leaf_runs(root: Path) -> list[tuple[float, Path]]:
    """Return (latent_noise_scale, leaf full-sweep root) pairs for one cap run."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = _read_json(manifest_path)
    if manifest.get("mode") == _LATENT_META_MODE and manifest.get("kind") == "full":
        return [
            (float(item.get("latent_noise_scale", 0.0)), root / str(item["path"]))
            for item in manifest.get("runs", [])
            if item.get("path")
        ]
    settings = manifest.get("settings") or {}
    return [(float(settings.get("latent_noise_scale", 0.0) or 0.0), root)]


def _preview(source: Path, target: Path, max_side: int = 560) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=88, optimize=True)
    return True


def _fmt_token(value: object) -> str:
    text = str(value).strip().lower().replace(".", "p")
    return "native" if text in {"native", "none", "off", ""} else text


def _collect(
    root: Path,
    runs: Sequence[tuple[int, Path]],
) -> dict[tuple[str, str, str, str, str, str], dict[int, dict[float, tuple[Path, dict[str, str]]]]]:
    """Group by source+recipe excluding scale/cap; cells are cap -> scale -> output."""
    groups: dict[
        tuple[str, str, str, str, str, str],
        dict[int, dict[float, tuple[Path, dict[str, str]]]],
    ] = {}
    for cap, cap_root in runs:
        for latent, leaf in _leaf_runs(cap_root):
            for row in _read_rows(leaf):
                if row.get("status") != "complete" or not row.get("output"):
                    continue
                output = leaf / row["output"]
                if not output.is_file():
                    continue
                pixel = row.get("pixel_noise") or row.get("noise") or "0"
                pre = row.get("pre_mp") or "native"
                key = (
                    row.get("source_id", ""),
                    row.get("source", row.get("source_id", "image")),
                    row.get("bucket", ""),
                    str(pre),
                    str(pixel),
                    f"{latent:.9g}",
                )
                scale = float(row.get("scale", "0") or 0)
                groups.setdefault(key, {}).setdefault(cap, {})[scale] = (output, row)
    return groups


def _write_report(root: Path, runs: Sequence[tuple[int, Path]]) -> int:
    groups = _collect(root, runs)
    caps = sorted({cap for cap, _ in runs})
    scales = sorted({scale for cells in groups.values() for by_scale in cells.values() for scale in by_scale})
    preview_root = root / "tile-cap-thumbs"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 tile-cap comparison</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:14px;margin:18px 0;overflow-x:auto}",
        "table{border-collapse:collapse;min-width:900px}th,td{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}th{background:#f7f7f7;position:sticky;top:0}",
        ".cell img{display:block;width:360px;max-height:360px;object-fit:contain;background:#fafafa}.label{font-weight:650;margin-bottom:6px}.sub{font-size:13px;color:#555;margin-top:6px}",
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        "<h1>SeedVR2 tile-upscale-resolution comparison</h1>",
        "<p class='meta'>Rows vary the SeedVR2 backend tile resolution cap; columns vary requested output scale. Pixel/input/latent noise and preprocessing are held within each card.</p>",
        "<p class='meta'>This is the useful test for whether 3× was being limited by the 2048 backend cap rather than by the model itself.</p>",
    ]

    for key in sorted(groups, key=lambda item: (item[2], item[1], item[3], item[5])):
        source_id, source_name, bucket, pre, pixel, latent = key
        cells = groups[key]
        parts.append("<div class='card'>")
        parts.append(f"<h2>{html.escape(source_name)}</h2>")
        parts.append(
            f"<p class='meta'>{html.escape(bucket)} · pre={html.escape(pre)} · pixel noise={html.escape(pixel)} · latent noise={html.escape(latent)} · input noise=0</p>"
        )
        parts.append("<table><thead><tr><th>backend cap</th>")
        for scale in scales:
            parts.append(f"<th>{scale:g}×</th>")
        parts.append("</tr></thead><tbody>")
        for cap in caps:
            parts.append(f"<tr><th>{cap}px</th>")
            for scale in scales:
                item = cells.get(cap, {}).get(scale)
                if item is None:
                    parts.append("<td>—</td>")
                    continue
                output, row = item
                rel_output = output.relative_to(root)
                thumb_rel = Path("tile-cap-thumbs") / source_id / f"pre-{_fmt_token(pre)}__scale-{scale:g}x__cap-{cap}__latent-{_fmt_token(latent)}.jpg"
                _preview(output, root / thumb_rel)
                pred = row.get("predicted_output_mp") or row.get("predicted_full_output_mp") or ""
                parts.append("<td class='cell'>")
                parts.append(f"<div class='label'>cap {cap} · {scale:g}×</div>")
                parts.append(
                    f"<a href='{html.escape(rel_output.as_posix())}'><img src='{html.escape(thumb_rel.as_posix())}' alt='comparison'></a>"
                )
                if pred:
                    parts.append(f"<div class='sub'>predicted full output ≈ {html.escape(str(pred))} MP</div>")
                parts.append("</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    if not groups:
        parts.append("<p>No completed full-image results were found in the cap runs.</p>")
    parts.append("</body></html>")
    (root / "index.html").write_text("".join(parts), encoding="utf-8")
    return len(groups)


def _write_manifest(root: Path, runs: Sequence[tuple[int, Path]]) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": _META_MODE,
        "kind": "full",
        "tile_upscale_resolution_values": [cap for cap, _ in runs],
        "runs": [
            {"tile_upscale_resolution": cap, "path": path.relative_to(root).as_posix()}
            for cap, path in runs
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _meta_runs(root: Path) -> list[tuple[int, Path]] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("mode") != _META_MODE:
        return None
    return [
        (int(item["tile_upscale_resolution"]), root / str(item["path"]))
        for item in manifest.get("runs", [])
        if item.get("path")
    ]


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw or "-h" in raw:
        # Preserve the established help and advertise the wrapper-only axis.
        try:
            return int(_base.main(raw) or 0)
        finally:
            print("\nAdditional full-sweep comparison option:")
            print("  --tile-upscale-resolution-values CSV   run one full sweep per backend cap, e.g. 2048,3072")

    base_argv, caps = _prepare_args(raw)
    if caps is None:
        return int(_base.main(base_argv) or 0)
    if len(base_argv) < 2 or base_argv[0].startswith("-") or base_argv[1].startswith("-"):
        raise SystemExit("seedvr2-sweep-full expects INPUT OUTPUT before optional flags")

    root = Path(base_argv[1]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[int, Path]] = []
    for index, cap in enumerate(caps, start=1):
        run_dir = root / f"tile-cap-{cap}"
        run_argv = list(base_argv)
        run_argv[1] = str(run_dir)
        run_argv.extend(["--tile-upscale-resolution", str(cap)])
        print(f"Tile-cap sweep {index}/{len(caps)}: tile_upscale_resolution={cap}")
        rc = int(_base.main(run_argv) or 0)
        if rc:
            return rc
        runs.append((cap, run_dir))

    _write_manifest(root, runs)
    groups = _write_report(root, runs)
    print(f"Tile-cap comparison groups: {groups}")
    print(f"Tile-cap comparison report: {root / 'index.html'}")
    return 0


def report_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print("usage: seedvr2-sweep-full-report SWEEP_DIR")
        print("Rebuild full-sweep reports, including tile-cap comparison sweeps; no inference is performed.")
        return 0
    if len(raw) != 1:
        return int(_base.report_main(raw) or 0)

    root = Path(raw[0]).expanduser().resolve()
    runs = _meta_runs(root)
    if runs is None:
        return int(_base.report_main(raw) or 0)

    for _, child in runs:
        rc = int(_base.report_main([str(child)]) or 0)
        if rc:
            return rc
    groups = _write_report(root, runs)
    print(f"Rebuilt {groups} tile-cap comparison group(s); no SeedVR2 inference performed.")
    print(f"Report: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
