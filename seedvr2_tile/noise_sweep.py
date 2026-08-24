from __future__ import annotations

import csv
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from . import probe_report_dispatch as _report
from . import probe_scene_sweep as _probe
from . import sweep as _full
from .noise_controls import extract_value_option, fmt_value, parse_values, replace_option, upstream_noise

_META_MODE = "latent-noise-meta-v1"


def _has_option(argv: Sequence[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in argv)


def _prepare_args(raw: Sequence[str]) -> tuple[list[str], tuple[float, ...]]:
    argv = list(raw)
    if _has_option(argv, "--pixel-noise-values") and _has_option(argv, "--noise-values"):
        raise SystemExit("use either --pixel-noise-values or legacy --noise-values, not both")
    argv = replace_option(argv, {"--pixel-noise-values"}, "--noise-values")
    argv, latent_raw = extract_value_option(argv, {"--latent-noise-values"}, default="0")
    try:
        latent_values = parse_values(latent_raw or "0")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return argv, latent_values


def _print_help(base_builder: Callable[[], object]) -> int:
    parser = base_builder()
    parser.add_argument(
        "--pixel-noise-values",
        metavar="CSV",
        dest="noise_values",
        help="RGB Gaussian preprocessing noise values; explicit alias for legacy --noise-values",
    )
    parser.add_argument(
        "--latent-noise-values",
        metavar="CSV",
        default="0",
        help="Numz conditioning latent noise scales to sweep, e.g. 0,0.015,0.03,0.05 (default: 0)",
    )
    parser.epilog = (
        "Noise semantics: pixel noise is applied to RGB after pre-resize; upstream "
        "input_noise_scale is held at 0; latent noise uses Numz's diffusion-scheduled conditioning path."
    )
    parser.print_help()
    return 0


def _annotate_run(root: Path, *, latent: float) -> None:
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = manifest.setdefault("settings", {})
        settings["pixel_noise_values"] = list(settings.get("noise_values", [0.0]))
        settings["input_noise_scale"] = 0.0
        settings["latent_noise_scale"] = latent
        manifest["noise_semantics"] = {
            "noise_values": "legacy name for pixel_noise_values",
            "pixel_noise_values": "Gaussian RGB preprocessing noise after optional pre-resize",
            "input_noise_scale": "Numz pre-VAE input noise; intentionally fixed at 0 for these sweeps",
            "latent_noise_scale": "Numz diffusion-scheduled noise applied to the conditioning latent",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    csv_path = root / "results.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0].keys()) if rows else []
        for field in ("pixel_noise", "input_noise_scale", "latent_noise_scale"):
            if field not in fields:
                fields.append(field)
        for row in rows:
            row["pixel_noise"] = row.get("noise", "0")
            row["input_noise_scale"] = "0.000000"
            row["latent_noise_scale"] = f"{latent:.6f}"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    html_path = root / "index.html"
    if html_path.is_file():
        text = html_path.read_text(encoding="utf-8")
        banner = (
            "<p class='meta'><strong>Noise controls:</strong> "
            f"latent noise={latent:g}; Numz input noise=0; "
            "any legacy <code>noise=</code> labels below mean RGB pixel preprocessing noise.</p>"
        )
        marker = "</h1>"
        if banner not in text and marker in text:
            text = text.replace(marker, marker + banner, 1)
            html_path.write_text(text, encoding="utf-8")


def _overlay_groups(root: Path, runs: Sequence[tuple[float, Path]]) -> list[tuple[Path, list[tuple[float, Path]]]]:
    overlay_root = root / "overlay-crops"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    groups: dict[Path, list[tuple[float, Path]]] = {}
    for latent, run_dir in runs:
        reports = run_dir / "reports"
        if not reports.is_dir():
            continue
        for source_crop in reports.glob("**/crops/*.png"):
            rel = source_crop.relative_to(reports)
            # Put each exact pre/scale crop in its own directory so Butterfly sees
            # only the latent-noise variants of the same registered scene.
            group_rel = rel.parent.parent / source_crop.stem
            target_dir = overlay_root / group_rel
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"latent-{fmt_value(latent)}.png"
            shutil.copy2(source_crop, target)
            groups.setdefault(group_rel, []).append((latent, target))
    return sorted(groups.items(), key=lambda item: item[0].as_posix())


def _write_meta(
    root: Path,
    *,
    kind: str,
    runs: Sequence[tuple[float, Path]],
    overlay_groups: Sequence[tuple[Path, list[tuple[float, Path]]]] = (),
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": _META_MODE,
        "kind": kind,
        "input_noise_scale": 0.0,
        "latent_noise_values": [value for value, _ in runs],
        "runs": [
            {"latent_noise_scale": value, "path": path.relative_to(root).as_posix()}
            for value, path in runs
        ],
        "overlay_root": "overlay-crops" if overlay_groups else None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 latent-noise sweep</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:14px;margin:14px 0}.row{display:flex;gap:12px;overflow-x:auto}.row img{width:260px;height:260px;object-fit:contain;border:1px solid #ddd}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        "<h1>SeedVR2 latent-noise sweep</h1>",
        "<p class='meta'>Numz input noise is fixed at 0. RGB preprocessing noise and latent noise are independent controls.</p>",
        "<h2>Per-latent reports</h2><ul>",
    ]
    for latent, path in runs:
        rel = path.relative_to(root) / "index.html"
        parts.append(f"<li>latent={latent:g}: <a href='{html.escape(rel.as_posix())}'>report</a></li>")
    parts.append("</ul>")
    if overlay_groups:
        parts.append("<h2>Butterfly overlay groups</h2><p class='meta'>Each directory contains the same registered crop rendered at different latent noise scales.</p>")
        for group_rel, images in overlay_groups:
            parts.append(f"<div class='card'><h3>{html.escape(group_rel.as_posix())}</h3><p><code>overlay-crops/{html.escape(group_rel.as_posix())}/</code></p><div class='row'>")
            for latent, path in sorted(images):
                rel = path.relative_to(root)
                parts.append(f"<div><div>latent={latent:g}</div><a href='{html.escape(rel.as_posix())}'><img src='{html.escape(rel.as_posix())}'></a></div>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    (root / "index.html").write_text("".join(parts), encoding="utf-8")


def _run(base_main: Callable[[list[str] | None], int], raw: Sequence[str], *, kind: str) -> int:
    argv, latent_values = _prepare_args(raw)
    if len(argv) < 2 or argv[0].startswith("-") or argv[1].startswith("-"):
        raise SystemExit("sweep wrappers expect INPUT OUTPUT before optional flags")

    root = Path(argv[1]).expanduser().resolve()
    if len(latent_values) == 1:
        latent = latent_values[0]
        with upstream_noise(latent=latent, input_noise=0.0):
            rc = int(base_main(argv) or 0)
        if rc == 0:
            _annotate_run(root, latent=latent)
        return rc

    root.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[float, Path]] = []
    for index, latent in enumerate(latent_values, start=1):
        run_dir = root / f"latent-noise-{fmt_value(latent)}"
        run_argv = list(argv)
        run_argv[1] = str(run_dir)
        print(f"Latent sweep {index}/{len(latent_values)}: latent_noise_scale={latent:g}")
        with upstream_noise(latent=latent, input_noise=0.0):
            rc = int(base_main(run_argv) or 0)
        if rc:
            return rc
        _annotate_run(run_dir, latent=latent)
        runs.append((latent, run_dir))

    overlays = _overlay_groups(root, runs) if kind == "probe" else []
    _write_meta(root, kind=kind, runs=runs, overlay_groups=overlays)
    print(f"Latent-noise sweep report: {root / 'index.html'}")
    if overlays:
        print(f"Butterfly overlay crops: {root / 'overlay-crops'}")
    return 0


def probe_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw or "-h" in raw:
        return _print_help(_probe._build_parser)
    return _run(_probe.main, raw, kind="probe")


def full_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw or "-h" in raw:
        return _print_help(_full._build_parser)
    return _run(_full.main, raw, kind="full")


def report_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or "--help" in raw or "-h" in raw:
        return _report.main(raw)
    root = Path(raw[0]).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return _report.main(raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != _META_MODE:
        return _report.main(raw)

    runs: list[tuple[float, Path]] = []
    for item in manifest.get("runs", []):
        latent = float(item["latent_noise_scale"])
        run_dir = root / item["path"]
        rc = _report.main([str(run_dir), *raw[1:]])
        if rc:
            return int(rc)
        _annotate_run(run_dir, latent=latent)
        runs.append((latent, run_dir))
    overlays = _overlay_groups(root, runs) if manifest.get("kind") == "probe" else []
    _write_meta(root, kind=str(manifest.get("kind", "probe")), runs=runs, overlay_groups=overlays)
    print(f"Rebuilt latent-noise meta report: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(probe_main())
