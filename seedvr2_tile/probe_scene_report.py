from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from . import __version__
from . import probe_sweep as _base
from . import probe_sweep_report as _legacy
from .sweep import SourceImage, _fmt_float, _pre_label


SceneWindow = tuple[float, float, float, float]

_DETAIL_FRACTION = 0.50
_DEFAULT_CELL_SIZE = 420
_REPORT_TILE = 1024
_REPORT_OVERLAP = 64
_REPORT_STRATEGY = "chess"
_SCENE_BY_PATH: dict[str, tuple[_base.PlannedProbeResult, SceneWindow]] = {}

_ORIGINAL_SHEET_CELL = _base._sheet_cell
_ORIGINAL_MAKE_PROBE_SHEET = _base._make_probe_sheet


def _core_box(result: _base.PlannedProbeResult) -> tuple[int, int, int, int]:
    boxes = _legacy._core_boxes(
        result.processed_width,
        result.processed_height,
        _REPORT_TILE,
        _REPORT_OVERLAP,
        _REPORT_STRATEGY,
    )
    box = boxes.get(result.selected_tile_index)
    if box is None:
        raise RuntimeError(
            f"cannot reconstruct tile {result.selected_tile_index} for "
            f"{result.processed_width}x{result.processed_height}"
        )
    return box


def _reference_result(
    source: SourceImage,
    probe: _base.Probe,
    noise: float,
    pre_values: Sequence[float | None],
    scales: Sequence[float],
    result_index: dict[tuple[str, str, float | None, float, float], _base.PlannedProbeResult],
) -> _base.PlannedProbeResult:
    for scale in scales:
        result = result_index.get(
            (source.source_id, probe.probe_id, probe.reference_pre_megapixels, scale, noise)
        )
        if result is not None:
            return result
    for pre_mp in pre_values:
        for scale in scales:
            result = result_index.get((source.source_id, probe.probe_id, pre_mp, scale, noise))
            if result is not None:
                return result
    raise RuntimeError(f"no completed result for {source.source_id}/{probe.probe_id}/noise={noise:g}")


def _scene_window(
    probe: _base.Probe,
    reference: _base.PlannedProbeResult,
    results: Sequence[_base.PlannedProbeResult],
) -> tuple[SceneWindow, float]:
    """Build one full-image normalized rectangle that fits every saved core."""
    rx0, ry0, rx1, ry1 = _core_box(reference)
    side_px = min(rx1 - rx0, ry1 - ry0) * _DETAIL_FRACTION
    half_x = side_px / (2.0 * reference.processed_width)
    half_y = side_px / (2.0 * reference.processed_height)
    if half_x <= 0 or half_y <= 0:
        raise RuntimeError("invalid zero-sized reference crop")

    cx, cy = probe.center_x, probe.center_y
    fit = 1.0
    seen: set[tuple[int, int, int]] = set()
    for result in results:
        geometry = (result.processed_width, result.processed_height, result.selected_tile_index)
        if geometry in seen:
            continue
        seen.add(geometry)
        x0, y0, x1, y1 = _core_box(result)
        nx0 = x0 / result.processed_width
        ny0 = y0 / result.processed_height
        nx1 = x1 / result.processed_width
        ny1 = y1 / result.processed_height
        local_fit = min(
            (cx - nx0) / half_x,
            (nx1 - cx) / half_x,
            (cy - ny0) / half_y,
            (ny1 - cy) / half_y,
        )
        if local_fit <= 0:
            raise RuntimeError(
                "probe center lies on/outside a candidate core; cannot create an exact scene-registered crop"
            )
        fit = min(fit, local_fit)

    fit = max(0.0, min(1.0, fit))
    hx, hy = half_x * fit, half_y * fit
    return (cx - hx, cy - hy, cx + hx, cy + hy), _DETAIL_FRACTION * fit


def _source_box(
    result: _base.PlannedProbeResult,
    saved_size: tuple[int, int],
    window: SceneWindow,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _core_box(result)
    core_w, core_h = max(1, x1 - x0), max(1, y1 - y0)
    wx0, wy0, wx1, wy1 = window
    gx0, gy0 = wx0 * result.processed_width, wy0 * result.processed_height
    gx1, gy1 = wx1 * result.processed_width, wy1 * result.processed_height
    saved_w, saved_h = saved_size
    left = ((gx0 - x0) / core_w) * saved_w
    top = ((gy0 - y0) / core_h) * saved_h
    right = ((gx1 - x0) / core_w) * saved_w
    bottom = ((gy1 - y0) / core_h) * saved_h
    eps = 1e-3
    if left < -eps or top < -eps or right > saved_w + eps or bottom > saved_h + eps:
        raise RuntimeError("shared scene window escaped a saved probe core")
    return (
        min(float(saved_w), max(0.0, left)),
        min(float(saved_h), max(0.0, top)),
        min(float(saved_w), max(0.0, right)),
        min(float(saved_h), max(0.0, bottom)),
    )


def _render_scene_crop(
    image: Image.Image,
    size: int,
    result: _base.PlannedProbeResult,
    window: SceneWindow,
) -> Image.Image:
    rgb = image.convert("RGB")
    box = _source_box(result, rgb.size, window)
    return rgb.resize((size, size), Image.Resampling.LANCZOS, box=box)


def _save_scene_crop(
    path: Path,
    target: Path,
    *,
    size: int,
    result: _base.PlannedProbeResult,
    window: SceneWindow,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as opened:
        rendered = _render_scene_crop(opened, size, result, window)
    rendered.save(target, format="PNG", compress_level=4)


def _scene_sheet_cell(
    path: Path | None,
    *,
    cell_size: int,
    footer: str,
    missing: str = "MISSING",
) -> Image.Image:
    footer_h = 56
    canvas = Image.new("RGB", (cell_size, cell_size + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    if path is not None and path.is_file() and str(path) in _SCENE_BY_PATH:
        result, window = _SCENE_BY_PATH[str(path)]
        with Image.open(path) as opened:
            rendered = _render_scene_crop(opened, cell_size, result, window)
        canvas.paste(rendered, (0, 0))
    elif path is not None and path.is_file():
        with Image.open(path) as opened:
            canvas.paste(_legacy._detail_fit_cell(opened, cell_size), (0, 0))
    else:
        draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline="gray", width=1)
        draw.multiline_text((10, 10), missing, fill="black", font=_base._font(13), spacing=3)
    draw.rectangle((0, 0, cell_size - 1, cell_size + footer_h - 1), outline="gray", width=1)
    draw.multiline_text((7, cell_size + 5), footer, fill="black", font=_base._font(12), spacing=2)
    return canvas


def _make_probe_sheet_detail_first(*args, target: Path, **kwargs) -> None:
    source: SourceImage = kwargs["source"]
    probe: _base.Probe = kwargs["probe"]
    noise: float = kwargs["noise"]
    pre_values = kwargs["pre_values"]
    scales = kwargs["scales"]
    output_root: Path = kwargs["output_root"]
    result_index = kwargs["result_index"]
    cell_size = int(kwargs["cell_size"])

    results = [
        result_index[key]
        for pre_mp in pre_values
        for scale in scales
        if (key := (source.source_id, probe.probe_id, pre_mp, scale, noise)) in result_index
    ]
    reference = _reference_result(source, probe, noise, pre_values, scales, result_index)
    window, effective_fraction = _scene_window(probe, reference, results)

    crop_dir = target.parent / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for stale in crop_dir.glob("*.png"):
        stale.unlink()

    scene_by_path: dict[str, tuple[_base.PlannedProbeResult, SceneWindow]] = {}
    for row_index, pre_mp in enumerate(pre_values, start=1):
        token = _legacy._pre_crop_token(pre_mp)
        any_result = next(
            (
                result_index[(source.source_id, probe.probe_id, pre_mp, scale, noise)]
                for scale in scales
                if (source.source_id, probe.probe_id, pre_mp, scale, noise) in result_index
            ),
            None,
        )
        if any_result is not None:
            input_path = _base._input_core_path(output_root, source, probe, pre_mp, noise)
            scene_by_path[str(input_path)] = (any_result, window)
            _save_scene_crop(
                input_path,
                crop_dir / f"{row_index:02d}_pre-{token}__00-input.png",
                size=cell_size,
                result=any_result,
                window=window,
            )
        for col_index, scale in enumerate(scales, start=1):
            result = result_index.get((source.source_id, probe.probe_id, pre_mp, scale, noise))
            if result is None:
                continue
            scene_by_path[str(result.output_core_path)] = (result, window)
            _save_scene_crop(
                result.output_core_path,
                crop_dir / f"{row_index:02d}_pre-{token}__{col_index:02d}-scale-{_fmt_float(scale)}x.png",
                size=cell_size,
                result=result,
                window=window,
            )

    metadata = {
        "coordinate_space": "normalized-full-preprocessed-image",
        "probe_center": [probe.center_x, probe.center_y],
        "requested_reference_core_fraction": _DETAIL_FRACTION,
        "effective_reference_core_fraction": effective_fraction,
        "window": {"x0": window[0], "y0": window[1], "x1": window[2], "y1": window[3]},
        "note": "Every PNG in crops/ renders this exact same normalized scene rectangle.",
    }
    (target.parent / "scene-window.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    global _SCENE_BY_PATH
    _SCENE_BY_PATH = scene_by_path
    try:
        _base._sheet_cell = _scene_sheet_cell
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target, **kwargs)
        _base._sheet_cell = _ORIGINAL_SHEET_CELL
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target.with_name("overview.png"), **kwargs)
    finally:
        _base._sheet_cell = _ORIGINAL_SHEET_CELL
        _SCENE_BY_PATH = {}


def _write_html(
    output_root: Path,
    *,
    sources: Sequence[SourceImage],
    probes_by_source: dict[str, list[_base.Probe]],
    noise_values: Sequence[float],
    plan_only: bool,
) -> None:
    pct = int(round(_DETAIL_FRACTION * 100))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 probe sweep</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px}"
        ".card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}.meta{color:#555}"
        ".sheet-scroll{overflow-x:auto;max-width:100%;padding-bottom:8px}.sheet{max-width:none;height:auto;border:1px solid #ddd;display:block}"
        "details{margin-top:10px}summary{cursor:pointer;font-weight:600}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>",
        "</head><body><h1>SeedVR2 probe sweep</h1>",
        f"<p class='meta'>seedvr2-tile {html.escape(__version__)} · model 3B FP8 · {'plan only' if plan_only else 'probe inference complete'}</p>",
    ]
    if not plan_only:
        parts.append(
            f"<p><strong>Scene-registered detail view:</strong> the requested crop starts at {pct}% of the reference core and is converted once into a normalized full-image rectangle. "
            "Every preprocessing row and output scale renders that same rectangle. If necessary, it is tightened once globally so it fits every saved core; rows are never independently clamped. "
            "The exact rectangle is in <code>scene-window.json</code> and overlay-ready PNGs are in <code>crops/</code>.</p>"
        )
    for source in sources:
        if not source.selected:
            continue
        parts.append(f"<div class='card'><h2>{html.escape(source.relative.as_posix())}</h2>")
        parts.append(f"<p class='meta'>{source.width}×{source.height} · {source.megapixels:.2f} MP · {source.bucket}</p>")
        for probe in probes_by_source.get(source.source_id, []):
            parts.append(
                f"<h3>{html.escape(probe.probe_id)} — {html.escape(probe.label)}</h3>"
                f"<p class='meta'>normalized center ({probe.center_x:.3f}, {probe.center_y:.3f}), selected from {_pre_label(probe.reference_pre_megapixels)}</p>"
            )
            if plan_only:
                continue
            for noise in noise_values:
                base = Path("reports") / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}"
                detail = html.escape((base / "comparison.png").as_posix())
                overview = html.escape((base / "overview.png").as_posix())
                crops = html.escape((base / "crops").as_posix())
                parts.append(f"<h4>noise={noise:g} · scene-registered detail crop</h4><p class='meta'>Overlay crops: <code>{crops}/</code></p>")
                parts.append(f"<div class='sheet-scroll'><a href='{detail}'><img class='sheet' src='{detail}'></a></div>")
                parts.append(f"<details><summary>Whole probe-core overview</summary><div class='sheet-scroll'><a href='{overview}'><img class='sheet' src='{overview}'></a></div></details>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_root / "index.html").write_text("".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    global _DETAIL_FRACTION, _REPORT_TILE, _REPORT_OVERLAP, _REPORT_STRATEGY
    _DETAIL_FRACTION = _legacy._pop_float_option(args, "--comparison-crop-fraction", _DETAIL_FRACTION)
    if not 0 < _DETAIL_FRACTION <= 1:
        raise SystemExit("--comparison-crop-fraction must be in (0, 1]")
    _REPORT_TILE = int(_legacy._read_option(args, "--tile", "1024"))
    _REPORT_OVERLAP = int(_legacy._read_option(args, "--overlap", "64"))
    _REPORT_STRATEGY = _legacy._read_option(args, "--strategy", "chess")
    if "--cell-size" not in args and not any(arg.startswith("--cell-size=") for arg in args):
        args.extend(["--cell-size", str(_DEFAULT_CELL_SIZE)])
    _base._make_probe_sheet = _make_probe_sheet_detail_first
    _base._write_html = _write_html
    return _base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
