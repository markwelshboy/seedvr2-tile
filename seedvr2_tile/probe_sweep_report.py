from __future__ import annotations

import html
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from . import __version__
from . import probe_sweep as _base
from .sweep import SourceImage, _fmt_float, _pre_label
from .tiling import make_tiles


_DETAIL_FRACTION = 0.50
_DEFAULT_CELL_SIZE = 420
_REPORT_TILE = 1024
_REPORT_OVERLAP = 64
_REPORT_STRATEGY = "chess"
_DETAIL_FOCUS_BY_PATH: dict[str, tuple[float, float]] = {}

_ORIGINAL_SHEET_CELL = _base._sheet_cell
_ORIGINAL_MAKE_PROBE_SHEET = _base._make_probe_sheet


def _detail_fit_cell(
    image: Image.Image,
    size: int,
    focus: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """Render a square detail crop centered on a normalized focus point."""
    rgb = image.convert("RGB")
    side = max(1, int(round(min(rgb.width, rgb.height) * _DETAIL_FRACTION)))
    fx = min(1.0, max(0.0, focus[0]))
    fy = min(1.0, max(0.0, focus[1]))
    center_x = fx * rgb.width
    center_y = fy * rgb.height
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(0, left), max(0, rgb.width - side))
    top = min(max(0, top), max(0, rgb.height - side))
    crop = rgb.crop((left, top, left + side, top + side))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def _save_detail_crop(
    source_path: Path,
    target: Path,
    *,
    size: int,
    focus: tuple[float, float],
) -> None:
    """Save the exact normalized detail crop used by the comparison sheet."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        rendered = _detail_fit_cell(opened, size, focus)
    rendered.save(target, format="PNG", compress_level=4)


def _pre_crop_token(pre_mp: float | None) -> str:
    return "native" if pre_mp is None else f"{_fmt_float(pre_mp)}mp"


@lru_cache(maxsize=64)
def _core_boxes(
    width: int,
    height: int,
    tile: int,
    overlap: int,
    strategy: str,
) -> dict[int, tuple[int, int, int, int]]:
    blank = Image.new("RGB", (width, height))
    pairs = make_tiles(
        blank,
        image_index=0,
        tile_width=tile,
        tile_height=tile,
        padding=overlap,
        strategy=strategy,
    )
    return {spec.tile_index: spec.core_box for spec, _ in pairs}


def _focus_for_result(probe: _base.Probe, result: _base.PlannedProbeResult) -> tuple[float, float]:
    boxes = _core_boxes(
        result.processed_width,
        result.processed_height,
        _REPORT_TILE,
        _REPORT_OVERLAP,
        _REPORT_STRATEGY,
    )
    box = boxes.get(result.selected_tile_index)
    if box is None:
        return (0.5, 0.5)
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    global_x = probe.center_x * result.processed_width
    global_y = probe.center_y * result.processed_height
    local_x = (global_x - x0) / width
    local_y = (global_y - y0) / height
    return (
        min(1.0, max(0.0, local_x)),
        min(1.0, max(0.0, local_y)),
    )


def _detail_sheet_cell(
    path: Path | None,
    *,
    cell_size: int,
    footer: str,
    missing: str = "MISSING",
) -> Image.Image:
    footer_h = 56
    canvas = Image.new("RGB", (cell_size, cell_size + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    if path is not None and path.is_file():
        with Image.open(path) as opened:
            focus = _DETAIL_FOCUS_BY_PATH.get(str(path), (0.5, 0.5))
            rendered = _detail_fit_cell(opened, cell_size, focus)
        canvas.paste(rendered, (0, 0))
    else:
        draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline="gray", width=1)
        draw.multiline_text((10, 10), missing, fill="black", font=_base._font(13), spacing=3)
    draw.rectangle((0, 0, cell_size - 1, cell_size + footer_h - 1), outline="gray", width=1)
    draw.multiline_text((7, cell_size + 5), footer, fill="black", font=_base._font(12), spacing=2)
    return canvas


def _make_probe_sheet_detail_first(*args, target: Path, **kwargs) -> None:
    """Write aligned detail crops, a comparison sheet and a whole-core overview."""
    source: SourceImage = kwargs["source"]
    probe: _base.Probe = kwargs["probe"]
    noise: float = kwargs["noise"]
    pre_values = kwargs["pre_values"]
    scales = kwargs["scales"]
    output_root: Path = kwargs["output_root"]
    result_index = kwargs["result_index"]
    cell_size = int(kwargs["cell_size"])

    crop_dir = target.parent / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for stale in crop_dir.glob("*.png"):
        stale.unlink()

    focus_by_path: dict[str, tuple[float, float]] = {}
    for row_index, pre_mp in enumerate(pre_values, start=1):
        pre_token = _pre_crop_token(pre_mp)
        any_result = next(
            (
                result_index[(source.source_id, probe.probe_id, pre_mp, scale, noise)]
                for scale in scales
                if (source.source_id, probe.probe_id, pre_mp, scale, noise) in result_index
            ),
            None,
        )
        if any_result is not None:
            focus = _focus_for_result(probe, any_result)
            input_path = _base._input_core_path(output_root, source, probe, pre_mp, noise)
            focus_by_path[str(input_path)] = focus
            _save_detail_crop(
                input_path,
                crop_dir / f"{row_index:02d}_pre-{pre_token}__00-input.png",
                size=cell_size,
                focus=focus,
            )

        for col_index, scale in enumerate(scales, start=1):
            result = result_index.get((source.source_id, probe.probe_id, pre_mp, scale, noise))
            if result is not None:
                focus = _focus_for_result(probe, result)
                focus_by_path[str(result.output_core_path)] = focus
                _save_detail_crop(
                    result.output_core_path,
                    crop_dir / (
                        f"{row_index:02d}_pre-{pre_token}__{col_index:02d}-"
                        f"scale-{_fmt_float(scale)}x.png"
                    ),
                    size=cell_size,
                    focus=focus,
                )

    global _DETAIL_FOCUS_BY_PATH
    _DETAIL_FOCUS_BY_PATH = focus_by_path
    try:
        _base._sheet_cell = _detail_sheet_cell
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target, **kwargs)

        _base._sheet_cell = _ORIGINAL_SHEET_CELL
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target.with_name("overview.png"), **kwargs)
    finally:
        _base._sheet_cell = _detail_sheet_cell
        _DETAIL_FOCUS_BY_PATH = {}


def _write_html(
    output_root: Path,
    *,
    sources: Sequence[SourceImage],
    probes_by_source: dict[str, list[_base.Probe]],
    noise_values: Sequence[float],
    plan_only: bool,
) -> None:
    crop_percent = int(round(_DETAIL_FRACTION * 100))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 probe sweep</title>",
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px}"
        ".card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}.meta{color:#555}"
        ".sheet-scroll{overflow-x:auto;max-width:100%;padding-bottom:8px}"
        ".sheet{max-width:none;height:auto;border:1px solid #ddd;display:block}"
        "details{margin-top:10px}summary{cursor:pointer;font-weight:600}"
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}"
        "</style></head><body><h1>SeedVR2 probe sweep</h1>",
        f"<p class='meta'>seedvr2-tile {html.escape(__version__)} · model 3B FP8 · "
        f"{'plan only' if plan_only else 'probe inference complete'}</p>",
        "<p>This report preprocesses the full source first, then runs SeedVR2 only on representative tiles from the actual post-preprocess tile grid.</p>",
    ]
    if not plan_only:
        parts.append(
            f"<p><strong>Detail view:</strong> comparison cells show a {crop_percent}% square crop centered on the same normalized probe location in every preprocessing/scale result. "
            "The same normalized crops are also saved as individual PNGs under each case's <code>crops/</code> directory for flip/overlay comparison. "
            "The whole-core overview is retained below each comparison. Sheets are shown at native size and scroll horizontally rather than being shrunk to the browser width.</p>"
        )

    for source in sources:
        if not source.selected:
            continue
        probes = probes_by_source.get(source.source_id, [])
        parts.append(f"<div class='card'><h2>{html.escape(source.relative.as_posix())}</h2>")
        parts.append(f"<p class='meta'>{source.width}×{source.height} · {source.megapixels:.2f} MP · {source.bucket}</p>")
        for probe in probes:
            parts.append(
                f"<h3>{html.escape(probe.probe_id)} — {html.escape(probe.label)}</h3>"
                f"<p class='meta'>normalized center ({probe.center_x:.3f}, {probe.center_y:.3f}), "
                f"selected from {_pre_label(probe.reference_pre_megapixels)}</p>"
            )
            if plan_only:
                continue
            for noise in noise_values:
                base = Path("reports") / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}"
                detail_rel = base / "comparison.png"
                overview_rel = base / "overview.png"
                crops_rel = base / "crops"
                detail_url = html.escape(detail_rel.as_posix())
                overview_url = html.escape(overview_rel.as_posix())
                crops_text = html.escape(crops_rel.as_posix())
                parts.append(f"<h4>noise={noise:g} · detail crop</h4>")
                parts.append(
                    f"<p class='meta'>Individual overlay-ready crops: <code>{crops_text}/</code></p>"
                )
                parts.append(
                    f"<div class='sheet-scroll'><a href='{detail_url}'><img class='sheet' src='{detail_url}' alt='detail comparison'></a></div>"
                )
                parts.append(
                    f"<details><summary>Whole probe-core overview</summary>"
                    f"<div class='sheet-scroll'><a href='{overview_url}'><img class='sheet' src='{overview_url}' alt='probe overview'></a></div>"
                    f"</details>"
                )
        parts.append("</div>")
    parts.append("</body></html>")
    (output_root / "index.html").write_text("".join(parts), encoding="utf-8")


def _pop_float_option(args: list[str], name: str, default: float) -> float:
    for index, arg in enumerate(list(args)):
        if arg == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            raw = args[index + 1]
            del args[index:index + 2]
            return float(raw)
        prefix = name + "="
        if arg.startswith(prefix):
            raw = arg[len(prefix):]
            del args[index]
            return float(raw)
    return default


def _read_option(args: Sequence[str], name: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        prefix = name + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    global _DETAIL_FRACTION, _REPORT_TILE, _REPORT_OVERLAP, _REPORT_STRATEGY
    _DETAIL_FRACTION = _pop_float_option(args, "--comparison-crop-fraction", _DETAIL_FRACTION)
    if not 0 < _DETAIL_FRACTION <= 1:
        raise SystemExit("--comparison-crop-fraction must be in (0, 1]")
    _REPORT_TILE = int(_read_option(args, "--tile", "1024"))
    _REPORT_OVERLAP = int(_read_option(args, "--overlap", "64"))
    _REPORT_STRATEGY = _read_option(args, "--strategy", "chess")

    if "--cell-size" not in args and not any(arg.startswith("--cell-size=") for arg in args):
        args.extend(["--cell-size", str(_DEFAULT_CELL_SIZE)])

    _base._make_probe_sheet = _make_probe_sheet_detail_first
    _base._write_html = _write_html
    return _base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
