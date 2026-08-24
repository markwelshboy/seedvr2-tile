from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

from . import __version__
from . import probe_sweep as _base
from .sweep import SourceImage, _fmt_float, _pre_label


_DETAIL_FRACTION = 0.50
_DEFAULT_CELL_SIZE = 420
_ORIGINAL_FIT_CELL = _base._fit_cell
_ORIGINAL_MAKE_PROBE_SHEET = _base._make_probe_sheet


def _detail_fit_cell(image: Image.Image, size: int) -> Image.Image:
    """Render a matched center crop instead of shrinking the whole probe core.

    Every result core represents the same spatial tile at a different requested
    output scale. Cropping the same normalized central fraction therefore keeps
    the visual comparison spatially aligned while making texture/detail large
    enough to judge in the contact sheet.
    """
    rgb = image.convert("RGB")
    side = max(1, int(round(min(rgb.width, rgb.height) * _DETAIL_FRACTION)))
    left = max(0, (rgb.width - side) // 2)
    top = max(0, (rgb.height - side) // 2)
    crop = rgb.crop((left, top, left + side, top + side))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def _make_probe_sheet_detail_first(*args, target: Path, **kwargs) -> None:
    """Write the zoomed comparison plus a whole-core overview companion."""
    try:
        _base._fit_cell = _detail_fit_cell
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target, **kwargs)

        _base._fit_cell = _ORIGINAL_FIT_CELL
        _ORIGINAL_MAKE_PROBE_SHEET(*args, target=target.with_name("overview.png"), **kwargs)
    finally:
        _base._fit_cell = _detail_fit_cell


def _write_html(
    output_root: Path,
    *,
    sources: Sequence[SourceImage],
    probes_by_source: dict[str, list[_base.Probe]],
    noise_values: Sequence[float],
    plan_only: bool,
) -> None:
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
            "<p><strong>Detail view:</strong> comparison cells show the matched central 50% of each probe core. "
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
                detail_url = html.escape(detail_rel.as_posix())
                overview_url = html.escape(overview_rel.as_posix())
                parts.append(f"<h4>noise={noise:g} · detail crop</h4>")
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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cell-size" not in args:
        args.extend(["--cell-size", str(_DEFAULT_CELL_SIZE)])

    _base._fit_cell = _detail_fit_cell
    _base._make_probe_sheet = _make_probe_sheet_detail_first
    _base._write_html = _write_html
    return _base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
