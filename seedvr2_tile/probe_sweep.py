from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import __version__
from .backend import BackendOptions, ensure_models, resolve_model_name, resolve_seedvr2_root, run_group
from .fbcnn import FBCNNOptions, release_fbcnn
from .preprocess import PreprocessOptions, preprocess_image
from .sweep import (
    BUCKETS,
    DEFAULT_NOISE,
    DEFAULT_PRESETS,
    DEFAULT_SCALES,
    SourceImage,
    Variant,
    _discover_sources,
    _fmt_float,
    _mark_selected,
    _parse_csv_floats,
    _parse_pre_values,
    _pre_label,
    _predicted_output_mp,
    variant_applies,
    variant_id,
)
from .tiling import TileSpec, make_tiles


@dataclass(frozen=True)
class Probe:
    probe_id: str
    label: str
    center_x: float
    center_y: float
    reference_pre_megapixels: float | None
    reference_tile_index: int
    brightness: float
    detail: float


@dataclass(frozen=True)
class ProbeCandidate:
    spec: TileSpec
    center_x: float
    center_y: float
    brightness: float
    detail: float


@dataclass(frozen=True)
class PlannedProbeResult:
    source_id: str
    source_name: str
    bucket: str
    probe_id: str
    probe_label: str
    pre_megapixels: float | None
    scale: float
    noise: float
    processed_width: int
    processed_height: int
    total_tiles: int
    selected_tile_index: int
    selected_filename: str
    group_resolution: int
    input_core_path: Path
    output_core_path: Path


def _round_even(value: float) -> int:
    return max(16, int(round(value / 2.0) * 2))


def extract_core(spec: TileSpec, processed: Image.Image, *, scale: float) -> Image.Image:
    """Extract exactly the scaled core region that this processed tile contributes.

    This mirrors Stitcher.add's processing-tile resize and coordinate mapping but
    returns only the non-overlap core, which is the cleanest unit for probe
    comparisons because it does not depend on neighboring probe tiles being run.
    """
    process_w, process_h = spec.process_size
    out_w = max(1, round(process_w * scale))
    out_h = max(1, round(process_h * scale))
    arr = np.asarray(processed.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    pad_l, pad_t, _, _ = spec.synthetic_padding
    sx0, sy0, _, _ = spec.source_box
    desired_x0 = sx0 - pad_l
    desired_y0 = sy0 - pad_t
    cx0, cy0, cx1, cy1 = spec.core_box
    x0 = round((cx0 - desired_x0) * scale)
    y0 = round((cy0 - desired_y0) * scale)
    x1 = round((cx1 - desired_x0) * scale)
    y1 = round((cy1 - desired_y0) * scale)
    core = resized[y0:y1, x0:x1]
    if core.size == 0:
        raise RuntimeError(f"empty probe core extracted from {spec.filename}")
    return Image.fromarray(core, mode="RGB")


def _processed_mp(source: SourceImage, pre_mp: float | None) -> float:
    return source.megapixels if pre_mp is None else pre_mp


def _valid_pre_values(source: SourceImage, values: Sequence[float | None]) -> list[float | None]:
    result: list[float | None] = []
    for value in values:
        if value is None or value < source.megapixels * 0.98:
            result.append(value)
    return result


def _reference_pre(source: SourceImage, values: Sequence[float | None]) -> float | None:
    valid = _valid_pre_values(source, values)
    if not valid:
        return None
    return max(valid, key=lambda value: _processed_mp(source, value))


def _preprocess_full(
    source: SourceImage,
    *,
    pre_mp: float | None,
    noise: float,
    source_index: int,
    args: argparse.Namespace,
) -> Image.Image:
    with Image.open(source.source_path) as opened:
        opened = ImageOps.exif_transpose(opened)
        rgb = opened.convert("RGB")
    options = PreprocessOptions(
        fbcnn=FBCNNOptions(
            enabled=args.fbcnn,
            quality=args.jpeg_quality,
            root=None,
            device=args.fbcnn_device,
        ),
        megapixels=pre_mp,
        resample=args.pre_resample,
        noise=noise,
        noise_seed=args.seed,
    )
    processed, _, _ = preprocess_image(
        rgb,
        None,
        options,
        image_index=source_index,
        base_seed=args.seed,
    )
    return processed


def _core_metrics(image: Image.Image, spec: TileSpec) -> tuple[float, float]:
    core = image.crop(spec.core_box).convert("L")
    arr = np.asarray(core, dtype=np.float32) / 255.0
    brightness = float(arr.mean()) if arr.size else 0.0
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        detail = 0.0
    else:
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)
        detail = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    return brightness, detail


def _probe_candidates(image: Image.Image, tile_pairs: Sequence[tuple[TileSpec, Image.Image]]) -> list[ProbeCandidate]:
    result: list[ProbeCandidate] = []
    for spec, _ in tile_pairs:
        cx0, cy0, cx1, cy1 = spec.core_box
        center_x = ((cx0 + cx1) / 2.0) / image.width
        center_y = ((cy0 + cy1) / 2.0) / image.height
        brightness, detail = _core_metrics(image, spec)
        result.append(
            ProbeCandidate(
                spec=spec,
                center_x=center_x,
                center_y=center_y,
                brightness=brightness,
                detail=detail,
            )
        )
    return result


def _choose_distinct_probe_candidates(candidates: Sequence[ProbeCandidate], count: int) -> list[tuple[str, ProbeCandidate]]:
    if count <= 0 or not candidates:
        return []
    count = min(count, len(candidates))
    chosen: list[tuple[str, ProbeCandidate]] = []
    used: set[int] = set()

    selectors = [
        ("detail", sorted(candidates, key=lambda item: (-item.detail, item.spec.tile_index))),
        ("dark", sorted(candidates, key=lambda item: (item.brightness, item.spec.tile_index))),
        (
            "center",
            sorted(
                candidates,
                key=lambda item: ((item.center_x - 0.5) ** 2 + (item.center_y - 0.5) ** 2, item.spec.tile_index),
            ),
        ),
    ]
    for label, ordered in selectors:
        if len(chosen) >= count:
            break
        candidate = next((item for item in ordered if item.spec.tile_index not in used), None)
        if candidate is None:
            continue
        used.add(candidate.spec.tile_index)
        chosen.append((label, candidate))

    if len(chosen) < count:
        remaining = sorted(candidates, key=lambda item: (-item.detail, item.brightness, item.spec.tile_index))
        for candidate in remaining:
            if len(chosen) >= count:
                break
            if candidate.spec.tile_index in used:
                continue
            used.add(candidate.spec.tile_index)
            chosen.append(("extra", candidate))
    return chosen


def select_probes(
    source: SourceImage,
    *,
    pre_values: Sequence[float | None],
    source_index: int,
    args: argparse.Namespace,
) -> list[Probe]:
    reference_pre = _reference_pre(source, pre_values)
    reference = _preprocess_full(
        source,
        pre_mp=reference_pre,
        noise=0.0,
        source_index=source_index,
        args=args,
    )
    tile_pairs = make_tiles(
        reference,
        image_index=source_index,
        tile_width=args.tile,
        tile_height=args.tile,
        padding=args.overlap,
        strategy=args.strategy,
    )
    candidates = _probe_candidates(reference, tile_pairs)
    selected = _choose_distinct_probe_candidates(candidates, args.probe_tiles)
    probes: list[Probe] = []
    for index, (label, candidate) in enumerate(selected, start=1):
        probes.append(
            Probe(
                probe_id=f"p{index}-{label}",
                label=label,
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                reference_pre_megapixels=reference_pre,
                reference_tile_index=candidate.spec.tile_index,
                brightness=candidate.brightness,
                detail=candidate.detail,
            )
        )
    return probes


def _nearest_tile(probe: Probe, image: Image.Image, tile_pairs: Sequence[tuple[TileSpec, Image.Image]]) -> tuple[TileSpec, Image.Image]:
    def distance(pair: tuple[TileSpec, Image.Image]) -> tuple[float, int]:
        spec, _ = pair
        x0, y0, x1, y1 = spec.core_box
        cx = ((x0 + x1) / 2.0) / image.width
        cy = ((y0 + y1) / 2.0) / image.height
        return ((cx - probe.center_x) ** 2 + (cy - probe.center_y) ** 2, spec.tile_index)

    return min(tile_pairs, key=distance)


def _inference_filename(
    *,
    source: SourceImage,
    pre_mp: float | None,
    noise: float,
    tile_index: int,
    resolution: int,
) -> str:
    token = json.dumps(
        {
            "source": source.source_id,
            "pre": pre_mp,
            "noise": noise,
            "tile": tile_index,
            "resolution": resolution,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return f"probe_{digest}.png"


def _case_dir(output_root: Path, source: SourceImage, probe: Probe, noise: float) -> Path:
    return output_root / "reports" / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}"


def _input_core_path(output_root: Path, source: SourceImage, probe: Probe, pre_mp: float | None, noise: float) -> Path:
    pre = "native" if pre_mp is None else f"{_fmt_float(pre_mp)}mp"
    return _case_dir(output_root, source, probe, noise) / "inputs" / f"pre-{pre}.png"


def _output_core_path(output_root: Path, source: SourceImage, probe: Probe, variant: Variant) -> Path:
    return _case_dir(output_root, source, probe, variant.noise) / "results" / f"{variant.variant_id}.png"


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", compress_level=4)


def _fit_cell(image: Image.Image, size: int) -> Image.Image:
    rgb = image.convert("RGB")
    rgb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2))
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _sheet_cell(path: Path | None, *, cell_size: int, footer: str, missing: str = "MISSING") -> Image.Image:
    footer_h = 56
    canvas = Image.new("RGB", (cell_size, cell_size + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    if path is not None and path.is_file():
        with Image.open(path) as opened:
            rendered = _fit_cell(opened, cell_size)
        canvas.paste(rendered, (0, 0))
    else:
        draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline="gray", width=1)
        draw.multiline_text((10, 10), missing, fill="black", font=_font(13), spacing=3)
    draw.rectangle((0, 0, cell_size - 1, cell_size + footer_h - 1), outline="gray", width=1)
    draw.multiline_text((7, cell_size + 5), footer, fill="black", font=_font(12), spacing=2)
    return canvas


def _make_probe_sheet(
    *,
    source: SourceImage,
    probe: Probe,
    noise: float,
    pre_values: Sequence[float | None],
    scales: Sequence[float],
    output_root: Path,
    result_index: dict[tuple[str, str, float | None, float, float], PlannedProbeResult],
    cell_size: int,
    target: Path,
    max_output_mp: float,
) -> None:
    row_label_w = 118
    header_h = 66
    footer_h = 56
    cell_h = cell_size + footer_h
    cols = 1 + len(scales)
    width = row_label_w + cols * cell_size
    height = header_h + len(pre_values) * cell_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 8),
        f"{source.relative.name} | {source.megapixels:.2f} MP | {probe.probe_id} ({probe.label}) | noise={noise:g}",
        fill="black",
        font=_font(16),
    )
    draw.text(
        (10, 32),
        f"probe center=({probe.center_x:.3f}, {probe.center_y:.3f}) | reference pre={_pre_label(probe.reference_pre_megapixels)}",
        fill="black",
        font=_font(12),
    )
    headers = ["ACTUAL INPUT", *[f"{scale:g}x" for scale in scales]]
    for col, header in enumerate(headers):
        draw.text((row_label_w + col * cell_size + 7, 48), header, fill="black", font=_font(12))

    for row, pre_mp in enumerate(pre_values):
        y = header_h + row * cell_h
        draw.multiline_text((7, y + 10), f"pre\n{_pre_label(pre_mp)}", fill="black", font=_font(13), spacing=3)
        input_path = _input_core_path(output_root, source, probe, pre_mp, noise)
        any_result = next(
            (
                result_index[(source.source_id, probe.probe_id, pre_mp, scale, noise)]
                for scale in scales
                if (source.source_id, probe.probe_id, pre_mp, scale, noise) in result_index
            ),
            None,
        )
        if any_result:
            input_footer = (
                f"{any_result.processed_width}x{any_result.processed_height}\n"
                f"tile {any_result.selected_tile_index + 1}/{any_result.total_tiles}"
            )
            input_cell = _sheet_cell(input_path, cell_size=cell_size, footer=input_footer)
        else:
            input_cell = _sheet_cell(None, cell_size=cell_size, footer="not applicable", missing="SKIP")
        sheet.paste(input_cell, (row_label_w, y))

        for col, scale in enumerate(scales, start=1):
            key = (source.source_id, probe.probe_id, pre_mp, scale, noise)
            result = result_index.get(key)
            if result is None:
                variant = Variant(variant_id(source.bucket, pre_mp, scale, noise), source.bucket, pre_mp, scale, noise)
                applies, _ = variant_applies(source, variant, max_output_mp)
                footer = f"scale={scale:g}x"
                cell = _sheet_cell(None, cell_size=cell_size, footer=footer, missing="SKIP" if not applies else "MISSING")
            else:
                predicted = _predicted_output_mp(source.megapixels, pre_mp, scale)
                footer = f"scale={scale:g}x  r{result.group_resolution}\npred full≈{predicted:.2f} MP"
                cell = _sheet_cell(result.output_core_path, cell_size=cell_size, footer=footer)
            sheet.paste(cell, (row_label_w + col * cell_size, y))

    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", compress_level=4)


def _serialize_probe(probe: Probe) -> dict:
    return {
        "probe_id": probe.probe_id,
        "label": probe.label,
        "center_x": probe.center_x,
        "center_y": probe.center_y,
        "reference_pre_megapixels": "native" if probe.reference_pre_megapixels is None else probe.reference_pre_megapixels,
        "reference_tile_index": probe.reference_tile_index,
        "brightness": probe.brightness,
        "detail": probe.detail,
    }


def _write_results_csv(output_root: Path, rows: Sequence[dict]) -> None:
    path = output_root / "results.csv"
    fieldnames = [
        "source_id",
        "source",
        "bucket",
        "source_mp",
        "probe_id",
        "probe_label",
        "probe_center_x",
        "probe_center_y",
        "pre_mp",
        "scale",
        "noise",
        "processed_width",
        "processed_height",
        "total_tiles",
        "selected_tile_index",
        "group_resolution",
        "predicted_full_output_mp",
        "input_core",
        "output_core",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(
    output_root: Path,
    *,
    sources: Sequence[SourceImage],
    probes_by_source: dict[str, list[Probe]],
    noise_values: Sequence[float],
    plan_only: bool,
) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 probe sweep</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px}"
        ".card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}.meta{color:#555}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>",
        "</head><body><h1>SeedVR2 probe sweep</h1>",
        f"<p class='meta'>seedvr2-tile {html.escape(__version__)} · model 3B FP8 · {'plan only' if plan_only else 'probe inference complete'}</p>",
        "<p>This report preprocesses the full source first, then runs SeedVR2 only on representative tiles from the actual post-preprocess tile grid.</p>",
    ]
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
                rel = Path("reports") / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}" / "comparison.png"
                parts.append(f"<h4>noise={noise:g}</h4><a href='{html.escape(rel.as_posix())}'><img src='{html.escape(rel.as_posix())}'></a>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_root / "index.html").write_text("".join(parts), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-sweep",
        description="Preprocess full images, then sweep SeedVR2 on representative post-preprocess probe tiles.",
    )
    parser.add_argument("input", help="input image, directory or glob")
    parser.add_argument("output", type=Path, help="experiment output directory")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--only-bucket", action="append", choices=BUCKETS, dest="only_buckets")
    parser.add_argument("--small-max", type=float, default=1.25)
    parser.add_argument("--medium-max", type=float, default=4.0)
    parser.add_argument("--pre-small", type=_parse_pre_values, default=DEFAULT_PRESETS["small"])
    parser.add_argument("--pre-medium", type=_parse_pre_values, default=DEFAULT_PRESETS["medium"])
    parser.add_argument("--pre-large", type=_parse_pre_values, default=DEFAULT_PRESETS["large"])
    parser.add_argument("--scales", type=lambda value: _parse_csv_floats(value, allow_zero=False), default=DEFAULT_SCALES)
    parser.add_argument("--noise-values", type=_parse_csv_floats, default=DEFAULT_NOISE)
    parser.add_argument("--max-output-mp", type=float, default=20.0)
    parser.add_argument("--probe-tiles", type=int, default=3, help="representative tiles per source (default: 3: detail, dark, center)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fbcnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jpeg-quality", default="auto")
    parser.add_argument("--fbcnn-device", default="auto")
    parser.add_argument("--pre-resample", choices=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--tile-upscale-resolution", type=int, default=2048)
    parser.add_argument("--strategy", choices=["chess", "linear"], default="chess")
    parser.add_argument("--attention-mode", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], default="sdpa")
    parser.add_argument("--color-correction", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], default="lab")
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--seedvr2-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--cuda-device")
    parser.add_argument("--no-model-download", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.probe_tiles <= 0:
        raise SystemExit("--probe-tiles must be > 0")
    if args.samples_per_bucket <= 0:
        raise SystemExit("--samples-per-bucket must be > 0")
    if args.small_max <= 0 or args.medium_max <= args.small_max:
        raise SystemExit("bucket thresholds must satisfy 0 < small-max < medium-max")
    if args.max_output_mp < 0:
        raise SystemExit("--max-output-mp must be >= 0")

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    only_buckets = set(args.only_buckets or BUCKETS)
    sources = _discover_sources(
        args.input,
        recursive=args.recursive,
        small_max=args.small_max,
        medium_max=args.medium_max,
    )
    if not sources:
        raise SystemExit("no input images found")
    sources = _mark_selected(
        sources,
        samples_per_bucket=args.samples_per_bucket,
        all_images=args.all_images,
        only_buckets=only_buckets,
    )
    selected_sources = [source for source in sources if source.selected]
    pre_by_bucket = {"small": args.pre_small, "medium": args.pre_medium, "large": args.pre_large}

    print(f"Discovered {len(sources)} image(s); selected {len(selected_sources)}.")
    probes_by_source: dict[str, list[Probe]] = {}
    for source_index, source in enumerate(selected_sources):
        probes = select_probes(
            source,
            pre_values=pre_by_bucket[source.bucket],
            source_index=source_index,
            args=args,
        )
        probes_by_source[source.source_id] = probes
        labels = ", ".join(f"{probe.probe_id}@({probe.center_x:.2f},{probe.center_y:.2f})" for probe in probes)
        print(f"  {source.relative}: {len(probes)} probe(s): {labels}")
    release_fbcnn()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seedvr2_tile_version": __version__,
        "model": "3b",
        "mode": "post-preprocess-probe-tiles",
        "settings": {
            "probe_tiles": args.probe_tiles,
            "small_max": args.small_max,
            "medium_max": args.medium_max,
            "pre_by_bucket": {
                bucket: ["native" if value is None else value for value in values]
                for bucket, values in pre_by_bucket.items()
            },
            "scales": list(args.scales),
            "noise_values": list(args.noise_values),
            "max_output_mp": args.max_output_mp,
            "seed": args.seed,
            "tile": args.tile,
            "overlap": args.overlap,
            "tile_upscale_resolution": args.tile_upscale_resolution,
            "strategy": args.strategy,
            "attention_mode": args.attention_mode,
            "color_correction": args.color_correction,
        },
        "sources": [
            {
                "source_id": source.source_id,
                "source": source.relative.as_posix(),
                "width": source.width,
                "height": source.height,
                "megapixels": source.megapixels,
                "bucket": source.bucket,
                "selected": source.selected,
                "probes": [_serialize_probe(probe) for probe in probes_by_source.get(source.source_id, [])],
            }
            for source in sources
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.plan_only:
        _write_html(
            output_root,
            sources=sources,
            probes_by_source=probes_by_source,
            noise_values=args.noise_values,
            plan_only=True,
        )
        print(f"Probe plan: {output_root / 'index.html'}")
        return 0

    backend = BackendOptions(
        seedvr2_root=resolve_seedvr2_root(args.seedvr2_root),
        seed=args.seed,
        dit_model=resolve_model_name("3b"),
        model_dir=args.model_dir,
        download_model=not args.no_model_download,
        cuda_device=args.cuda_device,
        attention_mode=args.attention_mode,
        color_correction=args.color_correction,
    )
    ensure_models(backend)

    planned_results: list[PlannedProbeResult] = []
    inference_specs: dict[tuple[str, float | None, float, int, int], TileSpec] = {}
    inference_inputs: dict[tuple[str, float | None, float, int, int], Image.Image] = {}

    with tempfile.TemporaryDirectory(prefix="seedvr2-probe-sweep-") as temp_dir_raw:
        work_root = Path(temp_dir_raw)
        for source_index, source in enumerate(selected_sources):
            probes = probes_by_source[source.source_id]
            for noise in args.noise_values:
                for pre_mp in pre_by_bucket[source.bucket]:
                    if pre_mp is not None and pre_mp >= source.megapixels * 0.98:
                        continue
                    processed = _preprocess_full(
                        source,
                        pre_mp=pre_mp,
                        noise=noise,
                        source_index=source_index,
                        args=args,
                    )
                    tile_pairs = make_tiles(
                        processed,
                        image_index=source_index,
                        tile_width=args.tile,
                        tile_height=args.tile,
                        padding=args.overlap,
                        strategy=args.strategy,
                    )
                    selected_by_probe = {probe.probe_id: _nearest_tile(probe, processed, tile_pairs) for probe in probes}

                    for probe in probes:
                        spec, _ = selected_by_probe[probe.probe_id]
                        input_core = processed.crop(spec.core_box)
                        input_path = _input_core_path(output_root, source, probe, pre_mp, noise)
                        _save_png(input_core, input_path)

                    for scale in args.scales:
                        variant = Variant(variant_id(source.bucket, pre_mp, scale, noise), source.bucket, pre_mp, scale, noise)
                        applies, _ = variant_applies(source, variant, args.max_output_mp)
                        if not applies:
                            continue
                        for probe in probes:
                            spec, tile_image = selected_by_probe[probe.probe_id]
                            process_w, process_h = spec.process_size
                            desired_resolution = _round_even(min(process_w, process_h) * scale)
                            resolution = _round_even(min(desired_resolution, args.tile_upscale_resolution))
                            inference_key = (source.source_id, pre_mp, noise, spec.tile_index, resolution)
                            if inference_key not in inference_specs:
                                filename = _inference_filename(
                                    source=source,
                                    pre_mp=pre_mp,
                                    noise=noise,
                                    tile_index=spec.tile_index,
                                    resolution=resolution,
                                )
                                saved_spec = replace(spec, filename=filename)
                                inference_specs[inference_key] = saved_spec
                                inference_inputs[inference_key] = tile_image.copy()
                            saved_spec = inference_specs[inference_key]
                            planned_results.append(
                                PlannedProbeResult(
                                    source_id=source.source_id,
                                    source_name=source.relative.as_posix(),
                                    bucket=source.bucket,
                                    probe_id=probe.probe_id,
                                    probe_label=probe.label,
                                    pre_megapixels=pre_mp,
                                    scale=scale,
                                    noise=noise,
                                    processed_width=processed.width,
                                    processed_height=processed.height,
                                    total_tiles=len(tile_pairs),
                                    selected_tile_index=spec.tile_index,
                                    selected_filename=saved_spec.filename,
                                    group_resolution=resolution,
                                    input_core_path=_input_core_path(output_root, source, probe, pre_mp, noise),
                                    output_core_path=_output_core_path(output_root, source, probe, variant),
                                )
                            )
        release_fbcnn()

        groups: dict[int, list[tuple[tuple[str, float | None, float, int, int], TileSpec, Image.Image]]] = {}
        for key, spec in inference_specs.items():
            groups.setdefault(key[-1], []).append((key, spec, inference_inputs[key]))

        total_requested = len(planned_results)
        unique_inferences = len(inference_specs)
        print(f"Probe result cells: {total_requested}; unique SeedVR2 tile inferences: {unique_inferences}")
        for resolution, entries in sorted(groups.items()):
            input_dir = work_root / f"r{resolution}" / "input"
            output_dir = work_root / f"r{resolution}" / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            for _, spec, tile_image in entries:
                tile_image.save(input_dir / spec.filename, format="PNG")
            print(f"Group r{resolution}: {len(entries)} unique probe tile(s)")
            run_group(backend, input_dir=input_dir, output_dir=output_dir, resolution=resolution)

        result_index: dict[tuple[str, str, float | None, float, float], PlannedProbeResult] = {}
        rows: list[dict] = []
        source_by_id = {source.source_id: source for source in selected_sources}
        probe_by_key = {
            (source_id, probe.probe_id): probe
            for source_id, probes in probes_by_source.items()
            for probe in probes
        }
        for planned in planned_results:
            source = source_by_id[planned.source_id]
            probe = probe_by_key[(planned.source_id, planned.probe_id)]
            inference_key = (
                planned.source_id,
                planned.pre_megapixels,
                planned.noise,
                planned.selected_tile_index,
                planned.group_resolution,
            )
            spec = inference_specs[inference_key]
            processed_path = work_root / f"r{planned.group_resolution}" / "output" / spec.filename
            if not processed_path.is_file():
                raise FileNotFoundError(f"SeedVR2 did not produce expected probe tile: {processed_path}")
            with Image.open(processed_path) as processed_tile:
                core = extract_core(spec, processed_tile, scale=planned.scale)
            _save_png(core, planned.output_core_path)
            key = (planned.source_id, planned.probe_id, planned.pre_megapixels, planned.scale, planned.noise)
            result_index[key] = planned
            rows.append(
                {
                    "source_id": source.source_id,
                    "source": source.relative.as_posix(),
                    "bucket": source.bucket,
                    "source_mp": f"{source.megapixels:.6f}",
                    "probe_id": probe.probe_id,
                    "probe_label": probe.label,
                    "probe_center_x": f"{probe.center_x:.6f}",
                    "probe_center_y": f"{probe.center_y:.6f}",
                    "pre_mp": "native" if planned.pre_megapixels is None else f"{planned.pre_megapixels:.6f}",
                    "scale": f"{planned.scale:.6f}",
                    "noise": f"{planned.noise:.6f}",
                    "processed_width": planned.processed_width,
                    "processed_height": planned.processed_height,
                    "total_tiles": planned.total_tiles,
                    "selected_tile_index": planned.selected_tile_index,
                    "group_resolution": planned.group_resolution,
                    "predicted_full_output_mp": f"{_predicted_output_mp(source.megapixels, planned.pre_megapixels, planned.scale):.6f}",
                    "input_core": planned.input_core_path.relative_to(output_root).as_posix(),
                    "output_core": planned.output_core_path.relative_to(output_root).as_posix(),
                }
            )

        _write_results_csv(output_root, rows)
        for source in selected_sources:
            for probe in probes_by_source[source.source_id]:
                for noise in args.noise_values:
                    target = _case_dir(output_root, source, probe, noise) / "comparison.png"
                    _make_probe_sheet(
                        source=source,
                        probe=probe,
                        noise=noise,
                        pre_values=pre_by_bucket[source.bucket],
                        scales=args.scales,
                        output_root=output_root,
                        result_index=result_index,
                        cell_size=args.cell_size,
                        target=target,
                        max_output_mp=args.max_output_mp,
                    )

    manifest["summary"] = {
        "probe_result_cells": len(planned_results),
        "unique_seedvr2_tile_inferences": len(inference_specs),
        "backend_resolutions": sorted({result.group_resolution for result in planned_results}),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_html(
        output_root,
        sources=sources,
        probes_by_source=probes_by_source,
        noise_values=args.noise_values,
        plan_only=False,
    )
    print(f"Probe sweep report: {output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
