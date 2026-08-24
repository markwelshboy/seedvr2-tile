from __future__ import annotations

import argparse
import csv
import html
import json
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from . import __version__
from . import probe_sweep as _base
from .backend import BackendOptions, ensure_models, resolve_model_name, resolve_seedvr2_root, run_group
from .fbcnn import release_fbcnn
from .stitching import Stitcher
from .sweep import (
    BUCKETS,
    SourceImage,
    Variant,
    _discover_sources,
    _fmt_float,
    _mark_selected,
    _predicted_output_mp,
    _pre_label,
    variant_applies,
    variant_id,
)
from .tiling import TileSpec, make_tiles


SceneWindow = tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneResult:
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
    tile_indices: tuple[int, ...]
    group_resolution: int
    capture_window: SceneWindow
    input_scene_path: Path
    output_scene_path: Path


def _round_even(value: float) -> int:
    return max(16, int(round(value / 2.0) * 2))


def _pre_token(pre_mp: float | None) -> str:
    return "native" if pre_mp is None else f"{_fmt_float(pre_mp)}mp"


def _case_dir(root: Path, source: SourceImage, probe: _base.Probe, noise: float) -> Path:
    return root / "reports" / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}"


def _input_scene_path(root: Path, source: SourceImage, probe: _base.Probe, pre_mp: float | None, noise: float) -> Path:
    return _case_dir(root, source, probe, noise) / "inputs" / f"pre-{_pre_token(pre_mp)}.png"


def _output_scene_path(root: Path, source: SourceImage, probe: _base.Probe, variant: Variant) -> Path:
    return _case_dir(root, source, probe, variant.noise) / "results" / f"{variant.variant_id}.png"


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", compress_level=4)


def _fit_axis(center: float, half: float) -> tuple[float, float]:
    lo, hi = center - half, center + half
    if lo < 0.0:
        hi -= lo
        lo = 0.0
    if hi > 1.0:
        lo -= hi - 1.0
        hi = 1.0
    return max(0.0, lo), min(1.0, hi)


def _window_from_reference(
    probe: _base.Probe,
    reference: Image.Image,
    reference_spec: TileSpec,
    fraction: float,
) -> SceneWindow:
    x0, y0, x1, y1 = reference_spec.core_box
    side = min(x1 - x0, y1 - y0) * fraction
    if side <= 0:
        raise RuntimeError("invalid zero-sized probe scene window")
    nx0, nx1 = _fit_axis(probe.center_x, side / (2.0 * reference.width))
    ny0, ny1 = _fit_axis(probe.center_y, side / (2.0 * reference.height))
    return (nx0, ny0, nx1, ny1)


def _display_window(capture: SceneWindow, probe: _base.Probe, capture_fraction: float, display_fraction: float) -> SceneWindow:
    ratio = min(1.0, display_fraction / capture_fraction)
    cx0, cy0, cx1, cy1 = capture
    width = (cx1 - cx0) * ratio
    height = (cy1 - cy0) * ratio
    x0 = min(max(probe.center_x - width / 2.0, cx0), cx1 - width)
    y0 = min(max(probe.center_y - height / 2.0, cy0), cy1 - height)
    return (x0, y0, x0 + width, y0 + height)


def _crop_normalized(image: Image.Image, window: SceneWindow, output_size: tuple[int, int] | None = None) -> Image.Image:
    x0, y0, x1, y1 = window
    box = (x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height)
    if output_size is None:
        output_size = (
            max(1, round((x1 - x0) * image.width)),
            max(1, round((y1 - y0) * image.height)),
        )
    return image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS, box=box)


def _crop_scene_capture(image: Image.Image, capture: SceneWindow, display: SceneWindow, size: int) -> Image.Image:
    cx0, cy0, cx1, cy1 = capture
    dx0, dy0, dx1, dy1 = display
    cw = max(1e-12, cx1 - cx0)
    ch = max(1e-12, cy1 - cy0)
    local = (
        (dx0 - cx0) / cw,
        (dy0 - cy0) / ch,
        (dx1 - cx0) / cw,
        (dy1 - cy0) / ch,
    )
    return _crop_normalized(image, local, (size, size))


def _intersects(box: tuple[int, int, int, int], window: SceneWindow, width: int, height: int) -> bool:
    wx0, wy0, wx1, wy1 = window
    gx0, gy0, gx1, gy1 = wx0 * width, wy0 * height, wx1 * width, wy1 * height
    x0, y0, x1, y1 = box
    return x1 > gx0 and x0 < gx1 and y1 > gy0 and y0 < gy1


def _tiles_for_window(
    tile_pairs: Sequence[tuple[TileSpec, Image.Image]],
    window: SceneWindow,
    width: int,
    height: int,
) -> list[tuple[TileSpec, Image.Image]]:
    selected = [pair for pair in tile_pairs if _intersects(pair[0].source_box, window, width, height)]
    if not selected:
        raise RuntimeError("no spatial tiles intersect probe scene window")
    return selected


def _capture_windows_for_source(
    source: SourceImage,
    probes: Sequence[_base.Probe],
    *,
    source_index: int,
    args: argparse.Namespace,
) -> dict[str, SceneWindow]:
    if not probes:
        return {}
    reference_pre = probes[0].reference_pre_megapixels
    reference = _base._preprocess_full(
        source,
        pre_mp=reference_pre,
        noise=0.0,
        source_index=source_index,
        args=args,
    )
    pairs = make_tiles(
        reference,
        image_index=source_index,
        tile_width=args.tile,
        tile_height=args.tile,
        padding=args.overlap,
        strategy=args.strategy,
    )
    by_index = {spec.tile_index: spec for spec, _ in pairs}
    windows: dict[str, SceneWindow] = {}
    for probe in probes:
        spec = by_index.get(probe.reference_tile_index)
        if spec is None:
            raise RuntimeError(f"missing reference tile {probe.reference_tile_index} for {probe.probe_id}")
        windows[probe.probe_id] = _window_from_reference(
            probe,
            reference,
            spec,
            args.probe_capture_fraction,
        )
    return windows


def _render_reports(
    root: Path,
    *,
    sources: Sequence[SourceImage],
    probes_by_source: dict[str, list[_base.Probe]],
    capture_windows: dict[tuple[str, str], SceneWindow],
    pre_by_bucket: dict[str, Sequence[float | None]],
    scales: Sequence[float],
    noise_values: Sequence[float],
    capture_fraction: float,
    comparison_fraction: float,
    cell_size: int,
    result_index: dict[tuple[str, str, float | None, float, float], SceneResult],
    max_output_mp: float,
) -> int:
    sheets = 0
    for source in sources:
        if not source.selected:
            continue
        for probe in probes_by_source.get(source.source_id, []):
            capture = capture_windows[(source.source_id, probe.probe_id)]
            display = _display_window(capture, probe, capture_fraction, comparison_fraction)
            for noise in noise_values:
                case = _case_dir(root, source, probe, noise)
                crop_dir = case / "crops"
                crop_dir.mkdir(parents=True, exist_ok=True)
                for stale in crop_dir.glob("*.png"):
                    stale.unlink()

                rows: list[list[tuple[Path | None, str]]] = []
                for row_index, pre_mp in enumerate(pre_by_bucket[source.bucket], start=1):
                    any_result = next(
                        (
                            result_index[(source.source_id, probe.probe_id, pre_mp, scale, noise)]
                            for scale in scales
                            if (source.source_id, probe.probe_id, pre_mp, scale, noise) in result_index
                        ),
                        None,
                    )
                    row: list[tuple[Path | None, str]] = []
                    if any_result is None:
                        row.append((None, "not applicable"))
                    else:
                        input_path = any_result.input_scene_path
                        with Image.open(input_path) as opened:
                            crop = _crop_scene_capture(opened, capture, display, cell_size)
                        crop_path = crop_dir / f"{row_index:02d}_pre-{_pre_token(pre_mp)}__00-input.png"
                        _save_png(crop, crop_path)
                        row.append((crop_path, f"{any_result.processed_width}x{any_result.processed_height}\ntiles {','.join(str(i + 1) for i in any_result.tile_indices)}"))

                    for col_index, scale in enumerate(scales, start=1):
                        result = result_index.get((source.source_id, probe.probe_id, pre_mp, scale, noise))
                        if result is None:
                            variant = Variant(variant_id(source.bucket, pre_mp, scale, noise), source.bucket, pre_mp, scale, noise)
                            applies, _ = variant_applies(source, variant, max_output_mp)
                            row.append((None, "SKIP" if not applies else "MISSING"))
                            continue
                        with Image.open(result.output_scene_path) as opened:
                            crop = _crop_scene_capture(opened, capture, display, cell_size)
                        crop_path = crop_dir / f"{row_index:02d}_pre-{_pre_token(pre_mp)}__{col_index:02d}-scale-{_fmt_float(scale)}x.png"
                        _save_png(crop, crop_path)
                        pred = _predicted_output_mp(source.megapixels, pre_mp, scale)
                        row.append((crop_path, f"scale={scale:g}x r{result.group_resolution}\npred full≈{pred:.2f} MP"))
                    rows.append(row)

                row_label_w, header_h, footer_h = 118, 66, 56
                cols = 1 + len(scales)
                sheet = Image.new("RGB", (row_label_w + cols * cell_size, header_h + len(rows) * (cell_size + footer_h)), "white")
                draw = ImageDraw.Draw(sheet)
                draw.text((10, 8), f"{source.relative.name} | {source.megapixels:.2f} MP | {probe.probe_id} ({probe.label}) | noise={noise:g}", fill="black", font=_base._font(16))
                draw.text((10, 32), f"scene-registered window | probe=({probe.center_x:.3f}, {probe.center_y:.3f}) | display={comparison_fraction:g} ref-core", fill="black", font=_base._font(12))
                headers = ["ACTUAL INPUT", *[f"{scale:g}x" for scale in scales]]
                for col, header in enumerate(headers):
                    draw.text((row_label_w + col * cell_size + 7, 48), header, fill="black", font=_base._font(12))
                for row_i, (pre_mp, row) in enumerate(zip(pre_by_bucket[source.bucket], rows)):
                    y = header_h + row_i * (cell_size + footer_h)
                    draw.multiline_text((7, y + 10), f"pre\n{_pre_label(pre_mp)}", fill="black", font=_base._font(13), spacing=3)
                    for col, (path, footer) in enumerate(row):
                        canvas = Image.new("RGB", (cell_size, cell_size + footer_h), "white")
                        cdraw = ImageDraw.Draw(canvas)
                        if path is not None and path.is_file():
                            with Image.open(path) as opened:
                                canvas.paste(opened.convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS), (0, 0))
                        else:
                            cdraw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline="gray", width=1)
                            cdraw.text((10, 10), footer, fill="black", font=_base._font(13))
                        cdraw.rectangle((0, 0, cell_size - 1, cell_size + footer_h - 1), outline="gray", width=1)
                        cdraw.multiline_text((7, cell_size + 5), footer, fill="black", font=_base._font(12), spacing=2)
                        sheet.paste(canvas, (row_label_w + col * cell_size, y))
                case.mkdir(parents=True, exist_ok=True)
                sheet.save(case / "comparison.png", format="PNG", compress_level=4)
                (case / "scene-window.json").write_text(
                    json.dumps(
                        {
                            "coordinate_space": "normalized-full-preprocessed-image",
                            "probe_center": [probe.center_x, probe.center_y],
                            "capture_fraction": capture_fraction,
                            "display_fraction": comparison_fraction,
                            "capture_window": {"x0": capture[0], "y0": capture[1], "x1": capture[2], "y1": capture[3]},
                            "display_window": {"x0": display[0], "y0": display[1], "x1": display[2], "y1": display[3]},
                            "note": "All crops/ PNGs render the same display_window; captures are stitched from every contributing tile.",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                sheets += 1

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 scene probe sweep</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px}.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}.meta{color:#555}.sheet-scroll{overflow-x:auto;max-width:100%}.sheet{max-width:none;height:auto;border:1px solid #ddd;display:block}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        "<h1>SeedVR2 scene probe sweep</h1>",
        f"<p class='meta'>seedvr2-tile {html.escape(__version__)} · scene-registered multi-tile probes</p>",
        "<p>Each probe uses one normalized scene rectangle. Every spatial tile whose overlap region contributes to that rectangle is inferred and combined with the normal multiband stitcher before the comparison crop is taken.</p>",
    ]
    for source in sources:
        if not source.selected:
            continue
        parts.append(f"<div class='card'><h2>{html.escape(source.relative.as_posix())}</h2><p class='meta'>{source.width}×{source.height} · {source.megapixels:.2f} MP · {source.bucket}</p>")
        for probe in probes_by_source.get(source.source_id, []):
            parts.append(f"<h3>{html.escape(probe.probe_id)} — {html.escape(probe.label)}</h3><p class='meta'>normalized probe ({probe.center_x:.3f}, {probe.center_y:.3f})</p>")
            for noise in noise_values:
                rel = Path("reports") / source.bucket / source.source_id / probe.probe_id / f"noise-{_fmt_float(noise)}" / "comparison.png"
                crops = rel.parent / "crops"
                parts.append(f"<h4>noise={noise:g}</h4><p class='meta'>Overlay-ready crops: <code>{html.escape(crops.as_posix())}/</code></p><div class='sheet-scroll'><a href='{html.escape(rel.as_posix())}'><img class='sheet' src='{html.escape(rel.as_posix())}'></a></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (root / "index.html").write_text("".join(parts), encoding="utf-8")
    return sheets


def _write_results_csv(root: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "source_id", "source", "bucket", "source_mp", "probe_id", "probe_label", "pre_mp", "scale", "noise",
        "processed_width", "processed_height", "total_tiles", "tile_indices", "group_resolution",
        "capture_x0", "capture_y0", "capture_x1", "capture_y1", "predicted_full_output_mp", "input_scene", "output_scene",
    ]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = _base._build_parser()
    parser.prog = "seedvr2-sweep"
    parser.description = "Run scene-registered SeedVR2 probe sweeps with exact multi-tile stitching around each probe."
    parser.set_defaults(cell_size=420)
    parser.add_argument("--probe-capture-fraction", type=float, default=0.50, help="reference-core fraction captured/inferred around each probe (default: 0.50)")
    parser.add_argument("--comparison-crop-fraction", type=float, default=0.50, help="reference-core fraction shown/exported in reports; may be smaller than capture (default: 0.50)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.probe_tiles <= 0 or args.samples_per_bucket <= 0:
        raise SystemExit("--probe-tiles and --samples-per-bucket must be > 0")
    if not 0 < args.probe_capture_fraction <= 1:
        raise SystemExit("--probe-capture-fraction must be in (0, 1]")
    if not 0 < args.comparison_crop_fraction <= 1:
        raise SystemExit("--comparison-crop-fraction must be in (0, 1]")
    if args.comparison_crop_fraction > args.probe_capture_fraction:
        args.probe_capture_fraction = args.comparison_crop_fraction
    if args.small_max <= 0 or args.medium_max <= args.small_max:
        raise SystemExit("bucket thresholds must satisfy 0 < small-max < medium-max")
    if args.max_output_mp < 0:
        raise SystemExit("--max-output-mp must be >= 0")

    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    only_buckets = set(args.only_buckets or BUCKETS)
    sources = _discover_sources(args.input, recursive=args.recursive, small_max=args.small_max, medium_max=args.medium_max)
    if not sources:
        raise SystemExit("no input images found")
    sources = _mark_selected(sources, samples_per_bucket=args.samples_per_bucket, all_images=args.all_images, only_buckets=only_buckets)
    selected_sources = [source for source in sources if source.selected]
    pre_by_bucket = {"small": args.pre_small, "medium": args.pre_medium, "large": args.pre_large}

    print(f"Discovered {len(sources)} image(s); selected {len(selected_sources)}.")
    probes_by_source: dict[str, list[_base.Probe]] = {}
    capture_windows: dict[tuple[str, str], SceneWindow] = {}
    for source_index, source in enumerate(selected_sources):
        probes = _base.select_probes(source, pre_values=pre_by_bucket[source.bucket], source_index=source_index, args=args)
        probes_by_source[source.source_id] = probes
        windows = _capture_windows_for_source(source, probes, source_index=source_index, args=args)
        for probe in probes:
            capture_windows[(source.source_id, probe.probe_id)] = windows[probe.probe_id]
        print(f"  {source.relative}: " + ", ".join(f"{p.probe_id}@({p.center_x:.2f},{p.center_y:.2f})" for p in probes))
    release_fbcnn()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seedvr2_tile_version": __version__,
        "model": "3b",
        "mode": "post-preprocess-probe-scenes-v2",
        "settings": {
            "probe_tiles": args.probe_tiles,
            "probe_capture_fraction": args.probe_capture_fraction,
            "comparison_crop_fraction": args.comparison_crop_fraction,
            "small_max": args.small_max,
            "medium_max": args.medium_max,
            "pre_by_bucket": {bucket: ["native" if value is None else value for value in values] for bucket, values in pre_by_bucket.items()},
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
        "sources": [],
    }
    for source in sources:
        entry = {
            "source_id": source.source_id,
            "source": source.relative.as_posix(),
            "width": source.width,
            "height": source.height,
            "megapixels": source.megapixels,
            "bucket": source.bucket,
            "selected": source.selected,
            "probes": [],
        }
        for probe in probes_by_source.get(source.source_id, []):
            item = _base._serialize_probe(probe)
            window = capture_windows[(source.source_id, probe.probe_id)]
            item["capture_window"] = {"x0": window[0], "y0": window[1], "x1": window[2], "y1": window[3]}
            entry["probes"].append(item)
        manifest["sources"].append(entry)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.plan_only:
        print(f"Probe plan manifest: {root / 'manifest.json'}")
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

    planned: list[SceneResult] = []
    inference_specs: dict[tuple[str, float | None, float, int, int], TileSpec] = {}
    inference_inputs: dict[tuple[str, float | None, float, int, int], Image.Image] = {}

    with tempfile.TemporaryDirectory(prefix="seedvr2-scene-probe-") as temp_raw:
        work_root = Path(temp_raw)
        for source_index, source in enumerate(selected_sources):
            probes = probes_by_source[source.source_id]
            for noise in args.noise_values:
                for pre_mp in pre_by_bucket[source.bucket]:
                    if pre_mp is not None and pre_mp >= source.megapixels * 0.98:
                        continue
                    processed = _base._preprocess_full(source, pre_mp=pre_mp, noise=noise, source_index=source_index, args=args)
                    tile_pairs = make_tiles(processed, image_index=source_index, tile_width=args.tile, tile_height=args.tile, padding=args.overlap, strategy=args.strategy)

                    selected_tiles: dict[str, list[tuple[TileSpec, Image.Image]]] = {}
                    for probe in probes:
                        capture = capture_windows[(source.source_id, probe.probe_id)]
                        selected_tiles[probe.probe_id] = _tiles_for_window(tile_pairs, capture, processed.width, processed.height)
                        _save_png(_crop_normalized(processed, capture), _input_scene_path(root, source, probe, pre_mp, noise))

                    for scale in args.scales:
                        variant = Variant(variant_id(source.bucket, pre_mp, scale, noise), source.bucket, pre_mp, scale, noise)
                        applies, _ = variant_applies(source, variant, args.max_output_mp)
                        if not applies:
                            continue
                        for probe in probes:
                            pairs = selected_tiles[probe.probe_id]
                            resolutions: set[int] = set()
                            tile_indices: list[int] = []
                            for spec, tile_image in pairs:
                                desired = _round_even(min(spec.process_size) * scale)
                                resolution = _round_even(min(desired, args.tile_upscale_resolution))
                                resolutions.add(resolution)
                                tile_indices.append(spec.tile_index)
                                key = (source.source_id, pre_mp, noise, spec.tile_index, resolution)
                                if key not in inference_specs:
                                    filename = _base._inference_filename(source=source, pre_mp=pre_mp, noise=noise, tile_index=spec.tile_index, resolution=resolution)
                                    inference_specs[key] = replace(spec, filename=filename)
                                    inference_inputs[key] = tile_image.copy()
                            if len(resolutions) != 1:
                                raise RuntimeError("probe tiles unexpectedly require mixed backend resolutions")
                            resolution = next(iter(resolutions))
                            planned.append(
                                SceneResult(
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
                                    tile_indices=tuple(sorted(tile_indices)),
                                    group_resolution=resolution,
                                    capture_window=capture_windows[(source.source_id, probe.probe_id)],
                                    input_scene_path=_input_scene_path(root, source, probe, pre_mp, noise),
                                    output_scene_path=_output_scene_path(root, source, probe, variant),
                                )
                            )
        release_fbcnn()

        groups: dict[int, list[tuple[tuple[str, float | None, float, int, int], TileSpec, Image.Image]]] = {}
        for key, spec in inference_specs.items():
            groups.setdefault(key[-1], []).append((key, spec, inference_inputs[key]))
        print(f"Probe result cells: {len(planned)}; unique SeedVR2 tile inferences: {len(inference_specs)}")
        for resolution, entries in sorted(groups.items()):
            input_dir = work_root / f"r{resolution}" / "input"
            output_dir = work_root / f"r{resolution}" / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            for _, spec, tile_image in entries:
                tile_image.save(input_dir / spec.filename, format="PNG")
            print(f"Group r{resolution}: {len(entries)} unique tile(s)")
            run_group(backend, input_dir=input_dir, output_dir=output_dir, resolution=resolution)

        rows: list[dict] = []
        result_index: dict[tuple[str, str, float | None, float, float], SceneResult] = {}
        source_by_id = {source.source_id: source for source in selected_sources}
        for result in planned:
            stitcher = Stitcher(
                max(1, round(result.processed_width * result.scale)),
                max(1, round(result.processed_height * result.scale)),
                method="multiband",
            )
            for tile_index in result.tile_indices:
                key = (result.source_id, result.pre_megapixels, result.noise, tile_index, result.group_resolution)
                spec = inference_specs[key]
                processed_path = work_root / f"r{result.group_resolution}" / "output" / spec.filename
                if not processed_path.is_file():
                    raise FileNotFoundError(f"SeedVR2 did not produce expected probe tile: {processed_path}")
                with Image.open(processed_path) as processed_tile:
                    stitcher.add(spec, processed_tile, scale=result.scale)
            stitched = stitcher.finish()
            _save_png(_crop_normalized(stitched, result.capture_window), result.output_scene_path)
            idx = (result.source_id, result.probe_id, result.pre_megapixels, result.scale, result.noise)
            result_index[idx] = result
            source = source_by_id[result.source_id]
            rows.append(
                {
                    "source_id": result.source_id,
                    "source": result.source_name,
                    "bucket": result.bucket,
                    "source_mp": f"{source.megapixels:.6f}",
                    "probe_id": result.probe_id,
                    "probe_label": result.probe_label,
                    "pre_mp": "native" if result.pre_megapixels is None else f"{result.pre_megapixels:.6f}",
                    "scale": f"{result.scale:.6f}",
                    "noise": f"{result.noise:.6f}",
                    "processed_width": result.processed_width,
                    "processed_height": result.processed_height,
                    "total_tiles": result.total_tiles,
                    "tile_indices": ",".join(str(i) for i in result.tile_indices),
                    "group_resolution": result.group_resolution,
                    "capture_x0": f"{result.capture_window[0]:.9f}",
                    "capture_y0": f"{result.capture_window[1]:.9f}",
                    "capture_x1": f"{result.capture_window[2]:.9f}",
                    "capture_y1": f"{result.capture_window[3]:.9f}",
                    "predicted_full_output_mp": f"{_predicted_output_mp(source.megapixels, result.pre_megapixels, result.scale):.6f}",
                    "input_scene": result.input_scene_path.relative_to(root).as_posix(),
                    "output_scene": result.output_scene_path.relative_to(root).as_posix(),
                }
            )

        _write_results_csv(root, rows)
        _render_reports(
            root,
            sources=sources,
            probes_by_source=probes_by_source,
            capture_windows=capture_windows,
            pre_by_bucket=pre_by_bucket,
            scales=args.scales,
            noise_values=args.noise_values,
            capture_fraction=args.probe_capture_fraction,
            comparison_fraction=args.comparison_crop_fraction,
            cell_size=args.cell_size,
            result_index=result_index,
            max_output_mp=args.max_output_mp,
        )

    manifest["summary"] = {
        "probe_result_cells": len(planned),
        "unique_seedvr2_tile_inferences": len(inference_specs),
        "backend_resolutions": sorted({result.group_resolution for result in planned}),
        "multi_tile_result_cells": sum(1 for result in planned if len(result.tile_indices) > 1),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Probe sweep report: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
