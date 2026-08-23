from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import __version__
from .cli import main as seedvr2_main
from .inputs import discover_inputs

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
BUCKETS = ("small", "medium", "large")
DEFAULT_PRESETS = {
    "small": ("native", 0.50, 0.75),
    "medium": ("native", 0.75, 1.00),
    "large": (1.00, 1.50, 2.00),
}
DEFAULT_SCALES = (1.5, 2.0, 3.0)
DEFAULT_NOISE = (0.0,)
DEFAULT_CROPS = {
    "center": (0.50, 0.50),
    "upper": (0.50, 0.30),
    "lower": (0.50, 0.70),
    "left": (0.30, 0.50),
    "right": (0.70, 0.50),
}


@dataclass(frozen=True)
class SourceImage:
    source_id: str
    source_path: Path
    relative: Path
    width: int
    height: int
    megapixels: float
    bucket: str
    selected: bool = False


@dataclass(frozen=True)
class Variant:
    variant_id: str
    bucket: str
    pre_megapixels: float | None
    scale: float
    noise: float


def megapixels_for_size(width: int, height: int) -> float:
    return (width * height) / float(1024 * 1024)


def bucket_for_megapixels(megapixels: float, small_max: float, medium_max: float) -> str:
    if megapixels < small_max:
        return "small"
    if megapixels <= medium_max:
        return "medium"
    return "large"


def _safe_stem(relative: Path, index: int) -> str:
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in relative.stem).strip("_")
    stem = stem or "image"
    return f"img{index:04d}_{stem[:48]}_{digest}"


def _spread_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    raw = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    result: list[int] = []
    for idx in raw:
        if idx not in result:
            result.append(idx)
    for idx in range(length):
        if len(result) >= count:
            break
        if idx not in result:
            result.append(idx)
    return sorted(result[:count])


def select_spread(items: Sequence[SourceImage], count: int) -> set[str]:
    ordered = sorted(items, key=lambda item: (item.megapixels, item.relative.as_posix()))
    return {ordered[idx].source_id for idx in _spread_indices(len(ordered), count)}


def _parse_csv_floats(value: str, *, allow_zero: bool = True) -> tuple[float, ...]:
    result: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        number = float(raw)
        if number < 0 or (number == 0 and not allow_zero):
            raise argparse.ArgumentTypeError(f"invalid value: {number}")
        result.append(number)
    if not result:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    return tuple(result)


def _parse_pre_values(value: str) -> tuple[float | None, ...]:
    result: list[float | None] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"native", "none", "off"}:
            result.append(None)
            continue
        number = float(token)
        if number <= 0:
            raise argparse.ArgumentTypeError("pre-megapixel values must be > 0 or 'native'")
        result.append(number)
    if not result:
        raise argparse.ArgumentTypeError("at least one pre-megapixel value is required")
    return tuple(result)


def _fmt_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def variant_id(bucket: str, pre_megapixels: float | None, scale: float, noise: float) -> str:
    pre = "native" if pre_megapixels is None else f"{_fmt_float(pre_megapixels)}mp"
    return f"{bucket}__pre-{pre}__scale-{_fmt_float(scale)}x__noise-{_fmt_float(noise)}"


def build_variants(
    bucket: str,
    pre_values: Sequence[float | None],
    scales: Sequence[float],
    noises: Sequence[float],
) -> list[Variant]:
    return [
        Variant(
            variant_id=variant_id(bucket, pre, scale, noise),
            bucket=bucket,
            pre_megapixels=pre,
            scale=scale,
            noise=noise,
        )
        for noise in noises
        for pre in pre_values
        for scale in scales
    ]


def _predicted_output_mp(source_mp: float, pre_mp: float | None, scale: float) -> float:
    processed_mp = source_mp if pre_mp is None else pre_mp
    return processed_mp * scale * scale


def variant_applies(source: SourceImage, variant: Variant, max_output_mp: float) -> tuple[bool, str | None]:
    if variant.pre_megapixels is not None and variant.pre_megapixels >= source.megapixels * 0.98:
        return False, "pre target would not downscale source"
    predicted = _predicted_output_mp(source.megapixels, variant.pre_megapixels, variant.scale)
    if max_output_mp > 0 and predicted > max_output_mp:
        return False, f"predicted output {predicted:.2f} MP exceeds cap {max_output_mp:.2f} MP"
    return True, None


def _discover_sources(
    input_spec: str,
    *,
    recursive: bool,
    small_max: float,
    medium_max: float,
) -> list[SourceImage]:
    items = discover_inputs([input_spec], recursive=recursive, extensions=IMAGE_EXTENSIONS)
    sources: list[SourceImage] = []
    for index, item in enumerate(items):
        with Image.open(item.path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
        mp = megapixels_for_size(width, height)
        sources.append(
            SourceImage(
                source_id=_safe_stem(item.relative, index),
                source_path=item.path,
                relative=item.relative,
                width=width,
                height=height,
                megapixels=mp,
                bucket=bucket_for_megapixels(mp, small_max, medium_max),
            )
        )
    return sources


def _mark_selected(
    sources: Sequence[SourceImage],
    *,
    samples_per_bucket: int,
    all_images: bool,
    only_buckets: set[str],
) -> list[SourceImage]:
    selected_ids: set[str] = set()
    for bucket in BUCKETS:
        if bucket not in only_buckets:
            continue
        candidates = [source for source in sources if source.bucket == bucket]
        if all_images:
            selected_ids.update(source.source_id for source in candidates)
        else:
            selected_ids.update(select_spread(candidates, samples_per_bucket))
    return [
        SourceImage(**{**asdict(source), "selected": source.source_id in selected_ids})
        for source in sources
    ]


def _stage_selected(sources: Sequence[SourceImage], root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for source in sources:
        if not source.selected:
            continue
        suffix = source.source_path.suffix.lower() or ".png"
        target = root / f"{source.source_id}{suffix}"
        try:
            target.hardlink_to(source.source_path)
        except OSError:
            shutil.copy2(source.source_path, target)
        staged[source.source_id] = target
    return staged


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB").copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _normalized_crop(image: Image.Image, center: tuple[float, float], fraction: float, size: tuple[int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    crop_side = max(16, int(round(min(rgb.size) * fraction)))
    cx = int(round(center[0] * rgb.width))
    cy = int(round(center[1] * rgb.height))
    left = min(max(0, cx - crop_side // 2), max(0, rgb.width - crop_side))
    top = min(max(0, cy - crop_side // 2), max(0, rgb.height - crop_side))
    crop = rgb.crop((left, top, left + crop_side, top + crop_side))
    return crop.resize(size, Image.Resampling.LANCZOS)


def _render_cell(
    image_path: Path | None,
    *,
    mode: str,
    center: tuple[float, float] | None,
    crop_fraction: float,
    cell_size: int,
    footer: str,
    missing_text: str | None = None,
) -> Image.Image:
    footer_h = 46
    canvas = Image.new("RGB", (cell_size, cell_size + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    if image_path is not None and image_path.is_file():
        with Image.open(image_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            if mode == "full":
                rendered = _fit_thumbnail(oriented, (cell_size, cell_size))
            else:
                assert center is not None
                rendered = _normalized_crop(oriented, center, crop_fraction, (cell_size, cell_size))
        canvas.paste(rendered, (0, 0))
    else:
        draw.rectangle((0, 0, cell_size - 1, cell_size - 1), outline="gray", width=1)
        text = missing_text or "MISSING"
        draw.multiline_text((12, 12), text, fill="black", font=_font(14), spacing=4)
    draw.rectangle((0, 0, cell_size - 1, cell_size + footer_h - 1), outline="gray", width=1)
    draw.multiline_text((8, cell_size + 5), footer, fill="black", font=_font(13), spacing=2)
    return canvas


def _variant_output_path(output_root: Path, source: SourceImage, variant: Variant) -> Path:
    return output_root / "results" / variant.bucket / variant.variant_id / f"{source.source_id}.png"


def _pre_label(pre_mp: float | None) -> str:
    return "native" if pre_mp is None else f"{pre_mp:g} MP"


def _make_contact_sheet(
    *,
    source: SourceImage,
    source_path: Path,
    output_root: Path,
    pre_values: Sequence[float | None],
    scales: Sequence[float],
    noise: float,
    mode: str,
    center: tuple[float, float] | None,
    crop_fraction: float,
    cell_size: int,
    max_output_mp: float,
    target: Path,
) -> None:
    row_label_w = 124
    header_h = 54
    cols = 1 + len(scales)
    cell_h = cell_size + 46
    width = row_label_w + cols * cell_size
    height = header_h + len(pre_values) * cell_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(17)
    label_font = _font(14)

    mode_label = "full image" if mode == "full" else f"{mode} crop"
    draw.text(
        (10, 8),
        f"{source.relative.name}  |  {source.megapixels:.2f} MP  |  noise={noise:g}  |  {mode_label}",
        fill="black",
        font=title_font,
    )
    headers = ["SOURCE", *[f"{scale:g}x" for scale in scales]]
    for col, header in enumerate(headers):
        x = row_label_w + col * cell_size
        draw.text((x + 8, 32), header, fill="black", font=label_font)

    for row, pre_mp in enumerate(pre_values):
        y = header_h + row * cell_h
        draw.multiline_text((8, y + 10), f"pre\n{_pre_label(pre_mp)}", fill="black", font=label_font, spacing=3)
        source_cell = _render_cell(
            source_path,
            mode=mode,
            center=center,
            crop_fraction=crop_fraction,
            cell_size=cell_size,
            footer=f"source\n{source.megapixels:.2f} MP",
        )
        sheet.paste(source_cell, (row_label_w, y))
        for col, scale in enumerate(scales, start=1):
            variant = Variant(
                variant_id=variant_id(source.bucket, pre_mp, scale, noise),
                bucket=source.bucket,
                pre_megapixels=pre_mp,
                scale=scale,
                noise=noise,
            )
            applies, reason = variant_applies(source, variant, max_output_mp)
            path = _variant_output_path(output_root, source, variant) if applies else None
            predicted = _predicted_output_mp(source.megapixels, pre_mp, scale)
            footer = f"pre={_pre_label(pre_mp)}  scale={scale:g}x\npred≈{predicted:.2f} MP"
            cell = _render_cell(
                path if applies else None,
                mode=mode,
                center=center,
                crop_fraction=crop_fraction,
                cell_size=cell_size,
                footer=footer,
                missing_text=("SKIP\n" + (reason or "")) if not applies else "MISSING",
            )
            sheet.paste(cell, (row_label_w + col * cell_size, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", compress_level=4)


def _write_csv(output_root: Path, rows: Sequence[dict]) -> None:
    path = output_root / "results.csv"
    fieldnames = [
        "source_id",
        "source",
        "bucket",
        "source_mp",
        "pre_mp",
        "scale",
        "noise",
        "predicted_output_mp",
        "status",
        "reason",
        "output",
        "actual_output_mp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(
    output_root: Path,
    *,
    sources: Sequence[SourceImage],
    pre_by_bucket: dict[str, Sequence[float | None]],
    noise_values: Sequence[float],
    crop_names: Sequence[str],
    failures: Sequence[dict],
    plan_only: bool,
) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>SeedVR2 sweep report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px}"
        "h2{margin-top:38px}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}.warn{background:#fff4df;padding:10px;border-radius:8px}"
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        "<h1>SeedVR2 sweep report</h1>",
        f"<p class='meta'>seedvr2-tile {html.escape(__version__)} · {'plan only' if plan_only else 'completed sweep'}</p>",
    ]
    if failures:
        parts.append(f"<p class='warn'>{len(failures)} variant run(s) reported errors. See <code>manifest.json</code>.</p>")
    for bucket in BUCKETS:
        bucket_sources = [source for source in sources if source.selected and source.bucket == bucket]
        if not bucket_sources:
            continue
        parts.append(f"<h2>{bucket.title()} bucket</h2>")
        parts.append(
            "<p class='meta'>pre-resize rows: "
            + ", ".join(html.escape(_pre_label(value)) for value in pre_by_bucket[bucket])
            + "</p>"
        )
        for source in bucket_sources:
            parts.append("<div class='card'>")
            parts.append(
                f"<h3>{html.escape(source.relative.as_posix())}</h3>"
                f"<p class='meta'>{source.width}×{source.height} · {source.megapixels:.2f} MP</p>"
            )
            if plan_only:
                parts.append("<p>Selected for sweep. Run without <code>--plan-only</code> to generate comparisons.</p>")
            else:
                for noise in noise_values:
                    noise_dir = f"noise-{_fmt_float(noise)}"
                    base = Path("reports") / bucket / source.source_id / noise_dir
                    links = [("full", base / "full.png")]
                    links.extend((name, base / f"crop-{name}.png") for name in crop_names)
                    for label, rel in links:
                        parts.append(f"<h4>noise={noise:g} · {html.escape(label)}</h4>")
                        parts.append(f"<a href='{html.escape(rel.as_posix())}'><img src='{html.escape(rel.as_posix())}' alt='comparison'></a>")
            parts.append("</div>")
    parts.append("</body></html>")
    (output_root / "index.html").write_text("".join(parts), encoding="utf-8")


def _serialize_source(source: SourceImage) -> dict:
    return {
        "source_id": source.source_id,
        "source": source.relative.as_posix(),
        "width": source.width,
        "height": source.height,
        "megapixels": source.megapixels,
        "bucket": source.bucket,
        "selected": source.selected,
    }


def _run_variant(
    *,
    variant: Variant,
    sources: Sequence[SourceImage],
    staged: dict[str, Path],
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[int, str | None]:
    applicable = []
    for source in sources:
        applies, _ = variant_applies(source, variant, args.max_output_mp)
        if applies:
            applicable.append(source)
    if not applicable:
        return 0, None

    variant_dir = output_root / "results" / variant.bucket / variant.variant_id
    cli_args: list[str] = ["run"]
    cli_args.extend(str(staged[source.source_id]) for source in applicable)
    cli_args.append(str(variant_dir))
    cli_args.extend(
        [
            "--model",
            "3b",
            "--scale",
            str(variant.scale),
            "--noise",
            str(variant.noise),
            "--noise-seed",
            str(args.seed),
            "--seed",
            str(args.seed),
            "--tile",
            str(args.tile),
            "--overlap",
            str(args.overlap),
            "--tile-upscale-resolution",
            str(args.tile_upscale_resolution),
            "--strategy",
            args.strategy,
            "--blend",
            args.blend,
            "--attention-mode",
            args.attention_mode,
            "--color-correction",
            args.color_correction,
            "--output-mode",
            "on",
            "--output-template",
            "#basename#",
            "--format",
            "png",
            "--overwrite",
        ]
    )
    if variant.pre_megapixels is not None:
        cli_args.extend(["--pre-megapixels", str(variant.pre_megapixels), "--pre-resample", args.pre_resample])
    if args.fbcnn:
        cli_args.extend(["--fbcnn", "--jpeg-quality", args.jpeg_quality, "--fbcnn-device", args.fbcnn_device])
    if args.no_model_download:
        cli_args.append("--no-model-download")

    try:
        rc = seedvr2_main(cli_args)
        if rc:
            return int(rc), f"seedvr2-tile returned {rc}"
        return 0, None
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _build_result_rows(
    sources: Sequence[SourceImage],
    variants: Sequence[Variant],
    output_root: Path,
    max_output_mp: float,
) -> list[dict]:
    rows: list[dict] = []
    for variant in variants:
        bucket_sources = [source for source in sources if source.selected and source.bucket == variant.bucket]
        for source in bucket_sources:
            applies, reason = variant_applies(source, variant, max_output_mp)
            output = _variant_output_path(output_root, source, variant)
            actual_mp: float | None = None
            status = "skipped"
            if applies:
                if output.is_file():
                    with Image.open(output) as opened:
                        actual_mp = megapixels_for_size(*opened.size)
                    status = "complete"
                else:
                    status = "missing"
            rows.append(
                {
                    "source_id": source.source_id,
                    "source": source.relative.as_posix(),
                    "bucket": source.bucket,
                    "source_mp": f"{source.megapixels:.6f}",
                    "pre_mp": "native" if variant.pre_megapixels is None else f"{variant.pre_megapixels:.6f}",
                    "scale": f"{variant.scale:.6f}",
                    "noise": f"{variant.noise:.6f}",
                    "predicted_output_mp": f"{_predicted_output_mp(source.megapixels, variant.pre_megapixels, variant.scale):.6f}",
                    "status": status,
                    "reason": reason or "",
                    "output": output.relative_to(output_root).as_posix() if output.is_file() else "",
                    "actual_output_mp": "" if actual_mp is None else f"{actual_mp:.6f}",
                }
            )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-sweep",
        description="Bucket, sample, sweep and visually compare SeedVR2 restoration settings.",
    )
    parser.add_argument("input", help="input image, directory or glob")
    parser.add_argument("output", type=Path, help="experiment output directory")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--all-images", action="store_true", help="sweep every image instead of a spread sample")
    parser.add_argument("--only-bucket", action="append", choices=BUCKETS, dest="only_buckets")
    parser.add_argument("--small-max", type=float, default=1.25, help="small bucket upper bound in MP (1024² pixels)")
    parser.add_argument("--medium-max", type=float, default=4.0, help="medium bucket upper bound in MP (1024² pixels)")
    parser.add_argument("--pre-small", type=_parse_pre_values, default=DEFAULT_PRESETS["small"])
    parser.add_argument("--pre-medium", type=_parse_pre_values, default=DEFAULT_PRESETS["medium"])
    parser.add_argument("--pre-large", type=_parse_pre_values, default=DEFAULT_PRESETS["large"])
    parser.add_argument("--scales", type=lambda v: _parse_csv_floats(v, allow_zero=False), default=DEFAULT_SCALES)
    parser.add_argument("--noise-values", type=_parse_csv_floats, default=DEFAULT_NOISE)
    parser.add_argument("--max-output-mp", type=float, default=20.0, help="skip combinations predicted to exceed this output MP; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fbcnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jpeg-quality", default="auto")
    parser.add_argument("--fbcnn-device", default="auto")
    parser.add_argument("--pre-resample", choices=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos")
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--tile-upscale-resolution", type=int, default=2048)
    parser.add_argument("--strategy", choices=["chess", "linear"], default="chess")
    parser.add_argument("--blend", choices=["multiband", "linear", "simple", "content-aware", "bilateral"], default="multiband")
    parser.add_argument("--attention-mode", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], default="sdpa")
    parser.add_argument("--color-correction", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], default="lab")
    parser.add_argument("--crop-fraction", type=float, default=0.30)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any variant run fails")
    parser.add_argument("--no-model-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.samples_per_bucket <= 0:
        raise SystemExit("--samples-per-bucket must be > 0")
    if args.small_max <= 0 or args.medium_max <= args.small_max:
        raise SystemExit("bucket thresholds must satisfy 0 < small-max < medium-max")
    if args.crop_fraction <= 0 or args.crop_fraction > 1:
        raise SystemExit("--crop-fraction must be in (0, 1]")
    if args.cell_size < 96:
        raise SystemExit("--cell-size must be >= 96")
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

    pre_by_bucket = {
        "small": args.pre_small,
        "medium": args.pre_medium,
        "large": args.pre_large,
    }
    variants: list[Variant] = []
    for bucket in BUCKETS:
        if bucket in only_buckets:
            variants.extend(build_variants(bucket, pre_by_bucket[bucket], args.scales, args.noise_values))

    selected = [source for source in sources if source.selected]
    print(f"Discovered {len(sources)} image(s); selected {len(selected)} for sweep.")
    for bucket in BUCKETS:
        all_count = sum(source.bucket == bucket for source in sources)
        selected_count = sum(source.bucket == bucket and source.selected for source in sources)
        print(f"  {bucket}: {all_count} total, {selected_count} selected")
    print(f"Planned variants: {len(variants)} bucket/parameter combinations; model=3b FP8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seedvr2_tile_version": __version__,
        "model": "3b",
        "input": args.input,
        "settings": {
            "small_max": args.small_max,
            "medium_max": args.medium_max,
            "samples_per_bucket": args.samples_per_bucket,
            "all_images": args.all_images,
            "only_buckets": sorted(only_buckets),
            "pre_by_bucket": {
                bucket: ["native" if value is None else value for value in values]
                for bucket, values in pre_by_bucket.items()
            },
            "scales": list(args.scales),
            "noise_values": list(args.noise_values),
            "max_output_mp": args.max_output_mp,
            "seed": args.seed,
            "fbcnn": args.fbcnn,
            "jpeg_quality": args.jpeg_quality,
            "pre_resample": args.pre_resample,
            "tile": args.tile,
            "overlap": args.overlap,
            "tile_upscale_resolution": args.tile_upscale_resolution,
            "strategy": args.strategy,
            "blend": args.blend,
            "attention_mode": args.attention_mode,
            "color_correction": args.color_correction,
            "crop_fraction": args.crop_fraction,
            "cell_size": args.cell_size,
        },
        "sources": [_serialize_source(source) for source in sources],
        "variants": [asdict(variant) for variant in variants],
        "failures": [],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.plan_only:
        _write_html(
            output_root,
            sources=sources,
            pre_by_bucket=pre_by_bucket,
            noise_values=args.noise_values,
            crop_names=DEFAULT_CROPS.keys(),
            failures=[],
            plan_only=True,
        )
        print(f"Plan written to {output_root / 'index.html'}")
        return 0

    failures: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="seedvr2-sweep-") as temp_dir:
        staged = _stage_selected(selected, Path(temp_dir) / "selected")
        for index, variant in enumerate(variants, start=1):
            bucket_sources = [
                source for source in selected
                if source.bucket == variant.bucket and variant_applies(source, variant, args.max_output_mp)[0]
            ]
            if not bucket_sources:
                print(f"[{index}/{len(variants)}] skip {variant.variant_id}: no applicable images")
                continue
            print(f"[{index}/{len(variants)}] {variant.variant_id}: {len(bucket_sources)} image(s)")
            rc, error = _run_variant(
                variant=variant,
                sources=bucket_sources,
                staged=staged,
                output_root=output_root,
                args=args,
            )
            if rc:
                failure = {"variant_id": variant.variant_id, "exit_code": rc, "error": error or "unknown error"}
                failures.append(failure)
                print(f"WARNING: {variant.variant_id}: {failure['error']}")

    result_rows = _build_result_rows(sources, variants, output_root, args.max_output_mp)
    _write_csv(output_root, result_rows)

    crop_names = list(DEFAULT_CROPS)
    for source in selected:
        for noise in args.noise_values:
            report_dir = output_root / "reports" / source.bucket / source.source_id / f"noise-{_fmt_float(noise)}"
            _make_contact_sheet(
                source=source,
                source_path=source.source_path,
                output_root=output_root,
                pre_values=pre_by_bucket[source.bucket],
                scales=args.scales,
                noise=noise,
                mode="full",
                center=None,
                crop_fraction=args.crop_fraction,
                cell_size=args.cell_size,
                max_output_mp=args.max_output_mp,
                target=report_dir / "full.png",
            )
            for crop_name, center in DEFAULT_CROPS.items():
                _make_contact_sheet(
                    source=source,
                    source_path=source.source_path,
                    output_root=output_root,
                    pre_values=pre_by_bucket[source.bucket],
                    scales=args.scales,
                    noise=noise,
                    mode=crop_name,
                    center=center,
                    crop_fraction=args.crop_fraction,
                    cell_size=args.cell_size,
                    max_output_mp=args.max_output_mp,
                    target=report_dir / f"crop-{crop_name}.png",
                )

    manifest["failures"] = failures
    manifest["result_summary"] = {
        "complete": sum(row["status"] == "complete" for row in result_rows),
        "missing": sum(row["status"] == "missing" for row in result_rows),
        "skipped": sum(row["status"] == "skipped" for row in result_rows),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_html(
        output_root,
        sources=sources,
        pre_by_bucket=pre_by_bucket,
        noise_values=args.noise_values,
        crop_names=crop_names,
        failures=failures,
        plan_only=False,
    )
    print(f"Sweep report: {output_root / 'index.html'}")
    if failures and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
