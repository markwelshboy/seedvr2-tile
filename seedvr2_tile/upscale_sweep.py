from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import tempfile
from typing import Sequence

from PIL import Image, ImageOps

from . import __version__
from .preprocess import PreprocessOptions, preprocess_image
from .spandrel_backend import MODEL_SPECS, SpandrelUpscaler
from .sweep import (
    BUCKETS,
    DEFAULT_CROPS,
    DEFAULT_PRESETS,
    SourceImage,
    Variant,
    _build_result_rows,
    _discover_sources,
    _fmt_float,
    _make_contact_sheet,
    _mark_selected,
    _parse_csv_floats,
    _parse_pre_values,
    _pre_label,
    _serialize_source,
    _stage_selected,
    _variant_output_path,
    _write_csv,
    build_variants,
    variant_applies,
)


def _write_html(
    output_root: Path,
    *,
    model_label: str,
    architecture: str,
    native_scale: int,
    sources: Sequence[SourceImage],
    pre_by_bucket: dict[str, Sequence[float | None]],
    noise_values: Sequence[float],
    crop_names: Sequence[str],
    failures: Sequence[dict],
    plan_only: bool,
) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(model_label)} upscale report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 18px}"
        "h2{margin-top:38px}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}.warn{background:#fff4df;padding:10px;border-radius:8px}"
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        f"<h1>{html.escape(model_label)} upscale report</h1>",
        f"<p class='meta'>Spandrel backend · architecture={html.escape(architecture)} · "
        f"native scale={native_scale}× · {'plan only' if plan_only else 'completed sweep'}</p>",
    ]
    if failures:
        parts.append(
            f"<p class='warn'>{len(failures)} variant/image run(s) reported errors. "
            "See <code>manifest.json</code>.</p>"
        )

    for bucket in BUCKETS:
        bucket_sources = [
            source for source in sources if source.selected and source.bucket == bucket
        ]
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
                f"<p class='meta'>{source.width}×{source.height} · "
                f"{source.megapixels:.2f} MP</p>"
            )
            if plan_only:
                parts.append("<p>Selected for sweep.</p>")
            else:
                for noise in noise_values:
                    base = (
                        Path("reports")
                        / bucket
                        / source.source_id
                        / f"noise-{_fmt_float(noise)}"
                    )
                    links = [("full", base / "full.png")]
                    links.extend(
                        (name, base / f"crop-{name}.png") for name in crop_names
                    )
                    for label, rel in links:
                        parts.append(
                            f"<h4>pixel noise={noise:g} · {html.escape(label)}</h4>"
                        )
                        safe = html.escape(rel.as_posix())
                        parts.append(
                            f"<a href='{safe}'><img src='{safe}' alt='comparison'></a>"
                        )
            parts.append("</div>")
    parts.append("</body></html>")
    (output_root / "index.html").write_text("".join(parts), encoding="utf-8")


def _split_rgb_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return image.convert("RGB"), None


def _run_image(
    *,
    backend: SpandrelUpscaler,
    source: SourceImage,
    staged_path: Path,
    variant: Variant,
    output_root: Path,
    args: argparse.Namespace,
    image_index: int,
) -> None:
    with Image.open(staged_path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        rgb, alpha = _split_rgb_alpha(oriented)

    options = PreprocessOptions(
        megapixels=variant.pre_megapixels,
        resample=args.pre_resample,
        noise=variant.noise,
        noise_seed=args.seed,
    )
    rgb, alpha, _messages = preprocess_image(
        rgb,
        alpha,
        options,
        image_index=image_index,
        base_seed=args.seed,
    )
    result = backend.upscale(rgb)

    if alpha is not None:
        scaled_alpha = alpha.resize(result.size, Image.Resampling.LANCZOS)
        result = result.convert("RGBA")
        result.putalpha(scaled_alpha)

    target = _variant_output_path(output_root, source, variant)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target, format="PNG", compress_level=4)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upscale",
        description=(
            "Bucket-aware full-image super-resolution using Spandrel-compatible "
            "PyTorch checkpoints."
        ),
    )
    parser.add_argument("input", help="input image, directory or glob")
    parser.add_argument("output", type=Path, help="experiment output directory")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "model alias/label; built-ins: "
            + ", ".join(sorted(MODEL_SPECS))
            + ". Arbitrary labels work with --model-file/--model-url."
        ),
    )
    parser.add_argument("--model-file", help="checkpoint path already present on the worker")
    parser.add_argument("--model-url", help="direct checkpoint URL; cached between jobs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use FP16 when the model reports support (default: on for CUDA)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=512,
        help="non-overlapping output core size in input pixels; 0 processes whole image",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="context pixels around each inference tile",
    )

    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="process every image instead of a spread sample",
    )
    parser.add_argument(
        "--only-bucket", action="append", choices=BUCKETS, dest="only_buckets"
    )
    parser.add_argument("--small-max", type=float, default=1.25)
    parser.add_argument("--medium-max", type=float, default=4.0)
    parser.add_argument("--pre-small", type=_parse_pre_values, default=DEFAULT_PRESETS["small"])
    parser.add_argument(
        "--pre-medium", type=_parse_pre_values, default=DEFAULT_PRESETS["medium"]
    )
    parser.add_argument("--pre-large", type=_parse_pre_values, default=DEFAULT_PRESETS["large"])
    parser.add_argument(
        "--scales",
        type=lambda value: _parse_csv_floats(value, allow_zero=False),
        default=(2.0,),
        help="requested model scale(s); must match the checkpoint native scale",
    )
    parser.add_argument(
        "--pixel-noise-values",
        type=_parse_csv_floats,
        default=None,
        help="Gaussian RGB preprocessing noise values",
    )
    parser.add_argument(
        "--noise-values",
        type=_parse_csv_floats,
        default=None,
        help="legacy alias for --pixel-noise-values",
    )
    parser.add_argument("--max-output-mp", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pre-resample",
        choices=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
        default="lanczos",
    )
    parser.add_argument("--crop-fraction", type=float, default=0.30)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.pixel_noise_values is not None and args.noise_values is not None:
        raise SystemExit(
            "use either --pixel-noise-values or legacy --noise-values, not both"
        )
    noise_values = tuple(
        args.pixel_noise_values
        if args.pixel_noise_values is not None
        else (args.noise_values if args.noise_values is not None else (0.0,))
    )

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
    selected = [source for source in sources if source.selected]
    source_index = {source.source_id: index for index, source in enumerate(selected)}

    print(f"Loading model {args.model!r}...")
    backend = SpandrelUpscaler(
        args.model,
        model_file=args.model_file,
        model_url=args.model_url,
        device=args.device,
        fp16=args.fp16,
        tile=args.tile,
        overlap=args.overlap,
    )
    info = backend.info
    print(
        f"Model: {info.name} · architecture={info.architecture} · "
        f"scale={info.scale}x · {info.dtype} · {info.device}"
    )

    requested_scales = tuple(float(scale) for scale in args.scales)
    bad_scales = [scale for scale in requested_scales if abs(scale - info.scale) > 1e-6]
    if bad_scales:
        raise SystemExit(
            f"{args.model} is a native {info.scale}x model; requested --scales "
            + ",".join(f"{value:g}" for value in bad_scales)
            + ". This backend intentionally avoids post-resize so comparisons remain honest."
        )

    pre_by_bucket = {
        "small": args.pre_small,
        "medium": args.pre_medium,
        "large": args.pre_large,
    }
    variants: list[Variant] = []
    for bucket in BUCKETS:
        if bucket in only_buckets:
            variants.extend(
                build_variants(
                    bucket,
                    pre_by_bucket[bucket],
                    requested_scales,
                    noise_values,
                )
            )

    print(f"Discovered {len(sources)} image(s); selected {len(selected)}.")
    for bucket in BUCKETS:
        all_count = sum(source.bucket == bucket for source in sources)
        selected_count = sum(
            source.bucket == bucket and source.selected for source in sources
        )
        print(f"  {bucket}: {all_count} total, {selected_count} selected")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "generic-upscale-v1",
        "seedvr2_tile_version": __version__,
        "backend": "spandrel",
        "model": args.model,
        "model_info": {
            "architecture": info.architecture,
            "native_scale": info.scale,
            "checkpoint": info.path.name,
            "device": info.device,
            "dtype": info.dtype,
        },
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
            "scales": list(requested_scales),
            "noise_values": list(noise_values),
            "pixel_noise_values": list(noise_values),
            "max_output_mp": args.max_output_mp,
            "seed": args.seed,
            "pre_resample": args.pre_resample,
            "tile": args.tile,
            "overlap": args.overlap,
            "crop_fraction": args.crop_fraction,
            "cell_size": args.cell_size,
        },
        "sources": [_serialize_source(source) for source in sources],
        "variants": [asdict(variant) for variant in variants],
        "failures": [],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if args.plan_only:
        _write_html(
            output_root,
            model_label=args.model,
            architecture=info.architecture,
            native_scale=info.scale,
            sources=sources,
            pre_by_bucket=pre_by_bucket,
            noise_values=noise_values,
            crop_names=list(DEFAULT_CROPS),
            failures=[],
            plan_only=True,
        )
        print(f"Plan written to {output_root / 'index.html'}")
        return 0

    failures: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="upscale-sweep-") as temp_dir:
        staged = _stage_selected(selected, Path(temp_dir) / "selected")
        for variant_index, variant in enumerate(variants, start=1):
            bucket_sources = [
                source
                for source in selected
                if source.bucket == variant.bucket
                and variant_applies(source, variant, args.max_output_mp)[0]
            ]
            if not bucket_sources:
                print(
                    f"[{variant_index}/{len(variants)}] skip {variant.variant_id}: "
                    "no applicable images"
                )
                continue
            print(
                f"[{variant_index}/{len(variants)}] {variant.variant_id}: "
                f"{len(bucket_sources)} image(s)"
            )
            for source in bucket_sources:
                try:
                    _run_image(
                        backend=backend,
                        source=source,
                        staged_path=staged[source.source_id],
                        variant=variant,
                        output_root=output_root,
                        args=args,
                        image_index=source_index[source.source_id],
                    )
                except Exception as exc:
                    failure = {
                        "variant_id": variant.variant_id,
                        "source_id": source.source_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(
                        f"WARNING: {variant.variant_id}/{source.source_id}: "
                        f"{failure['error']}"
                    )
                    if args.strict:
                        manifest["failures"] = failures
                        (output_root / "manifest.json").write_text(
                            json.dumps(manifest, indent=2), encoding="utf-8"
                        )
                        return 1

    result_rows = _build_result_rows(
        sources, variants, output_root, args.max_output_mp
    )
    _write_csv(output_root, result_rows)

    crop_names = list(DEFAULT_CROPS)
    for source in selected:
        for noise in noise_values:
            report_dir = (
                output_root
                / "reports"
                / source.bucket
                / source.source_id
                / f"noise-{_fmt_float(noise)}"
            )
            _make_contact_sheet(
                source=source,
                source_path=source.source_path,
                output_root=output_root,
                pre_values=pre_by_bucket[source.bucket],
                scales=requested_scales,
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
                    scales=requested_scales,
                    noise=noise,
                    mode=crop_name,
                    center=center,
                    crop_fraction=args.crop_fraction,
                    cell_size=args.cell_size,
                    max_output_mp=args.max_output_mp,
                    target=report_dir / f"crop-{crop_name}.png",
                )

    manifest["failures"] = failures
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_html(
        output_root,
        model_label=args.model,
        architecture=info.architecture,
        native_scale=info.scale,
        sources=sources,
        pre_by_bucket=pre_by_bucket,
        noise_values=noise_values,
        crop_names=crop_names,
        failures=failures,
        plan_only=False,
    )
    print(f"Report: {output_root / 'index.html'}")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
