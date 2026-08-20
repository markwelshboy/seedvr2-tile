from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import __version__
from .backend import (
    BackendOptions,
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    ensure_models,
    model_alias_lines,
    resolve_model_name,
    resolve_seedvr2_root,
    run_group,
    setup_upstream,
)
from .config import load_config
from .fbcnn import DEFAULT_FBCNN_ROOT, FBCNNOptions, normalize_quality, release_fbcnn, setup_fbcnn
from .inputs import discover_inputs
from .naming import (
    DEFAULT_OUTPUT_TEMPLATE,
    option_values,
    render_output_stem,
    validate_output_template,
)
from .preprocess import PreprocessOptions, preprocess_image
from .stitching import Stitcher
from .tiling import TileSpec, make_tiles, save_tiles

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageJob:
    index: int
    source: Path
    relative: Path
    output_path: Path
    width: int
    height: int
    scale: float
    output_width: int
    output_height: int
    alpha: Image.Image | None
    tiles: list[TileSpec]
    group_resolution: int


def _round_even(value: float) -> int:
    return max(16, int(round(value / 2.0) * 2))


def _compute_scale(width: int, height: int, args: argparse.Namespace) -> float:
    if args.long_edge:
        return args.long_edge / max(width, height)
    if args.short_edge:
        return args.short_edge / min(width, height)
    return args.scale


def _output_path(root: Path, relative: Path, fmt: str, stem: str) -> Path:
    ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}[fmt]
    return root / relative.parent / f"{stem}{ext}"


def _save_output(image: Image.Image, path: Path, fmt: str, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        image.save(path, format="PNG", compress_level=4)
    elif fmt == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=quality, subsampling=0)
    else:
        image.save(path, format="WEBP", quality=quality, method=6)


def _extract_config_path(argv: list[str]) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def _config_input_specs(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def _resolve_run_io(args: argparse.Namespace) -> tuple[list[str], Path]:
    """Resolve backward-compatible positional input(s)/output plus config IO.

    Without a configured/explicit output directory, the final positional path is
    the output and all preceding paths are inputs. That makes an unquoted shell
    glob work naturally: `seedvr2-tile images/*.jpg output/`.

    If config supplies `io.output` (or --output-dir is used), every positional
    path is treated as an input, which also makes glob expansion unambiguous.
    """
    positional = [str(x) for x in getattr(args, "paths", [])]
    configured_inputs = _config_input_specs(getattr(args, "input", None))
    configured_output = getattr(args, "output", None)

    if args.output_dir is not None:
        output = args.output_dir
        inputs = positional or configured_inputs
    elif configured_output is not None:
        output = Path(configured_output)
        inputs = positional or configured_inputs
    elif len(positional) >= 2:
        inputs = positional[:-1]
        output = Path(positional[-1])
    elif len(positional) == 1 and configured_inputs:
        inputs = configured_inputs
        output = Path(positional[0])
    else:
        raise ValueError(
            "input(s) and output directory are required. Use INPUT... OUTPUT, "
            "or set io.input/io.output in config, or pass --output-dir."
        )

    if not inputs:
        raise ValueError("at least one input image, directory, or glob is required")
    return inputs, Path(output).expanduser().resolve()


def _build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-tile",
        description="Standalone spatially tiled batch frontend for SeedVR2",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    setup = sub.add_parser("setup", help="clone/update the standalone Numz SeedVR2 backend")
    setup.add_argument("--root", type=Path, default=DEFAULT_CACHE_ROOT)
    setup.add_argument("--ref", default="main", help="upstream branch/tag/commit (default: main)")
    setup.add_argument("--install-deps", action="store_true", help="pip install -e the upstream checkout into this Python environment")
    setup.add_argument("--fbcnn", action=argparse.BooleanOptionalAction, default=False, help="also clone/update the official FBCNN JPEG artifact-removal backend")
    setup.add_argument("--fbcnn-root", type=Path, default=DEFAULT_FBCNN_ROOT)
    setup.add_argument("--fbcnn-ref", default="main", help="FBCNN branch/tag/commit (default: main)")

    sub.add_parser("models", help="list friendly SeedVR2 model aliases")

    run = sub.add_parser("run", help="tile, upscale, and stitch images")
    run.add_argument(
        "paths",
        nargs="*",
        help=(
            "input image(s), directory/directories or glob(s), followed by output directory. "
            "With io.output/--output-dir set, every positional path is an input"
        ),
    )
    run.add_argument("--output-dir", type=Path, help="explicit output directory; useful when inputs are shell-expanded or output also exists in config")
    run.add_argument("--config", type=Path, help="optional JSON config file; CLI flags override values from the file")
    run.add_argument("--output-mode", choices=["on", "all", "delta"], default="on", help="filename mode: configured template, all processing options, or only non-default options")
    run.add_argument("--output-template", default=DEFAULT_OUTPUT_TEMPLATE, help="filename stem template for --output-mode on; fields use #option-name# syntax")

    size = run.add_mutually_exclusive_group()
    size.add_argument("--scale", type=float, default=None, help="output scale factor applied after preprocessing (default: 2.0 when no size mode is set)")
    size.add_argument("--long-edge", type=int, help="target output longest edge after preprocessing")
    size.add_argument("--short-edge", type=int, help="target output shortest edge after preprocessing")
    run.add_argument("--tile", type=int, default=1024, help="square core tile size (default: 1024)")
    run.add_argument("--tile-width", type=int)
    run.add_argument("--tile-height", type=int)
    run.add_argument("--overlap", type=int, default=64, help="context/overlap per side (default: 64)")
    run.add_argument("--tile-upscale-resolution", type=int, default=2048, help="cap SeedVR2 short-side processing resolution per tile (default: 2048)")
    run.add_argument("--strategy", choices=["chess", "linear"], default="chess")
    run.add_argument("--blend", choices=["multiband", "linear", "simple", "content-aware", "bilateral"], default="multiband")
    run.add_argument("--format", choices=["png", "jpg", "webp"], default="png")
    run.add_argument("--quality", type=int, default=95)
    run.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--keep-work", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--work-dir", type=Path)
    run.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="prepare tiles and print backend commands without running SeedVR2")

    preprocess = run.add_argument_group("Optional preprocessing (order: FBCNN -> resize -> noise)")
    preprocess.add_argument("--fbcnn", dest="fbcnn_enabled", action=argparse.BooleanOptionalAction, default=False, help="remove JPEG artifacts with official FBCNN before other preprocessing")
    preprocess.add_argument("--jpeg-quality", dest="fbcnn_quality", default="auto", help="FBCNN JPEG QF: auto or explicit quality factor 1..100")
    preprocess.add_argument("--fbcnn-root", type=Path, help="official FBCNN checkout; defaults to FBCNN_ROOT or ~/.cache/seedvr2-tile/FBCNN")
    preprocess.add_argument("--fbcnn-device", default="auto", help="FBCNN device: auto, cpu, cuda, cuda:0, etc. (default: auto)")
    preprocess.add_argument("--pre-megapixels", type=float, dest="pre_megapixels", help="resize source RGB/alpha to this target megapixel count before tiling and upscaling (ComfyUI-style MP = 1024x1024 pixels)")
    preprocess.add_argument("--pre-resample", choices=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos", help="resample filter used by --pre-megapixels (default: lanczos)")
    preprocess.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise strength in [0, 1] applied after optional pre-resize")
    preprocess.add_argument("--noise-seed", type=int, help="base RNG seed for preprocessing noise; defaults to the main --seed plus image index")

    backend = run.add_argument_group("SeedVR2 backend")
    backend.add_argument("--seedvr2-root")
    backend.add_argument("--seed", type=int, default=42)
    model_select = backend.add_mutually_exclusive_group()
    model_select.add_argument("--model", dest="dit_model", default=DEFAULT_MODEL_ALIAS, help="friendly model alias (3b/3b-fp8, 3b-fp16, 7b, 7b-sharp, etc.) or exact filename; default: 3b (FP8)")
    model_select.add_argument("--dit-model", dest="dit_model", help="exact SeedVR2 DiT filename (expert/backward-compatible form)")
    backend.add_argument("--model-dir")
    backend.add_argument("--model-download", action=argparse.BooleanOptionalAction, default=True, help="preflight/download selected DiT + VAE using Numz downloader (default: enabled)")
    backend.add_argument("--cuda-device")
    backend.add_argument("--attention-mode", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], default="sdpa")
    backend.add_argument("--color-correction", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], default="lab")
    backend.add_argument("--blocks-to-swap", type=int, default=0)
    backend.add_argument("--swap-io-components", action=argparse.BooleanOptionalAction, default=False)
    backend.add_argument("--dit-offload-device", default="none")
    backend.add_argument("--vae-offload-device", default="none")
    backend.add_argument("--tensor-offload-device", default="cpu")
    backend.add_argument("--vae-tiled", action=argparse.BooleanOptionalAction, default=False)
    backend.add_argument("--vae-tile-size", type=int, default=1024)
    backend.add_argument("--vae-tile-overlap", type=int, default=128)
    backend.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    if defaults:
        run.set_defaults(**defaults)
    return parser


def _run(args: argparse.Namespace) -> int:
    input_specs, output_root = _resolve_run_io(args)

    if args.scale is None and args.long_edge is None and args.short_edge is None:
        args.scale = 2.0
    if args.scale is not None and args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if args.pre_megapixels is not None and args.pre_megapixels <= 0:
        raise ValueError("--pre-megapixels must be > 0")
    if args.noise < 0 or args.noise > 1:
        raise ValueError("--noise must be in the range [0, 1]")

    args.fbcnn_quality = normalize_quality(args.fbcnn_quality)
    if args.output_mode == "on":
        validate_output_template(args.output_template)

    requested_model = args.dit_model or DEFAULT_MODEL_ALIAS
    requested_key = requested_model.strip().lower()
    model_label = requested_key if requested_key in MODEL_ALIASES else requested_model.strip()
    args.dit_model = resolve_model_name(requested_model)

    tile_w = args.tile_width or args.tile
    tile_h = args.tile_height or args.tile
    if args.overlap >= min(tile_w, tile_h):
        raise ValueError("--overlap must be smaller than the tile dimensions")

    source_items = discover_inputs(input_specs, recursive=args.recursive, extensions=IMAGE_EXTENSIONS)
    seedvr2_root = resolve_seedvr2_root(args.seedvr2_root)

    options = BackendOptions(
        seedvr2_root=seedvr2_root,
        seed=args.seed,
        dit_model=args.dit_model,
        model_dir=args.model_dir,
        download_model=args.model_download,
        cuda_device=args.cuda_device,
        attention_mode=args.attention_mode,
        color_correction=args.color_correction,
        blocks_to_swap=args.blocks_to_swap,
        swap_io_components=args.swap_io_components,
        dit_offload_device=args.dit_offload_device,
        vae_offload_device=args.vae_offload_device,
        tensor_offload_device=args.tensor_offload_device,
        vae_tiled=args.vae_tiled,
        vae_tile_size=args.vae_tile_size,
        vae_tile_overlap=args.vae_tile_overlap,
        debug=args.debug,
    )

    # Resolve/validate/download the selected DiT + VAE before any expensive
    # preprocessing or tile preparation. Numz owns the actual download logic.
    ensure_models(options, dry_run=args.dry_run)

    preprocess_options = PreprocessOptions(
        fbcnn=FBCNNOptions(
            enabled=args.fbcnn_enabled,
            quality=args.fbcnn_quality,
            root=args.fbcnn_root,
            device=args.fbcnn_device,
        ),
        megapixels=args.pre_megapixels,
        resample=args.pre_resample,
        noise=args.noise,
        noise_seed=args.noise_seed,
    )

    naming_values = option_values(args, model_label=model_label)

    if args.work_dir:
        work_root = args.work_dir.expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        temporary = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="seedvr2-tile-"))
        temporary = True

    print(f"Input images: {len(source_items)}")
    print(f"Output mode: {args.output_mode}")
    if args.output_mode == "on":
        print(f"Output template: {args.output_template}")
    print(f"Work directory: {work_root}")

    jobs: list[ImageJob] = []
    groups: dict[int, list[tuple[TileSpec, Image.Image]]] = {}
    planned_outputs: set[Path] = set()

    try:
        for image_index, item in enumerate(source_items):
            src = item.path
            relative = item.relative
            output_stem = render_output_stem(
                basename=relative.stem,
                mode=args.output_mode,
                template=args.output_template,
                values=naming_values,
            )
            out_path = _output_path(output_root, relative, args.format, output_stem)
            if out_path in planned_outputs:
                raise ValueError(
                    f"output naming collision: multiple inputs map to {out_path}. "
                    "Include #basename# or another distinguishing field in --output-template."
                )
            planned_outputs.add(out_path)
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
                continue

            with Image.open(src) as opened:
                opened = ImageOps.exif_transpose(opened)
                alpha = opened.getchannel("A").copy() if "A" in opened.getbands() else None
                rgb = opened.convert("RGB")
                original_width, original_height = rgb.size
                rgb, alpha, preprocess_messages = preprocess_image(
                    rgb,
                    alpha,
                    preprocess_options,
                    image_index=image_index,
                    base_seed=args.seed,
                )
                width, height = rgb.size
                scale = _compute_scale(width, height, args)
                if scale <= 0:
                    raise ValueError(f"computed non-positive scale for {src}")
                output_width = max(1, round(width * scale))
                output_height = max(1, round(height * scale))

                tile_pairs = make_tiles(
                    rgb,
                    image_index=image_index,
                    tile_width=tile_w,
                    tile_height=tile_h,
                    padding=args.overlap,
                    strategy=args.strategy,
                )
                if preprocess_messages:
                    preview = rgb.copy()
                    if alpha is not None:
                        preview.putalpha(alpha)
                    pre_path = work_root / "preprocessed" / relative.with_suffix(".png")
                    pre_path.parent.mkdir(parents=True, exist_ok=True)
                    preview.save(pre_path, format="PNG")

            process_w, process_h = tile_pairs[0][0].process_size
            desired_resolution = _round_even(min(process_w, process_h) * scale)
            group_resolution = min(desired_resolution, args.tile_upscale_resolution)
            group_resolution = _round_even(group_resolution)
            groups.setdefault(group_resolution, []).extend(tile_pairs)
            jobs.append(
                ImageJob(
                    index=image_index,
                    source=src,
                    relative=relative,
                    output_path=out_path,
                    width=width,
                    height=height,
                    scale=scale,
                    output_width=output_width,
                    output_height=output_height,
                    alpha=alpha,
                    tiles=[s for s, _ in tile_pairs],
                    group_resolution=group_resolution,
                )
            )

            preamble = f"src={original_width}x{original_height}"
            if (width, height) != (original_width, original_height):
                preamble += f", proc={width}x{height}"
            if preprocess_messages:
                preamble += ", " + "; ".join(preprocess_messages)
            print(
                f"Plan {relative}: {preamble} -> out={output_width}x{output_height}, "
                f"scale={scale:.4f}, tiles={len(tile_pairs)}, SeedVR2 tile resolution={group_resolution}, "
                f"file={out_path.name}"
            )

        if not jobs:
            print("Nothing to do.")
            return 0

        # FBCNN may have used CUDA during preprocessing. Release it before the
        # much larger SeedVR2 backend is launched so it does not reserve VRAM.
        release_fbcnn()

        # One upstream directory invocation per distinct tile inference resolution.
        # With the common --scale workflow this is normally exactly one invocation,
        # so Numz's --cache_dit/--cache_vae persist across every tile in the batch.
        for resolution, tile_pairs in sorted(groups.items()):
            input_dir = work_root / f"r{resolution}" / "input"
            output_dir = work_root / f"r{resolution}" / "output"
            save_tiles(tile_pairs, input_dir)
            print(f"Group r{resolution}: {len(tile_pairs)} tiles")
            run_group(
                options,
                input_dir=input_dir,
                output_dir=output_dir,
                resolution=resolution,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            print("Dry run complete; stitching skipped because no SeedVR2 outputs were generated.")
            return 0

        for job_idx, job in enumerate(jobs, 1):
            print(f"Stitch {job_idx}/{len(jobs)}: {job.relative}")
            stitcher = Stitcher(job.output_width, job.output_height, method=args.blend)
            processed_dir = work_root / f"r{job.group_resolution}" / "output"
            for spec in job.tiles:
                processed_path = processed_dir / spec.filename
                if not processed_path.is_file():
                    raise FileNotFoundError(f"SeedVR2 did not produce expected tile: {processed_path}")
                with Image.open(processed_path) as processed:
                    stitcher.add(spec, processed, scale=job.scale)
            output = stitcher.finish()
            if job.alpha is not None and args.format != "jpg":
                alpha = job.alpha.resize((job.output_width, job.output_height), Image.Resampling.LANCZOS)
                output.putalpha(alpha)
            _save_output(output, job.output_path, args.format, args.quality)
            print(f"Saved: {job.output_path}")

        return 0
    finally:
        release_fbcnn()
        if temporary and not args.keep_work:
            shutil.rmtree(work_root, ignore_errors=True)
        elif args.keep_work:
            print(f"Kept work directory: {work_root}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Convenience: allow `seedvr2-tile INPUT... OUTPUT ...` in addition to `run`.
    if argv and argv[0] not in {"run", "setup", "models", "-h", "--help", "--version"}:
        argv.insert(0, "run")

    defaults: dict | None = None
    config_path = _extract_config_path(argv)
    if config_path is not None:
        defaults = load_config(config_path)
        if any(arg == flag or arg.startswith(flag + "=") for arg in argv for flag in ("--scale", "--long-edge", "--short-edge")):
            for key in ("scale", "long_edge", "short_edge"):
                defaults.pop(key, None)

    parser = _build_parser(defaults=defaults)
    args = parser.parse_args(argv)
    if args.command == "models":
        print("SeedVR2 model aliases (default: 3b -> FP8):")
        for line in model_alias_lines():
            print("  " + line)
        print("\nExact Numz registry/discovered filenames are also accepted with --model or --dit-model.")
        return 0
    if args.command == "setup":
        setup_upstream(args.root, args.ref, args.install_deps)
        print(f"SeedVR2 backend ready: {args.root.expanduser().resolve()}")
        if args.fbcnn:
            setup_fbcnn(args.fbcnn_root, args.fbcnn_ref)
            print(f"FBCNN backend ready: {args.fbcnn_root.expanduser().resolve()}")
        return 0
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
