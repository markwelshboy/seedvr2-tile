from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import __version__
from .backend import BackendOptions, DEFAULT_CACHE_ROOT, resolve_seedvr2_root, run_group, setup_upstream
from .stitching import Stitcher
from .tiling import TileSpec, make_tiles, save_tiles

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageJob:
    index: int
    source: Path
    relative: Path
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


def _find_images(path: Path, recursive: bool) -> tuple[Path, list[Path]]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    pattern = "**/*" if recursive else "*"
    images = sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"no supported images found in {path}")
    return path, images


def _compute_scale(width: int, height: int, args: argparse.Namespace) -> float:
    if args.long_edge:
        return args.long_edge / max(width, height)
    if args.short_edge:
        return args.short_edge / min(width, height)
    return args.scale


def _output_path(root: Path, relative: Path, fmt: str) -> Path:
    ext = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}[fmt]
    return root / relative.with_suffix(ext)


def _save_output(image: Image.Image, path: Path, fmt: str, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        image.save(path, format="PNG", compress_level=4)
    elif fmt == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=quality, subsampling=0)
    else:
        image.save(path, format="WEBP", quality=quality, method=6)


def _build_parser() -> argparse.ArgumentParser:
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

    run = sub.add_parser("run", help="tile, upscale, and stitch images")
    run.add_argument("input", type=Path)
    run.add_argument("output", type=Path)
    size = run.add_mutually_exclusive_group()
    size.add_argument("--scale", type=float, default=2.0, help="output scale factor (default: 2.0)")
    size.add_argument("--long-edge", type=int, help="target output longest edge")
    size.add_argument("--short-edge", type=int, help="target output shortest edge")
    run.add_argument("--tile", type=int, default=1024, help="square core tile size (default: 1024)")
    run.add_argument("--tile-width", type=int)
    run.add_argument("--tile-height", type=int)
    run.add_argument("--overlap", type=int, default=64, help="context/overlap per side (default: 64)")
    run.add_argument("--tile-upscale-resolution", type=int, default=2048, help="cap SeedVR2 short-side processing resolution per tile (default: 2048)")
    run.add_argument("--strategy", choices=["chess", "linear"], default="chess")
    run.add_argument("--blend", choices=["multiband", "linear", "simple", "content-aware", "bilateral"], default="multiband")
    run.add_argument("--format", choices=["png", "jpg", "webp"], default="png")
    run.add_argument("--quality", type=int, default=95)
    run.add_argument("--recursive", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--keep-work", action="store_true")
    run.add_argument("--work-dir", type=Path)
    run.add_argument("--dry-run", action="store_true", help="prepare tiles and print backend commands without running SeedVR2")

    backend = run.add_argument_group("SeedVR2 backend")
    backend.add_argument("--seedvr2-root")
    backend.add_argument("--seed", type=int, default=42)
    backend.add_argument("--dit-model")
    backend.add_argument("--model-dir")
    backend.add_argument("--cuda-device")
    backend.add_argument("--attention-mode", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], default="sdpa")
    backend.add_argument("--color-correction", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], default="lab")
    backend.add_argument("--blocks-to-swap", type=int, default=0)
    backend.add_argument("--swap-io-components", action="store_true")
    backend.add_argument("--dit-offload-device", default="none")
    backend.add_argument("--vae-offload-device", default="none")
    backend.add_argument("--tensor-offload-device", default="cpu")
    backend.add_argument("--vae-tiled", action="store_true")
    backend.add_argument("--vae-tile-size", type=int, default=1024)
    backend.add_argument("--vae-tile-overlap", type=int, default=128)
    backend.add_argument("--debug", action="store_true")

    return parser


def _run(args: argparse.Namespace) -> int:
    if args.scale is not None and args.scale <= 0:
        raise ValueError("--scale must be > 0")
    tile_w = args.tile_width or args.tile
    tile_h = args.tile_height or args.tile
    if args.overlap >= min(tile_w, tile_h):
        raise ValueError("--overlap must be smaller than the tile dimensions")

    source_root, sources = _find_images(args.input.expanduser().resolve(), args.recursive)
    output_root = args.output.expanduser().resolve()
    seedvr2_root = resolve_seedvr2_root(args.seedvr2_root)

    options = BackendOptions(
        seedvr2_root=seedvr2_root,
        seed=args.seed,
        dit_model=args.dit_model,
        model_dir=args.model_dir,
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

    if args.work_dir:
        work_root = args.work_dir.expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        temporary = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="seedvr2-tile-"))
        temporary = True

    print(f"Input images: {len(sources)}")
    print(f"Work directory: {work_root}")

    jobs: list[ImageJob] = []
    groups: dict[int, list[tuple[TileSpec, Image.Image]]] = {}

    try:
        for image_index, src in enumerate(sources):
            relative = src.relative_to(source_root)
            out_path = _output_path(output_root, relative, args.format)
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
                continue

            with Image.open(src) as opened:
                opened = ImageOps.exif_transpose(opened)
                alpha = opened.getchannel("A").copy() if "A" in opened.getbands() else None
                rgb = opened.convert("RGB")
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
            print(
                f"Plan {relative}: {width}x{height} -> {output_width}x{output_height}, "
                f"scale={scale:.4f}, tiles={len(tile_pairs)}, SeedVR2 tile resolution={group_resolution}"
            )

        if not jobs:
            print("Nothing to do.")
            return 0

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
            out_path = _output_path(output_root, job.relative, args.format)
            _save_output(output, out_path, args.format, args.quality)
            print(f"Saved: {out_path}")

        return 0
    finally:
        if temporary and not args.keep_work:
            shutil.rmtree(work_root, ignore_errors=True)
        elif args.keep_work:
            print(f"Kept work directory: {work_root}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] not in {"run", "setup", "-h", "--help", "--version"}:
        argv.insert(0, "run")
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup_upstream(args.root, args.ref, args.install_deps)
        print(f"SeedVR2 backend ready: {args.root.expanduser().resolve()}")
        return 0
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
