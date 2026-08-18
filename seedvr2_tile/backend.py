from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "seedvr2-tile" / "ComfyUI-SeedVR2_VideoUpscaler"
UPSTREAM_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"


def resolve_seedvr2_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("SEEDVR2_ROOT"):
        candidates.append(Path(os.environ["SEEDVR2_ROOT"]).expanduser())
    candidates.append(DEFAULT_CACHE_ROOT)

    for root in candidates:
        if (root / "inference_cli.py").is_file():
            return root.resolve()
    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not find SeedVR2 standalone inference_cli.py. "
        f"Searched: {searched}. Run 'seedvr2-tile setup' or set SEEDVR2_ROOT."
    )


def setup_upstream(root: Path, ref: str, install_deps: bool) -> None:
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("git"):
        raise RuntimeError("git is required for 'seedvr2-tile setup'")

    if (root / ".git").exists():
        subprocess.run(["git", "-C", str(root), "fetch", "--tags", "origin"], check=True)
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError(f"setup target exists and is not a git checkout: {root}")
    else:
        subprocess.run(["git", "clone", UPSTREAM_URL, str(root)], check=True)

    subprocess.run(["git", "-C", str(root), "checkout", ref], check=True)
    if ref in {"main", "master"}:
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "origin", ref], check=True)

    if install_deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root)], check=True)


@dataclass
class BackendOptions:
    seedvr2_root: Path
    seed: int = 42
    dit_model: str | None = None
    model_dir: str | None = None
    cuda_device: str | None = None
    attention_mode: str = "sdpa"
    color_correction: str = "lab"
    blocks_to_swap: int = 0
    swap_io_components: bool = False
    dit_offload_device: str = "none"
    vae_offload_device: str = "none"
    tensor_offload_device: str = "cpu"
    vae_tiled: bool = False
    vae_tile_size: int = 1024
    vae_tile_overlap: int = 128
    debug: bool = False


def build_command(
    options: BackendOptions,
    *,
    input_dir: Path,
    output_dir: Path,
    resolution: int,
) -> list[str]:
    cli = options.seedvr2_root / "inference_cli.py"
    cmd = [
        sys.executable,
        str(cli),
        str(input_dir),
        "--output",
        str(output_dir),
        "--output_format",
        "png",
        "--resolution",
        str(resolution),
        "--batch_size",
        "1",
        "--seed",
        str(options.seed),
        "--color_correction",
        options.color_correction,
        "--attention_mode",
        options.attention_mode,
        "--dit_offload_device",
        options.dit_offload_device,
        "--vae_offload_device",
        options.vae_offload_device,
        "--tensor_offload_device",
        options.tensor_offload_device,
        "--cache_dit",
        "--cache_vae",
    ]
    if options.dit_model:
        cmd += ["--dit_model", options.dit_model]
    if options.model_dir:
        cmd += ["--model_dir", options.model_dir]
    if options.cuda_device:
        cmd += ["--cuda_device", options.cuda_device]
    if options.blocks_to_swap:
        cmd += ["--blocks_to_swap", str(options.blocks_to_swap)]
    if options.swap_io_components:
        cmd.append("--swap_io_components")
    if options.vae_tiled:
        cmd += [
            "--vae_encode_tiled",
            "--vae_decode_tiled",
            "--vae_encode_tile_size",
            str(options.vae_tile_size),
            "--vae_decode_tile_size",
            str(options.vae_tile_size),
            "--vae_encode_tile_overlap",
            str(options.vae_tile_overlap),
            "--vae_decode_tile_overlap",
            str(options.vae_tile_overlap),
        ]
    if options.debug:
        cmd.append("--debug")
    return cmd


def run_group(options: BackendOptions, *, input_dir: Path, output_dir: Path, resolution: int, dry_run: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(options, input_dir=input_dir, output_dir=output_dir, resolution=resolution)
    print("SeedVR2:", shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=options.seedvr2_root, check=True)
