from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SECTION_MAP: dict[str, dict[str, str]] = {
    "io": {
        "input": "input", "output": "output", "format": "format", "quality": "quality",
        "recursive": "recursive", "overwrite": "overwrite", "keep_work": "keep_work",
        "work_dir": "work_dir", "dry_run": "dry_run",
    },
    "preprocess": {
        "megapixels": "pre_megapixels", "resample": "pre_resample",
        "noise": "noise", "noise_seed": "noise_seed",
    },
    "upscale": {
        "scale": "scale", "long_edge": "long_edge", "short_edge": "short_edge",
    },
    "tiling": {
        "tile": "tile", "tile_width": "tile_width", "tile_height": "tile_height",
        "overlap": "overlap", "tile_upscale_resolution": "tile_upscale_resolution",
        "strategy": "strategy", "blend": "blend",
    },
    "backend": {
        "seedvr2_root": "seedvr2_root", "seed": "seed", "dit_model": "dit_model",
        "model_dir": "model_dir", "cuda_device": "cuda_device", "attention_mode": "attention_mode",
        "color_correction": "color_correction", "blocks_to_swap": "blocks_to_swap",
        "swap_io_components": "swap_io_components", "dit_offload_device": "dit_offload_device",
        "vae_offload_device": "vae_offload_device", "tensor_offload_device": "tensor_offload_device",
        "vae_tiled": "vae_tiled", "vae_tile_size": "vae_tile_size",
        "vae_tile_overlap": "vae_tile_overlap", "debug": "debug",
    },
}


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")

    flat: dict[str, Any] = {}
    known_flat = {dest for section in SECTION_MAP.values() for dest in section.values()}
    known_flat |= {"pre_megapixels", "pre_resample"}

    for key, value in raw.items():
        normalized = key.replace("-", "_")
        if normalized in SECTION_MAP:
            if not isinstance(value, dict):
                raise ValueError(f"config section '{key}' must be an object")
            mapping = SECTION_MAP[normalized]
            for child, child_value in value.items():
                child_norm = child.replace("-", "_")
                if child_norm not in mapping:
                    raise ValueError(f"unknown config key '{key}.{child}'")
                flat[mapping[child_norm]] = child_value
        elif normalized in known_flat:
            flat[normalized] = value
        else:
            raise ValueError(f"unknown config key '{key}'")

    for path_key in ("input", "output", "work_dir"):
        if path_key in flat and flat[path_key] is not None:
            flat[path_key] = Path(flat[path_key])
    return flat
