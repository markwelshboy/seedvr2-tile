from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

DEFAULT_OUTPUT_TEMPLATE = "#basename#_#model#_#scale#"
MAX_OUTPUT_STEM = 220

# These are the image-affecting CLI options used by --output-mode all/delta.
# Keep the names identical to their CLI switches (minus the leading --).
OUTPUT_FIELDS: tuple[str, ...] = (
    "model",
    "scale",
    "long-edge",
    "short-edge",
    "fbcnn",
    "jpeg-quality",
    "fbcnn-device",
    "pre-megapixels",
    "pre-resample",
    "noise",
    "noise-seed",
    "tile",
    "tile-width",
    "tile-height",
    "overlap",
    "tile-upscale-resolution",
    "strategy",
    "blend",
    "format",
    "quality",
    "seed",
    "cuda-device",
    "attention-mode",
    "color-correction",
    "blocks-to-swap",
    "swap-io-components",
    "dit-offload-device",
    "vae-offload-device",
    "tensor-offload-device",
    "vae-tiled",
    "vae-tile-size",
    "vae-tile-overlap",
)

DEFAULT_OUTPUT_VALUES: dict[str, Any] = {
    "model": "3b",
    "scale": 2.0,
    "long-edge": None,
    "short-edge": None,
    "fbcnn": False,
    "jpeg-quality": "auto",
    "fbcnn-device": "auto",
    "pre-megapixels": None,
    "pre-resample": "lanczos",
    "noise": 0.0,
    "noise-seed": None,
    "tile": 1024,
    "tile-width": None,
    "tile-height": None,
    "overlap": 64,
    "tile-upscale-resolution": 2048,
    "strategy": "chess",
    "blend": "multiband",
    "format": "png",
    "quality": 95,
    "seed": 42,
    "cuda-device": None,
    "attention-mode": "sdpa",
    "color-correction": "lab",
    "blocks-to-swap": 0,
    "swap-io-components": False,
    "dit-offload-device": "none",
    "vae-offload-device": "none",
    "tensor-offload-device": "cpu",
    "vae-tiled": False,
    "vae-tile-size": 1024,
    "vae-tile-overlap": 128,
}

_TOKEN_RE = re.compile(r"#([^#]+)#")
_INVALID_FILENAME_RE = re.compile(r"[^A-Za-z0-9._+-]+")


def option_values(args: Any, *, model_label: str) -> dict[str, Any]:
    """Return output-template values keyed by the public CLI option names."""
    return {
        "model": model_label,
        "scale": getattr(args, "scale", None),
        "long-edge": getattr(args, "long_edge", None),
        "short-edge": getattr(args, "short_edge", None),
        "fbcnn": getattr(args, "fbcnn_enabled", False),
        "jpeg-quality": getattr(args, "fbcnn_quality", "auto"),
        "fbcnn-device": getattr(args, "fbcnn_device", "auto"),
        "pre-megapixels": getattr(args, "pre_megapixels", None),
        "pre-resample": getattr(args, "pre_resample", "lanczos"),
        "noise": getattr(args, "noise", 0.0),
        "noise-seed": getattr(args, "noise_seed", None),
        "tile": getattr(args, "tile", 1024),
        "tile-width": getattr(args, "tile_width", None),
        "tile-height": getattr(args, "tile_height", None),
        "overlap": getattr(args, "overlap", 64),
        "tile-upscale-resolution": getattr(args, "tile_upscale_resolution", 2048),
        "strategy": getattr(args, "strategy", "chess"),
        "blend": getattr(args, "blend", "multiband"),
        "format": getattr(args, "format", "png"),
        "quality": getattr(args, "quality", 95),
        "seed": getattr(args, "seed", 42),
        "cuda-device": getattr(args, "cuda_device", None),
        "attention-mode": getattr(args, "attention_mode", "sdpa"),
        "color-correction": getattr(args, "color_correction", "lab"),
        "blocks-to-swap": getattr(args, "blocks_to_swap", 0),
        "swap-io-components": getattr(args, "swap_io_components", False),
        "dit-offload-device": getattr(args, "dit_offload_device", "none"),
        "vae-offload-device": getattr(args, "vae_offload_device", "none"),
        "tensor-offload-device": getattr(args, "tensor_offload_device", "cpu"),
        "vae-tiled": getattr(args, "vae_tiled", False),
        "vae-tile-size": getattr(args, "vae_tile_size", 1024),
        "vae-tile-overlap": getattr(args, "vae_tile_overlap", 128),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe(value: Any) -> str:
    text = _format_value(value).strip()
    text = text.replace("/", "-").replace("\\", "-").replace(":", "-")
    text = _INVALID_FILENAME_RE.sub("-", text)
    return text.strip("-_") or "none"


def validate_output_template(template: str) -> tuple[str, ...]:
    if not template:
        raise ValueError("--output-template must not be empty")
    tokens = tuple(_TOKEN_RE.findall(template))
    if not tokens:
        raise ValueError("--output-template must contain at least one #field# placeholder")
    allowed = {"basename", *OUTPUT_FIELDS}
    unknown = sorted({token for token in tokens if token not in allowed})
    if unknown:
        raise ValueError(
            "unknown output-template field(s): " + ", ".join(f"#{x}#" for x in unknown)
            + ". Fields must match CLI option names; available fields: "
            + ", ".join(f"#{x}#" for x in ("basename", *OUTPUT_FIELDS))
        )
    return tokens


def render_template(template: str, *, basename: str, values: Mapping[str, Any]) -> str:
    validate_output_template(template)

    def replace(match: re.Match[str]) -> str:
        field = match.group(1)
        if field == "basename":
            return _safe(basename)
        return _safe(values.get(field))

    rendered = _TOKEN_RE.sub(replace, template)
    # A template is a file *stem*, not a path. Flatten accidental separators and
    # sanitize arbitrary literals while preserving common filename punctuation.
    rendered = rendered.replace("/", "-").replace("\\", "-")
    rendered = _INVALID_FILENAME_RE.sub("-", rendered)
    return rendered.strip("-_.") or _safe(basename)


def render_output_stem(
    *,
    basename: str,
    mode: str,
    template: str | None,
    values: Mapping[str, Any],
) -> str:
    if mode == "on":
        return render_template(template or DEFAULT_OUTPUT_TEMPLATE, basename=basename, values=values)
    if mode not in {"all", "delta"}:
        raise ValueError(f"unknown output mode: {mode}")

    fields = OUTPUT_FIELDS
    if mode == "delta":
        fields = tuple(field for field in OUTPUT_FIELDS if values.get(field) != DEFAULT_OUTPUT_VALUES[field])

    base = _safe(basename)
    if mode == "all":
        tags = {
            "model": "m", "scale": "s", "long-edge": "le", "short-edge": "se",
            "fbcnn": "fb", "jpeg-quality": "qf", "fbcnn-device": "fd",
            "pre-megapixels": "mp", "pre-resample": "rs", "noise": "n",
            "noise-seed": "ns", "tile": "t", "tile-width": "tw", "tile-height": "th",
            "overlap": "ov", "tile-upscale-resolution": "tr", "strategy": "st",
            "blend": "bl", "format": "fmt", "quality": "q", "seed": "sd",
            "cuda-device": "gpu", "attention-mode": "attn", "color-correction": "cc",
            "blocks-to-swap": "bs", "swap-io-components": "sw",
            "dit-offload-device": "do", "vae-offload-device": "vo",
            "tensor-offload-device": "to", "vae-tiled": "vt",
            "vae-tile-size": "vts", "vae-tile-overlap": "vto",
        }
        suffix = "_".join(f"{tags[field]}{_safe(values.get(field))}" for field in fields)
    else:
        suffix = "_".join(f"{field}-{_safe(values.get(field))}" for field in fields)
    stem = f"{base}_{suffix}" if suffix else base
    if len(stem) > MAX_OUTPUT_STEM:
        import hashlib
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        stem = stem[: MAX_OUTPUT_STEM - len(digest) - 2].rstrip("-_") + "__" + digest
    return stem
