from __future__ import annotations

import sys

from . import cli as _base
from .noise_controls import extract_value_option, replace_option, upstream_noise


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw or "-h" in raw:
        print(
            "Noise controls added by seedvr2-tile:\n"
            "  --pixel-noise FLOAT        RGB Gaussian preprocessing noise (alias for legacy --noise)\n"
            "  --input-noise-scale FLOAT  Numz input noise before VAE encoding (default: 0)\n"
            "  --latent-noise-scale FLOAT Numz diffusion-scheduled conditioning latent noise (default: 0)\n"
        )
        return _base.main(raw)

    raw = replace_option(raw, {"--pixel-noise"}, "--noise")
    raw, latent_raw = extract_value_option(raw, {"--latent-noise-scale"}, default="0")
    raw, input_raw = extract_value_option(raw, {"--input-noise-scale"}, default="0")
    latent = float(latent_raw or 0.0)
    input_noise = float(input_raw or 0.0)
    with upstream_noise(latent=latent, input_noise=input_noise):
        return _base.main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
