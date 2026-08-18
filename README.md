# seedvr2-tile

Standalone spatially tiled batch upscaling for [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) — **no ComfyUI server, workflow, browser, or API queue required**.

`seedvr2-tile` deliberately does not fork SeedVR2 inference. It prepares overlapping spatial tiles, runs Numz's native `inference_cli.py` in directory mode with model caching enabled, then stitches the processed tiles back into the final image.

## Why

SeedVR2's native VAE tiling reduces VAE memory use, but the DiT still processes the complete latent. Spatial tiling is different: each region of a large image is sent through SeedVR2 independently, allowing large final images while bounding per-inference VRAM use.

This frontend also supports an optional **preprocess stage before tiling**:

```text
source
  -> optional resize-to-megapixels
  -> optional Gaussian image noise
  -> spatial tiles
  -> SeedVR2
  -> stitch
```

That makes it possible to keep the source size and merely add variance, downscale before rebuilding detail, or combine both approaches.

## Status

Early standalone MVP. The orchestration, tiling, preprocessing, config, and stitching paths are structurally tested. Real-GPU validation against SeedVR2 model variants is still needed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Clone/update Numz SeedVR2 into ~/.cache/seedvr2-tile/... and install its deps
seedvr2-tile setup --install-deps
```

If you already have the standalone Numz repository elsewhere:

```bash
export SEEDVR2_ROOT=/path/to/ComfyUI-SeedVR2_VideoUpscaler
```

or pass `--seedvr2-root`.

## Basic batch use

```bash
seedvr2-tile input/ output/ --scale 2
```

Equivalent explicit form:

```bash
seedvr2-tile run input/ output/ --scale 2
```

Recommended starting point:

```bash
seedvr2-tile input/ output/ \
  --scale 2 \
  --tile 1024 \
  --overlap 64 \
  --tile-upscale-resolution 2048 \
  --strategy chess \
  --blend multiband
```

The SeedVR2 tile result is resized to the exact requested final geometric scale before stitching, so `--tile-upscale-resolution` acts as an inference-resolution ceiling rather than changing final dimensions.

## Preprocessing: resize and/or add detail variance

Preprocessing happens **before spatial tiling** and before final output dimensions are computed.

The implementation follows the relevant ComfyUI semantics:

- target megapixels use `megapixels * 1024 * 1024` total pixels while preserving aspect ratio
- noise is additive Gaussian noise: `clip(image + strength * N(0,1), 0, 1)`
- alpha follows the resize geometry but is never noise-corrupted

### Noise only, keep source dimensions

```bash
seedvr2-tile input/ output/ \
  --noise 0.015 \
  --scale 2
```

### Resize before the tiled upscale

```bash
seedvr2-tile input/ output/ \
  --pre-megapixels 1.0 \
  --pre-resample lanczos \
  --scale 2
```

### Resize and add noise

```bash
seedvr2-tile input/ output/ \
  --pre-megapixels 1.0 \
  --pre-resample lanczos \
  --noise 0.015 \
  --noise-seed 42 \
  --scale 2
```

Useful starting noise strengths from the advanced tiled workflow are approximately:

- `0.005` — almost nothing
- `0.01` — ultra-light
- `0.015` — light
- `0.03` — medium
- `0.04` — high

The final `--scale`, `--long-edge`, or `--short-edge` applies to the **preprocessed** dimensions. For example, taking a source down to ~1 MP and then using `--scale 2` yields roughly 4 MP output.

When preprocessing is active, `--keep-work` retains the exact preprocessed images under `preprocessed/` so the source, degraded/noised source, SeedVR2 tiles, and final result can be compared directly.

## Target-size modes

`--scale`, `--long-edge`, and `--short-edge` are mutually exclusive:

```bash
seedvr2-tile input.png output/ --scale 2
seedvr2-tile input.png output/ --long-edge 4096
seedvr2-tile input.png output/ --short-edge 2048
```

If none is supplied, the default is `--scale 2`.

## JSON configuration

For repeatable batch recipes, most run options can live in a JSON config file. The recommended sectional format is:

```json
{
  "preprocess": {
    "megapixels": 1.0,
    "resample": "lanczos",
    "noise": 0.015,
    "noise_seed": 42
  },
  "upscale": {
    "scale": 2.0
  },
  "tiling": {
    "tile": 1024,
    "overlap": 64,
    "tile_upscale_resolution": 2048,
    "strategy": "chess",
    "blend": "multiband"
  },
  "backend": {
    "attention_mode": "sageattn_2",
    "color_correction": "lab",
    "blocks_to_swap": 0,
    "vae_tiled": false
  },
  "io": {
    "format": "png",
    "recursive": false,
    "overwrite": false
  }
}
```

An example lives at `examples/detail.json`.

Run with positional paths:

```bash
seedvr2-tile input/ output/ --config examples/detail.json
```

You can alternatively put `input` and `output` under `io` and run:

```bash
seedvr2-tile --config my-job.json
```

**Explicit CLI options override config values**, so saved recipes remain easy to tweak:

```bash
seedvr2-tile input/ output/ \
  --config examples/detail.json \
  --noise 0.03 \
  --scale 3
```

Boolean values have matching `--no-*` forms, so a configured option can be disabled for one invocation, for example:

```bash
seedvr2-tile input/ output/ --config settings.json --no-vae-tiled
```

## SeedVR2 passthroughs

Useful backend options include:

```text
--dit-model MODEL
--model-dir PATH
--cuda-device 0
--attention-mode sdpa|flash_attn_2|flash_attn_3|sageattn_2|sageattn_3
--blocks-to-swap N
--swap-io-components
--dit-offload-device cpu
--vae-offload-device cpu
--tensor-offload-device cpu
--vae-tiled
--vae-tile-size 1024
--vae-tile-overlap 128
--color-correction lab|wavelet|wavelet_adaptive|hsv|adain|none
```

Example lower-VRAM run:

```bash
seedvr2-tile input/ output/ \
  --scale 2 \
  --tile 768 \
  --overlap 64 \
  --blocks-to-swap 24 \
  --swap-io-components \
  --dit-offload-device cpu \
  --vae-offload-device cpu \
  --vae-tiled
```

## Batch/cache behavior

Tiles are grouped by the SeedVR2 processing resolution they require. Each group is sent to Numz's CLI as a directory with:

```text
--batch_size 1
--cache_dit
--cache_vae
```

This keeps unrelated spatial tiles from being treated as temporal video frames while keeping models hot across the batch.

## Stitching

Available methods:

- `multiband` — Laplacian-pyramid blending (default)
- `content-aware` — multiband with local sharpness preference in overlap regions
- `bilateral` — multiband plus edge-preserving seam smoothing
- `linear` — cosine-feathered weighted accumulation
- `simple` — direct averaging in overlap regions

Tiles use real neighboring source pixels as context. Reflection padding is only introduced outside the source image or to fill a partial final tile to the fixed processing footprint; synthetic regions are discarded before stitching.

## Debugging

Prepare tiles and print the exact Numz command without running inference:

```bash
seedvr2-tile input/ output/ --dry-run --keep-work
```

Keep intermediate input/output tiles:

```bash
seedvr2-tile input/ output/ --keep-work
```

or use an explicit work directory:

```bash
seedvr2-tile input/ output/ --work-dir ./work
```

## Relationship to the ComfyUI tiling node

The spatial tiling approach is inspired by Moonwhaler's [`comfyui-seedvr2-tilingupscaler`](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler), but this project is a standalone implementation and does not import ComfyUI or the Moonwhaler node.
