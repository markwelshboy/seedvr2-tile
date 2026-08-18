# seedvr2-tile

Standalone spatially tiled batch upscaling for [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) — **no ComfyUI server, workflow, browser, or API queue required**.

`seedvr2-tile` deliberately does not fork SeedVR2 inference. It prepares overlapping spatial tiles, runs Numz's native `inference_cli.py` in directory mode with model caching enabled, then stitches the processed tiles back into the final image.

## Why

SeedVR2's native VAE tiling reduces VAE memory use, but the DiT still processes the complete latent. Spatial tiling is different: each region of a large image is sent through SeedVR2 independently, allowing large final images while bounding per-inference VRAM use.

## Status

Early standalone MVP. The batch/orchestration and stitching path is usable; real-GPU validation against multiple SeedVR2 model variants is still needed.

Tested structurally against Numz SeedVR2 CLI v2.5.x, which supports directory input, `--cache_dit`, `--cache_vae`, BlockSwap, VAE tiling, GGUF, and multiple attention backends.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Clone/update Numz SeedVR2 into ~/.cache/seedvr2-tile/... and install its deps
seedvr2-tile setup --install-deps
```

If you already have the standalone Numz repository elsewhere, skip setup and either set:

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

Recommended starting point for high-quality image work:

```bash
seedvr2-tile input/ output/ \
  --scale 2 \
  --tile 1024 \
  --overlap 64 \
  --tile-upscale-resolution 2048 \
  --strategy chess \
  --blend multiband
```

For a 4× final image while limiting how large each tile is actually processed by SeedVR2:

```bash
seedvr2-tile input/ output/ \
  --scale 4 \
  --tile 1024 \
  --overlap 64 \
  --tile-upscale-resolution 2048
```

The SeedVR2 tile result is resized to the exact final geometric scale before stitching, so `--tile-upscale-resolution` acts as an inference-resolution ceiling rather than changing the requested final dimensions.

## Target-size modes

`--scale`, `--long-edge`, and `--short-edge` are mutually exclusive:

```bash
seedvr2-tile input.png output/ --scale 2
seedvr2-tile input.png output/ --long-edge 4096
seedvr2-tile input.png output/ --short-edge 2048
```

With `--scale`, every image normally lands in the same tile-resolution group, so the entire directory of tiles is sent to the Numz CLI in one invocation and its DiT/VAE caches remain hot across the batch.

## SeedVR2 options

Useful passthroughs include:

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

Example low-VRAM run:

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

## Stitching

Available methods:

- `multiband` — Laplacian-pyramid blending (default)
- `content-aware` — multiband with a local sharpness preference in overlap regions
- `bilateral` — multiband plus edge-preserving seam smoothing
- `linear` — cosine-feathered weighted accumulation
- `simple` — direct averaging in overlap regions

Tiles use real neighboring image pixels as context. Reflection padding is only introduced outside the source image or to fill a partial final tile to the fixed processing footprint; that synthetic region is discarded before stitching.

## Debugging

Prepare all tiles and print the exact Numz commands without running inference:

```bash
seedvr2-tile input/ output/ --scale 2 --dry-run --keep-work
```

Keep intermediate input/output tiles:

```bash
seedvr2-tile input/ output/ --keep-work
```

or choose an explicit work directory:

```bash
seedvr2-tile input/ output/ --work-dir ./work
```

## Relationship to the ComfyUI tiling node

The spatial tiling approach is inspired by Moonwhaler's [`comfyui-seedvr2-tilingupscaler`](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler), but this project is a standalone implementation and does not import ComfyUI or the Moonwhaler node.
