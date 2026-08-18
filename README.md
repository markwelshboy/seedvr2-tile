# seedvr2-tile

Standalone spatially tiled batch upscaling for [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) — **no ComfyUI server, workflow, browser, or API queue required**.

`seedvr2-tile` deliberately does not fork SeedVR2 inference. It handles optional image cleanup/degradation, creates overlapping spatial tiles, runs Numz's native `inference_cli.py` with model caching enabled, and stitches the processed tiles back together.

## Pipeline

```text
source
  -> optional FBCNN JPEG artifact removal
  -> optional resize to target megapixels
  -> optional additive Gaussian noise
  -> spatial tiling
  -> SeedVR2
  -> stitch
  -> output
```

This lets the tool operate anywhere between faithful upscale and deliberately more perceptual reconstruction. For noisy or compressed photographs, intentionally removing bad high-frequency information before SeedVR2 can produce a more natural result than simply enlarging every source artifact.

## Status

Early standalone implementation. Geometry, preprocessing, config loading, and orchestration have unit coverage; real-GPU validation across SeedVR2/FBCNN combinations is still ongoing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Clone/update Numz SeedVR2 and install its dependencies.
seedvr2-tile setup --install-deps
```

If you want FBCNN JPEG cleanup too:

```bash
seedvr2-tile setup --install-deps --fbcnn
```

That checks out the official FBCNN repository under:

```text
~/.cache/seedvr2-tile/FBCNN
```

The official `fbcnn_color.pth` weight is downloaded from the FBCNN GitHub release on first use.

Existing checkouts can instead be supplied through:

```bash
export SEEDVR2_ROOT=/path/to/ComfyUI-SeedVR2_VideoUpscaler
export FBCNN_ROOT=/path/to/FBCNN
```

## Basic use

```bash
seedvr2-tile input/ output/ --scale 2
```

Equivalent explicit form:

```bash
seedvr2-tile run input/ output/ --scale 2
```

Recommended tiled starting point:

```bash
seedvr2-tile input/ output/ \
  --scale 2 \
  --tile 1024 \
  --overlap 64 \
  --tile-upscale-resolution 2048 \
  --strategy chess \
  --blend multiband
```

## Model selection and automatic download

The default model is **3B FP8**, because the smaller model is often preferable for natural image restoration as well as being lighter to run. Select models with friendly aliases:

```bash
seedvr2-tile input/ output/ --model 3b
seedvr2-tile input/ output/ --model 7b
seedvr2-tile input/ output/ --model 7b-sharp
```

List the built-in aliases:

```bash
seedvr2-tile models
```

Current aliases include:

```text
3b               3B FP8 (default)
3b-fp16          3B FP16
3b-q8            3B GGUF Q8_0
3b-q4            3B GGUF Q4_K_M
7b               7B mixed FP8
7b-fp16          7B FP16
7b-q4            7B GGUF Q4_K_M
7b-sharp         7B sharp mixed FP8
7b-sharp-fp16    7B sharp FP16
7b-sharp-q4      7B sharp GGUF Q4_K_M
```

An exact Numz filename is also accepted through `--model` or the backward-compatible `--dit-model` option.

Before preprocessing begins, `seedvr2-tile` runs a model preflight. It delegates to **Numz's own downloader**, which checks all registered model paths, downloads a missing DiT and shared VAE from Hugging Face, supports resumable downloads, validates hashes, and replaces corrupt cached files.

```bash
seedvr2-tile input/ output/ --model 3b
# -> model/VAE checked or downloaded first
# -> then FBCNN/resize/noise
# -> then tiling + SeedVR2
```

Choose a specific model directory if desired:

```bash
seedvr2-tile input/ output/ \
  --model 3b \
  --model-dir /models/seedvr2
```

For an intentionally offline run, disable the preflight download:

```bash
seedvr2-tile input/ output/ --model 3b --no-model-download
```

With download disabled, the selected model must already be discoverable by Numz when inference starts.

## Preprocessing

Preprocessing is optional and always runs in this order:

```text
FBCNN -> resize -> noise
```

### FBCNN JPEG artifact removal

Blind/automatic quality-factor estimation:

```bash
seedvr2-tile input/ output/ \
  --fbcnn \
  --jpeg-quality auto \
  --scale 2
```

Known JPEG quality factor:

```bash
seedvr2-tile input/ output/ \
  --fbcnn \
  --jpeg-quality 95 \
  --scale 2
```

`--jpeg-quality` is an actual JPEG quality factor from `1..100`, matching the official FBCNN flexible-control interface. Lower values tell FBCNN to perform stronger artifact restoration. `auto` uses the model's blind predicted quality factor.

FBCNN runs **before resizing** so it sees JPEG blocking/ringing on the original pixel grid. The model is released from memory before SeedVR2 starts so a CUDA FBCNN pass does not continue occupying VRAM during upscale inference.

Choose the FBCNN device independently if desired:

```bash
--fbcnn-device auto
--fbcnn-device cpu
--fbcnn-device cuda:0
```

FBCNN is pixel-based, so it can also clean a PNG/WebP that was originally sourced from JPEG; it does not require access to JPEG quantization tables.

### Resize to target megapixels

```bash
seedvr2-tile input/ output/ \
  --pre-megapixels 1.0 \
  --pre-resample lanczos \
  --scale 2
```

Megapixel sizing follows ComfyUI `Scale Image to Total Pixels` semantics: `1 MP = 1024 * 1024` total pixels, with aspect ratio preserved.

This is intentionally useful as a degradation stage. A grainy source can be downsampled to discard sensor/compression noise and then handed to SeedVR2 to reconstruct more coherent high-frequency detail.

### Add image noise

Noise only, keeping source dimensions:

```bash
seedvr2-tile input/ output/ \
  --noise 0.015 \
  --scale 2
```

Resize then add noise:

```bash
seedvr2-tile input/ output/ \
  --pre-megapixels 1.0 \
  --noise 0.015 \
  --noise-seed 42 \
  --scale 2
```

Noise matches the ComfyUI operation conceptually: `clip(image + strength * N(0,1), 0, 1)`. Useful experimental strengths include `0.005`, `0.01`, `0.015`, `0.03`, and `0.04`.

### Low-light / compressed-photo example

For a grainy JPEG where pure enlargement makes the artifacts worse:

```bash
seedvr2-tile input/ output/ \
  --fbcnn \
  --jpeg-quality auto \
  --pre-megapixels 1.0 \
  --noise 0 \
  --scale 2 \
  --tile 1024 \
  --overlap 64 \
  --blend multiband
```

If FBCNN reports or visually behaves too aggressively, compare `auto` against the known source QF. Noise can then be introduced separately if additional synthesized detail is useful.

## Output size modes

The final output size is computed **after preprocessing**.

`--scale`, `--long-edge`, and `--short-edge` are mutually exclusive:

```bash
seedvr2-tile input.png output/ --scale 2
seedvr2-tile input.png output/ --long-edge 4096
seedvr2-tile input.png output/ --short-edge 2048
```

For example, a 4 MP source preprocessed to 1 MP and then run with `--scale 2` produces roughly a 4 MP output.

## Spatial tiling

Spatial tiling is distinct from SeedVR2's VAE tiling. Each image region is independently sent through SeedVR2, bounding the DiT inference footprint as well as the VAE footprint.

Useful controls:

```text
--tile 1024
--tile-width 1024
--tile-height 1024
--overlap 64
--tile-upscale-resolution 2048
--strategy chess|linear
--blend multiband|content-aware|bilateral|linear|simple
```

Tiles use real neighboring source pixels as context. Reflection padding is only introduced outside usable source pixels / partial edge tiles and is discarded before stitching.

## SeedVR2 options

Useful passthroughs include:

```text
--model 3b|7b|7b-sharp|...
--dit-model EXACT_FILENAME
--model-dir PATH
--model-download / --no-model-download
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

With the normal `--scale` workflow, same-sized tile groups are submitted to Numz's CLI as a directory with `--cache_dit --cache_vae`, keeping the heavy models hot across the batch.

## JSON configuration

For repeatable recipes, use sectional JSON. Explicit CLI arguments override config values.

```json
{
  "preprocess": {
    "fbcnn": {
      "enabled": true,
      "quality": "auto",
      "device": "auto"
    },
    "megapixels": 1.0,
    "resample": "lanczos",
    "noise": 0.0,
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
    "model": "3b",
    "model_download": true,
    "attention_mode": "sdpa",
    "color_correction": "lab",
    "vae_tiled": false
  },
  "io": {
    "format": "png",
    "recursive": false,
    "overwrite": false
  }
}
```

Run a recipe:

```bash
seedvr2-tile input/ output/ --config examples/lowlight-jpeg-naturalize.json
```

Override one setting without editing it:

```bash
seedvr2-tile input/ output/ \
  --config examples/lowlight-jpeg-naturalize.json \
  --jpeg-quality 95
```

Boolean options have inverse CLI switches, so config values can be temporarily disabled:

```bash
seedvr2-tile input/ output/ --config recipe.json --no-fbcnn
seedvr2-tile input/ output/ --config recipe.json --no-vae-tiled
```

## Debugging and tuning

Prepare tiles and print SeedVR2 commands without running inference:

```bash
seedvr2-tile input/ output/ --dry-run --keep-work
```

Keep intermediate files:

```bash
seedvr2-tile input/ output/ --keep-work
```

When preprocessing is active, the work tree contains the exact image that enters spatial tiling:

```text
work/
├── preprocessed/
│   └── image.png
├── r2048/
│   ├── input/      # SeedVR2 input tiles
│   └── output/     # SeedVR2 output tiles
└── ...
```

That makes it practical to compare source → cleanup/degradation → SeedVR2 tiles → final result rather than tuning blind.

## Relationship to the ComfyUI tiling node

The spatial tiling approach is inspired by Moonwhaler's [`comfyui-seedvr2-tilingupscaler`](https://github.com/moonwhaler/comfyui-seedvr2-tilingupscaler), but this project is a standalone implementation and does not import ComfyUI or the Moonwhaler node.

FBCNN remains an external official backend from [`jiaxi-jiang/FBCNN`](https://github.com/jiaxi-jiang/FBCNN); its code is not vendored here.
