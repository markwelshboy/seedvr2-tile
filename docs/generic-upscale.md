# Generic PyTorch upscaling

`seedvr2-tile` also exposes a generic full-image sweep frontend named `upscale` for Spandrel-compatible PyTorch super-resolution checkpoints.

It deliberately reuses the same bucket classification, megapixel preprocessing, Lanczos resize semantics, output naming, CSV results, and contact-sheet reporting as the SeedVR2 full sweep. This makes model comparisons meaningful: the same preprocessed image can be sent to different SR networks.

## Built-in Real-ESRGAN x2

The `realesrgan-x2plus` alias downloads and caches the official `RealESRGAN_x2plus.pth` checkpoint from the Real-ESRGAN GitHub release.

```bash
upscale INPUT OUTPUT \
  --model realesrgan-x2plus \
  --all-images \
  --only-bucket medium \
  --pre-medium 1.25 \
  --scales 2 \
  --pixel-noise-values 0 \
  --strict
```

The model is a native x2 network. The generic runner intentionally requires `--scales 2` rather than applying an additional resize after inference.

## SPAN and arbitrary checkpoints

Spandrel can identify compatible checkpoints from their state dict. `span-x2` validates that the supplied checkpoint is SPAN and native x2:

```bash
upscale INPUT OUTPUT \
  --model span-x2 \
  --model-file /path/on/worker/span-x2.pth \
  --pre-medium 1.25 \
  --scales 2
```

or with a direct downloadable checkpoint URL:

```bash
upscale INPUT OUTPUT \
  --model span-x2 \
  --model-url https://example.invalid/span-x2.pth \
  --pre-medium 1.25 \
  --scales 2
```

For another Spandrel-compatible model, use any label together with `--model-file` or `--model-url`; the native model scale is detected automatically and checked against `--scales`.

## Tiling

The generic backend defaults to 512-pixel non-overlapping output cores with 32 pixels of inference context around each core:

```text
--tile 512 --overlap 32
```

Only the core is copied into the final image, so neighboring tiles do not average or blur one another. Increase overlap if a model has a larger useful receptive field. `--tile 0` runs the complete preprocessed image in one model call when memory allows.

## Model cache

Downloaded checkpoints are stored under `${UPSCALER_MODEL_DIR}` when set, otherwise `~/.cache/upscale-models`. The Podlets `upscale` command sets this to its persistent worker cache so checkpoints survive disposable jobs.

`.pth` files use PyTorch serialization. Use checkpoints from sources you trust.
