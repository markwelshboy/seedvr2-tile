# SeedVR2 sweep harness

`seedvr2-sweep` is a small experiment harness around `seedvr2-tile` for noisy / low-light photo restoration. It inventories an unsorted input directory, buckets images by source megapixels, chooses a deterministic spread sample from each bucket, runs a controlled parameter matrix with the 3B FP8 model, and builds visual comparison sheets plus machine-readable manifests.

## Default experiment

Buckets use SeedVR2's existing megapixel convention (`1 MP = 1024 * 1024` pixels):

- `small`: `< 1.25 MP`
- `medium`: `1.25 .. 4.0 MP`
- `large`: `> 4.0 MP`

The default sample is 3 images per bucket, spread across the bucket by source megapixels rather than selected randomly.

Default pre-resize rows:

- small: `native, 0.50, 0.75 MP`
- medium: `native, 0.75, 1.00 MP`
- large: `1.00, 1.50, 2.00 MP`

Default reconstruction columns are `1.5x, 2x, 3x`. Noise is held at `0` for the initial coarse sweep. A pre-resize target is skipped for a source when it would actually enlarge that source, and combinations predicted to exceed 20 MP output are skipped by default.

The model is deliberately fixed to `3b` (3B FP8) for sweep comparability.

## Run locally

```bash
seedvr2-sweep ~/images/lowlight/ ./seedvr2_sweep/
```

Inventory and plan without GPU inference:

```bash
seedvr2-sweep ~/images/lowlight/ ./seedvr2_sweep/ --plan-only
```

The output contains:

```text
seedvr2_sweep/
├── index.html
├── manifest.json
├── results.csv
├── results/
│   ├── small/
│   ├── medium/
│   └── large/
└── reports/
    └── <bucket>/<source-id>/noise-0/
        ├── full.png
        ├── crop-center.png
        ├── crop-upper.png
        ├── crop-lower.png
        ├── crop-left.png
        └── crop-right.png
```

Each comparison sheet is a matrix with pre-resize choices as rows and reconstruction scale as columns. The first comparison column repeats the same source image/crop, so every processed cell has an immediate baseline. Crops use identical normalized locations in the source and every result.

## Narrow noise refinement

After the coarse sheet identifies a useful resize/scale region, narrow the grid and add synthetic Gaussian noise:

```bash
seedvr2-sweep ~/images/lowlight/ ./seedvr2_noise_refine/ \
  --only-bucket medium \
  --pre-medium 0.75,1.0 \
  --scales 1.5,2 \
  --noise-values 0,0.005,0.01
```

This keeps the expensive noise axis out of the initial search while still making it easy to test whether a small amount of true Gaussian noise helps SeedVR2 reconstruct more coherent texture than the source sensor/compression noise.

## Useful controls

```text
--samples-per-bucket N
--all-images
--only-bucket small|medium|large   # repeatable
--small-max MP
--medium-max MP
--pre-small LIST
--pre-medium LIST
--pre-large LIST
--scales LIST
--noise-values LIST
--max-output-mp MP                # 0 disables the cap
--crop-fraction FRACTION
--cell-size PX
--fbcnn / --no-fbcnn
--jpeg-quality auto|QF
--seed N
--plan-only
--strict
```

The normal SeedVR2 tiling defaults remain fixed unless explicitly overridden: 1024 core tiles, 64 pixel overlap/context, 2048 tile upscale resolution, chess strategy, multiband stitch, SDPA, LAB color correction, Lanczos pre-resize, and seed 42.
