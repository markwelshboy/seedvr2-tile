# SeedVR2 sweep harness

`seedvr2-sweep` is a probe-first experiment harness for noisy / low-light photo restoration. It inventories an unsorted input directory, buckets images by source megapixels, chooses a deterministic spread sample from each bucket, preprocesses the **entire source image** for each candidate recipe, then runs SeedVR2 only on representative tiles from the real post-preprocess tile grid.

That ordering is important. A crop is never independently resized to an absolute megapixel target, because that would not represent what `--pre-megapixels` does to the full image.

The model is fixed to `3b` (3B FP8) for sweep comparability.

## Probe methodology

For each selected source image, the harness first chooses up to 3 distinct probe locations from the highest-resolution preprocessing candidate in that bucket:

1. `detail` — the tile with the greatest mean local gradient magnitude
2. `dark` — the tile with the lowest mean luminance
3. `center` — the remaining tile nearest the image center

If the preprocessed image contains fewer than 3 tiles, the probe count automatically collapses to the number of available tiles. A 0.5–0.75 MP image will commonly be only one 1024-core spatial tile, so there is no approximation to make in that case.

Probe locations are stored as normalized image coordinates. For every other preprocessing candidate, the harness:

1. opens the full source;
2. performs the real FBCNN / resize / Gaussian-noise preprocessing in normal pipeline order;
3. creates the normal SeedVR2 spatial tile grid with real neighboring context;
4. maps each stored normalized probe location onto the nearest actual tile in that grid;
5. sends only those selected processing tiles through SeedVR2;
6. extracts the non-overlap core from each processed tile for comparison.

The result therefore answers: "what would SeedVR2 have done to this region during a full run with these exact preprocessing settings?"

## Inference deduplication

The probe harness also deduplicates GPU work. A unique inference is identified by:

- source image;
- preprocessing megapixel target;
- noise value;
- selected tile;
- actual SeedVR2 backend processing resolution.

Scale itself is not part of that key once it produces the same backend resolution. For example, with the normal `--tile-upscale-resolution 2048` cap, 2x and 3x may use the same SeedVR2 inference result and differ only in the final core resize. The report records both requested result cells while only paying for one SeedVR2 tile inference.

All unique probe tiles are collected first and grouped by backend resolution, so the upstream SeedVR2 CLI processes a whole resolution group in one invocation with its normal DiT/VAE caching.

## Default experiment

Buckets use SeedVR2's existing megapixel convention (`1 MP = 1024 * 1024` pixels):

- `small`: `< 1.25 MP`
- `medium`: `1.25 .. 4.0 MP`
- `large`: `> 4.0 MP`

The default source sample is 3 images per bucket, spread across the bucket by source megapixels rather than selected randomly.

Default pre-resize rows:

- small: `native, 0.50, 0.75 MP`
- medium: `native, 0.75, 1.00 MP`
- large: `1.00, 1.50, 2.00 MP`

Default reconstruction columns are `1.5x, 2x, 3x`. Noise is held at `0` for the initial coarse sweep. A pre-resize target is skipped for a source when it would actually enlarge that source, and combinations predicted to exceed 20 MP full-image output are skipped by default.

## Run locally

```bash
seedvr2-sweep ~/images/lowlight/ ./seedvr2_sweep/
```

Inventory, bucket and select probe locations without SeedVR2 inference:

```bash
seedvr2-sweep ~/images/lowlight/ ./seedvr2_sweep/ --plan-only
```

The output contains:

```text
seedvr2_sweep/
├── index.html
├── manifest.json
├── results.csv
└── reports/
    └── <bucket>/<source-id>/<probe-id>/noise-0/
        ├── comparison.png
        ├── overview.png
        ├── inputs/
        │   ├── pre-native.png
        │   ├── pre-0p75mp.png
        │   └── ...
        └── results/
            ├── <variant>.png
            └── ...
```

Each `comparison.png` is detail-first: cells show a square crop centered on the same normalized probe location in every preprocessing/scale result instead of shrinking the entire probe core into the cell. The default detail crop covers 50% of the core and the default cell is 420 px. `overview.png` retains the entire probe core for context.

The first column is the **actual post-preprocess input core** selected for that probe and row; subsequent columns are the requested SeedVR2 scales. Labels include the full preprocessed dimensions, actual tile index / tile count, SeedVR2 backend resolution, and predicted full-image output megapixels.

`manifest.json` records the normalized probe locations and the number of requested result cells versus unique SeedVR2 tile inferences. `results.csv` records the tile mapping and output path for every rendered result cell.

## Rebuild reports without inference

Once a probe sweep has completed, the saved input/result cores contain everything needed to redraw the visual report. `seedvr2-sweep-report` reads the existing `manifest.json` and `results.csv` and regenerates `comparison.png`, `overview.png`, and `index.html` **without invoking SeedVR2 or loading a GPU model**:

```bash
seedvr2-sweep-report ./seedvr2_sweep/
```

Try a tighter detail view without rerunning inference:

```bash
seedvr2-sweep-report ./seedvr2_sweep/ \
  --comparison-crop-fraction 0.35 \
  --cell-size 480
```

Smaller `--comparison-crop-fraction` values zoom further in. The crop remains centered on the original normalized probe coordinate even when different preprocessing choices map that point onto different tile grids.

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
--probe-tiles N                   # default 3
--comparison-crop-fraction F      # default 0.50; smaller = tighter report zoom
--cell-size PX                    # default 420 for probe reports
--fbcnn / --no-fbcnn
--jpeg-quality auto|QF
--seed N
--plan-only
```

The normal SeedVR2 tiling defaults remain fixed unless explicitly overridden: 1024 core tiles, 64 pixel overlap/context, 2048 tile upscale resolution, chess strategy, SDPA, LAB color correction, Lanczos pre-resize, and seed 42.

## Full-image sweep

The original full-image matrix implementation is retained as an explicit fallback / validation tool:

```bash
seedvr2-sweep-full ~/images/lowlight/ ./seedvr2_full_sweep/
```

That runs the previous whole-image sweep and is useful for validating that a probe-derived winner transfers to the complete image. `seedvr2-sweep` itself is probe-first by default.
