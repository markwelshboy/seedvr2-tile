# SeedVR2 noise controls

`seedvr2-tile` distinguishes three different noise mechanisms.

| Control | Where it acts | Sweep spelling |
| --- | --- | --- |
| Pixel noise | RGB image after optional pre-resize, before spatial tiling | `--pixel-noise-values` |
| Input noise | Numz transformed image immediately before VAE encoding | held at `0` by the sweep wrappers |
| Latent noise | Numz diffusion-scheduled corruption of the SR conditioning latent | `--latent-noise-values` |

The historical `--noise-values` option remains accepted and means **pixel noise**. New experiments should prefer `--pixel-noise-values` so reports and commands are unambiguous.

## Recommended low-light latent-noise probe

Once pre-resolution and output scale are known, hold RGB pixel noise at zero and sweep only the latent control. For example, for the large bucket with 1 MP preprocessing and 3x output:

```bash
seedvr2-sweep INPUT OUTPUT \
  --only-bucket large \
  --pre-large 1.0 \
  --scales 3 \
  --pixel-noise-values 0 \
  --latent-noise-values 0,0.015,0.03,0.05
```

A multi-latent probe sweep creates one exact base run per latent value and a top-level report. Corresponding scene-registered crops are collected into `overlay-crops/`; each leaf directory contains only the different latent-noise versions of the same crop, which is suitable for image overlay/flip viewers.

The sweep deliberately sends Numz `--input_noise_scale 0` while testing latent noise. This avoids mixing two different corruption mechanisms.

## Full-image application

After choosing a latent value, apply it to every image in a bucket with the full-image command. Example:

```bash
seedvr2-sweep-full INPUT OUTPUT \
  --all-images \
  --only-bucket large \
  --pre-large 1.0 \
  --scales 3 \
  --pixel-noise-values 0 \
  --latent-noise-values 0.03 \
  --strict
```

Multiple latent values are supported by `seedvr2-sweep-full` too; each value is written to a separate `latent-noise-*` subdirectory.

For a multi-latent full-image sweep, the top-level report groups the same source image and exact restoration recipe side-by-side across latent values. Full-resolution copies for overlay/flip tools are written under `overlay-images/`, with one leaf directory per source/recipe. Browser previews use smaller JPEG thumbnails so the comparison page remains responsive even when the restored outputs are large.

An already-completed full-image latent sweep can rebuild these comparisons without inference:

```bash
seedvr2-sweep-full-report OUTPUT
```

## Direct tile CLI

The direct CLI also exposes explicit names:

```bash
seedvr2-tile run INPUT OUTPUT \
  --pixel-noise 0 \
  --input-noise-scale 0 \
  --latent-noise-scale 0.03
```

`--pixel-noise` is an alias for the historical `--noise` option.
