# Interactive sweep reports

Probe sweeps now generate an interactive `index.html` after the normal comparison report is built. The original generated page is preserved as `comparison-index.html`.

## Selection model

The selection unit is a source image, not an individual probe. A source may have detail, dark, center, or additional probe views, but all of those views share one selected recipe.

Selecting a recipe under any probe automatically checks the matching recipe under every other probe for that source. Selecting another recipe replaces the previous choice; one source cannot have multiple active recipes.

For multi-latent sweeps, the top-level interactive report aggregates all latent child runs, so the recipe identity includes:

- pre-resize megapixels
- output scale
- RGB pixel preprocessing noise
- Numz input noise scale
- Numz latent noise scale

The Download selected recipes button creates `seedvr2-selections.json` in the browser. The file uses format `seedvr2-selection-v1`, contains one entry per selected source, and also records common inference defaults such as model, seed, tile size, overlap, attention mode, and color correction when available.

Selections are cached in browser local storage using the report path, so reloading a report served from the same URL restores the current choices.

## Local report server

The report is static HTML/JavaScript, but serving it over HTTP gives the most predictable browser behavior:

```bash
seedvr2-sweep-serve /path/to/seedvr2_sweep
```

The server binds to `127.0.0.1` and chooses a free port by default. It prints the URL. To ask Python to open the default browser:

```bash
seedvr2-sweep-serve /path/to/seedvr2_sweep --open
```

Use `--bind` only when deliberately exposing the report beyond localhost.

## Report-only rebuilds

Probe report rebuilds remain GPU-free:

```bash
seedvr2-sweep-report /path/to/probe_sweep --comparison-crop-fraction 0.25 --cell-size 600
```

After rebuilding the normal probe artifacts, this command also regenerates the interactive selection page.

Full-image sweeps have a matching report-only command:

```bash
seedvr2-sweep-full-report /path/to/full_sweep
```

It rebuilds HTML and compact production previews from existing outputs/results. It does not invoke SeedVR2. Existing comparison-sheet PNGs are reused rather than regenerated.
