# Full sweep reports

`seedvr2-sweep-full` automatically switches its HTML presentation based on how many effective recipe combinations were requested.

When a selected bucket has exactly one pre-resize value, one scale value, and one pixel-noise value, the run is treated as a single-recipe production run. The report shows a lightweight preview of each completed full-resolution output and keeps the historical source/result contact sheets under a collapsed **Comparison views** disclosure.

When two or more effective recipe combinations are present, the existing comparison-oriented report is left unchanged.

For multi-value latent-noise runs, each latent value is already isolated in its own child run. Singleton child runs use the compact production presentation, while the top-level latent report remains the comparison surface across latent-noise values.
