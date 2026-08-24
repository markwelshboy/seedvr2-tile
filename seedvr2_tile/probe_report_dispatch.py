from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import probe_report_rebuild as _legacy
from . import probe_scene_sweep as _scene


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-sweep-report",
        description="Rebuild SeedVR2 probe reports from existing saved probe artifacts without inference.",
    )
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--comparison-crop-fraction", type=float, default=0.50)
    parser.add_argument("--cell-size", type=int, default=420)
    return parser


def _pre_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"native", "none", "off"}:
        return None
    return float(value)


def _source(item: dict) -> _scene.SourceImage:
    relative = Path(str(item["source"]))
    return _scene.SourceImage(
        source_id=str(item["source_id"]),
        source_path=relative,
        relative=relative,
        width=int(item["width"]),
        height=int(item["height"]),
        megapixels=float(item["megapixels"]),
        bucket=str(item["bucket"]),
        selected=bool(item.get("selected", False)),
    )


def _probe(item: dict) -> _scene._base.Probe:
    return _scene._base.Probe(
        probe_id=str(item["probe_id"]),
        label=str(item["label"]),
        center_x=float(item["center_x"]),
        center_y=float(item["center_y"]),
        reference_pre_megapixels=_pre_value(item.get("reference_pre_megapixels")),
        reference_tile_index=int(item["reference_tile_index"]),
        brightness=float(item.get("brightness", 0.0)),
        detail=float(item.get("detail", 0.0)),
    )


def _window(item: dict) -> _scene.SceneWindow:
    return (
        float(item["x0"]),
        float(item["y0"]),
        float(item["x1"]),
        float(item["y1"]),
    )


def _result(root: Path, row: dict[str, str]) -> _scene.SceneResult:
    return _scene.SceneResult(
        source_id=row["source_id"],
        source_name=row["source"],
        bucket=row["bucket"],
        probe_id=row["probe_id"],
        probe_label=row["probe_label"],
        pre_megapixels=_pre_value(row["pre_mp"]),
        scale=float(row["scale"]),
        noise=float(row["noise"]),
        processed_width=int(row["processed_width"]),
        processed_height=int(row["processed_height"]),
        total_tiles=int(row["total_tiles"]),
        tile_indices=tuple(int(value) for value in row["tile_indices"].split(",") if value != ""),
        group_resolution=int(row["group_resolution"]),
        capture_window=(
            float(row["capture_x0"]),
            float(row["capture_y0"]),
            float(row["capture_x1"]),
            float(row["capture_y1"]),
        ),
        input_scene_path=root / row["input_scene"],
        output_scene_path=root / row["output_scene"],
    )


def _rebuild_v2(args: argparse.Namespace, root: Path, manifest: dict) -> int:
    settings = manifest.get("settings") or {}
    capture_fraction = float(settings.get("probe_capture_fraction", 0.50))
    if not 0 < args.comparison_crop_fraction <= capture_fraction:
        raise SystemExit(
            f"--comparison-crop-fraction must be in (0, {capture_fraction:g}] for this sweep; "
            "rerun seedvr2-sweep with a larger --probe-capture-fraction to inspect a wider scene window"
        )
    if args.cell_size < 96:
        raise SystemExit("--cell-size must be >= 96")

    pre_raw = settings.get("pre_by_bucket") or {}
    pre_by_bucket = {
        bucket: tuple(_pre_value(value) for value in pre_raw.get(bucket, []))
        for bucket in ("small", "medium", "large")
    }
    scales = tuple(float(value) for value in settings.get("scales", []))
    noise_values = tuple(float(value) for value in settings.get("noise_values", []))
    if not scales or not noise_values:
        raise SystemExit("manifest is missing scales/noise_values")

    sources: list[_scene.SourceImage] = []
    probes_by_source: dict[str, list[_scene._base.Probe]] = {}
    capture_windows: dict[tuple[str, str], _scene.SceneWindow] = {}
    for item in manifest.get("sources", []):
        source = _source(item)
        sources.append(source)
        probes: list[_scene._base.Probe] = []
        for probe_item in item.get("probes", []):
            probe = _probe(probe_item)
            probes.append(probe)
            capture_windows[(source.source_id, probe.probe_id)] = _window(probe_item["capture_window"])
        probes_by_source[source.source_id] = probes

    results_path = root / "results.csv"
    if not results_path.is_file():
        raise SystemExit(f"results not found: {results_path}")
    result_index: dict[tuple[str, str, float | None, float, float], _scene.SceneResult] = {}
    missing: list[Path] = []
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result = _result(root, row)
            if not result.input_scene_path.is_file():
                missing.append(result.input_scene_path)
            if not result.output_scene_path.is_file():
                missing.append(result.output_scene_path)
            result_index[(result.source_id, result.probe_id, result.pre_megapixels, result.scale, result.noise)] = result
    if missing:
        sample = "\n  ".join(str(path) for path in missing[:8])
        raise SystemExit(f"saved scene artifacts are missing:\n  {sample}")

    sheets = _scene._render_reports(
        root,
        sources=sources,
        probes_by_source=probes_by_source,
        capture_windows=capture_windows,
        pre_by_bucket=pre_by_bucket,
        scales=scales,
        noise_values=noise_values,
        capture_fraction=capture_fraction,
        comparison_fraction=args.comparison_crop_fraction,
        cell_size=args.cell_size,
        result_index=result_index,
        max_output_mp=float(settings.get("max_output_mp", 20.0)),
    )
    print(f"Rebuilt {sheets} scene-registered probe sheet(s); no SeedVR2 inference performed.")
    print(f"Report: {root / 'index.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw)
    root = args.sweep_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") == "post-preprocess-probe-scenes-v2":
        return _rebuild_v2(args, root, manifest)

    try:
        return _legacy.main(raw)
    except RuntimeError as exc:
        if "probe center lies on/outside a candidate core" in str(exc):
            raise SystemExit(
                "This legacy probe sweep did not save every tile needed for exact scene registration. "
                "Regenerate it with the current seedvr2-sweep, which uses multi-tile scene probes."
            ) from exc
        raise


if __name__ == "__main__":
    raise SystemExit(main())
