from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import probe_sweep as _base
from . import probe_sweep_report as _report
from .sweep import SourceImage


def _pre_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"native", "none", "off"}:
        return None
    return float(value)


def _source_from_manifest(item: dict) -> SourceImage:
    relative = Path(str(item["source"]))
    return SourceImage(
        source_id=str(item["source_id"]),
        source_path=relative,
        relative=relative,
        width=int(item["width"]),
        height=int(item["height"]),
        megapixels=float(item["megapixels"]),
        bucket=str(item["bucket"]),
        selected=bool(item.get("selected", False)),
    )


def _probe_from_manifest(item: dict) -> _base.Probe:
    return _base.Probe(
        probe_id=str(item["probe_id"]),
        label=str(item["label"]),
        center_x=float(item["center_x"]),
        center_y=float(item["center_y"]),
        reference_pre_megapixels=_pre_value(item.get("reference_pre_megapixels")),
        reference_tile_index=int(item["reference_tile_index"]),
        brightness=float(item.get("brightness", 0.0)),
        detail=float(item.get("detail", 0.0)),
    )


def _result_from_csv(root: Path, row: dict[str, str]) -> _base.PlannedProbeResult:
    return _base.PlannedProbeResult(
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
        selected_tile_index=int(row["selected_tile_index"]),
        selected_filename="",
        group_resolution=int(row["group_resolution"]),
        input_core_path=root / row["input_core"],
        output_core_path=root / row["output_core"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-sweep-report",
        description="Rebuild probe comparison sheets and HTML from an existing SeedVR2 probe sweep without inference.",
    )
    parser.add_argument("sweep_dir", type=Path, help="existing fetched probe sweep directory")
    parser.add_argument(
        "--comparison-crop-fraction",
        type=float,
        default=0.50,
        help="fraction of the probe core shown in detail cells; smaller values zoom further (default: 0.50)",
    )
    parser.add_argument("--cell-size", type=int, default=420, help="comparison cell size in pixels (default: 420)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0 < args.comparison_crop_fraction <= 1:
        raise SystemExit("--comparison-crop-fraction must be in (0, 1]")
    if args.cell_size < 96:
        raise SystemExit("--cell-size must be >= 96")

    root = args.sweep_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    results_path = root / "results.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not results_path.is_file():
        raise SystemExit(f"results not found: {results_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "post-preprocess-probe-tiles":
        raise SystemExit(
            "this command expects a probe-first sweep (manifest mode 'post-preprocess-probe-tiles')"
        )

    settings = manifest.get("settings") or {}
    pre_raw = settings.get("pre_by_bucket") or {}
    pre_by_bucket = {
        bucket: tuple(_pre_value(value) for value in pre_raw.get(bucket, []))
        for bucket in ("small", "medium", "large")
    }
    scales = tuple(float(value) for value in settings.get("scales", []))
    noise_values = tuple(float(value) for value in settings.get("noise_values", []))
    if not scales or not noise_values:
        raise SystemExit("manifest is missing scales/noise_values needed to rebuild the report")

    sources: list[SourceImage] = []
    probes_by_source: dict[str, list[_base.Probe]] = {}
    for item in manifest.get("sources", []):
        source = _source_from_manifest(item)
        sources.append(source)
        probes_by_source[source.source_id] = [
            _probe_from_manifest(probe) for probe in item.get("probes", [])
        ]

    result_index: dict[tuple[str, str, float | None, float, float], _base.PlannedProbeResult] = {}
    missing_files: list[Path] = []
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result = _result_from_csv(root, row)
            if not result.input_core_path.is_file():
                missing_files.append(result.input_core_path)
            if not result.output_core_path.is_file():
                missing_files.append(result.output_core_path)
            key = (
                result.source_id,
                result.probe_id,
                result.pre_megapixels,
                result.scale,
                result.noise,
            )
            result_index[key] = result

    if missing_files:
        sample = "\n  ".join(str(path) for path in missing_files[:8])
        suffix = "" if len(missing_files) <= 8 else f"\n  ... and {len(missing_files) - 8} more"
        raise SystemExit(f"probe core files referenced by results.csv are missing:\n  {sample}{suffix}")

    _report._DETAIL_FRACTION = args.comparison_crop_fraction
    _report._REPORT_TILE = int(settings.get("tile", 1024))
    _report._REPORT_OVERLAP = int(settings.get("overlap", 64))
    _report._REPORT_STRATEGY = str(settings.get("strategy", "chess"))

    max_output_mp = float(settings.get("max_output_mp", 20.0))
    sheets = 0
    for source in sources:
        if not source.selected:
            continue
        pre_values = pre_by_bucket.get(source.bucket, ())
        for probe in probes_by_source.get(source.source_id, []):
            for noise in noise_values:
                target = _base._case_dir(root, source, probe, noise) / "comparison.png"
                _report._make_probe_sheet_detail_first(
                    source=source,
                    probe=probe,
                    noise=noise,
                    pre_values=pre_values,
                    scales=scales,
                    output_root=root,
                    result_index=result_index,
                    cell_size=args.cell_size,
                    target=target,
                    max_output_mp=max_output_mp,
                )
                sheets += 1

    _report._write_html(
        root,
        sources=sources,
        probes_by_source=probes_by_source,
        noise_values=noise_values,
        plan_only=False,
    )
    print(
        f"Rebuilt {sheets} probe comparison sheet(s) from existing cores; "
        f"no SeedVR2 inference performed."
    )
    print(f"Report: {root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
