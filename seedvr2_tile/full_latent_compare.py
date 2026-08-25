from __future__ import annotations

import csv
import html
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from .noise_controls import fmt_value


@dataclass(frozen=True)
class ComparisonItem:
    latent: float
    full_path: Path
    thumb_path: Path


@dataclass(frozen=True)
class ComparisonGroup:
    source_id: str
    source_name: str
    bucket: str
    pre_mp: str
    scale: str
    pixel_noise: str
    group_rel: Path
    items: tuple[ComparisonItem, ...]


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value).strip("._")
    return cleaned or "item"


def _recipe_token(pre_mp: str, scale: str, pixel_noise: str) -> str:
    pre = _safe_component(pre_mp.replace(".", "p"))
    scale_token = _safe_component(scale.replace(".", "p"))
    pixel = _safe_component(pixel_noise.replace(".", "p"))
    return f"pre-{pre}__scale-{scale_token}x__pixel-{pixel}"


def _write_thumb(source: Path, target: Path, *, max_edge: int = 560) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb.save(target, format="JPEG", quality=88, optimize=True)


def build_full_comparison_groups(
    root: Path,
    runs: Sequence[tuple[float, Path]],
) -> list[ComparisonGroup]:
    """Collect full-image outputs across latent runs, varying latent only.

    Matching is deliberately strict on source + pre-MP + scale + pixel noise so a
    group never mixes different restoration recipes merely because they share a
    source image.
    """
    overlay_root = root / "overlay-images"
    thumb_root = root / "comparison-thumbs"
    for path in (overlay_root, thumb_root):
        if path.exists():
            shutil.rmtree(path)

    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[float, Path]]] = {}
    for latent, run_dir in runs:
        csv_path = run_dir / "results.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "complete" or not row.get("output"):
                    continue
                source_path = run_dir / row["output"]
                if not source_path.is_file():
                    continue
                pixel_noise = row.get("pixel_noise") or row.get("noise") or "0"
                key = (
                    row.get("source_id", "source"),
                    row.get("source", row.get("source_id", "source")),
                    row.get("bucket", "unknown"),
                    row.get("pre_mp", "native"),
                    row.get("scale", "1"),
                    pixel_noise,
                )
                grouped.setdefault(key, []).append((latent, source_path))

    result: list[ComparisonGroup] = []
    for key, images in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][3], item[0][4], item[0][5])):
        source_id, source_name, bucket, pre_mp, scale, pixel_noise = key
        group_rel = Path(_safe_component(bucket)) / _safe_component(source_id) / _recipe_token(pre_mp, scale, pixel_noise)
        items: list[ComparisonItem] = []
        for latent, source_path in sorted(images, key=lambda item: item[0]):
            latent_token = fmt_value(latent)
            full_target = overlay_root / group_rel / f"latent-{latent_token}.png"
            full_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, full_target)

            thumb_target = thumb_root / group_rel / f"latent-{latent_token}.jpg"
            _write_thumb(source_path, thumb_target)
            items.append(ComparisonItem(latent=latent, full_path=full_target, thumb_path=thumb_target))

        result.append(
            ComparisonGroup(
                source_id=source_id,
                source_name=source_name,
                bucket=bucket,
                pre_mp=pre_mp,
                scale=scale,
                pixel_noise=pixel_noise,
                group_rel=group_rel,
                items=tuple(items),
            )
        )
    return result


def write_full_comparison_report(root: Path, runs: Sequence[tuple[float, Path]]) -> list[ComparisonGroup]:
    groups = build_full_comparison_groups(root, runs)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SeedVR2 latent-noise comparison</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;padding:0 18px;color:#111}.meta{color:#555}.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}.row{display:flex;gap:14px;overflow-x:auto;padding-bottom:8px}.variant{flex:0 0 300px}.variant img{width:300px;height:340px;object-fit:contain;background:#fafafa;border:1px solid #ddd}.label{font-weight:650;margin:0 0 6px}.path{font-size:.9em;color:#555;word-break:break-all}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>",
        "<h1>SeedVR2 latent-noise comparison</h1>",
        "<p class='meta'>Rows are grouped by source image and exact restoration recipe. Only Numz latent noise changes within each row; Numz input noise is fixed at 0.</p>",
        "<p class='meta'>Click any preview for the full-resolution output. Each <code>overlay-images/...</code> leaf directory contains only the latent variants of that exact image/recipe for Butterfly overlay or flip comparison.</p>",
        "<h2>Per-latent reports</h2><ul>",
    ]
    for latent, run_dir in runs:
        rel = run_dir.relative_to(root) / "index.html"
        parts.append(f"<li>latent={latent:g}: <a href='{html.escape(rel.as_posix())}'>report</a></li>")
    parts.append("</ul>")

    if not groups:
        parts.append("<p><strong>No completed full-image outputs were found in the child result CSVs.</strong></p>")
    for group in groups:
        parts.append("<div class='card'>")
        parts.append(f"<h2>{html.escape(group.source_name)}</h2>")
        parts.append(
            f"<p class='meta'>{html.escape(group.bucket)} · pre={html.escape(group.pre_mp)} · "
            f"scale={html.escape(group.scale)}x · pixel-noise={html.escape(group.pixel_noise)}</p>"
        )
        parts.append(f"<p class='path'>Butterfly: <code>overlay-images/{html.escape(group.group_rel.as_posix())}/</code></p><div class='row'>")
        for item in group.items:
            full_rel = item.full_path.relative_to(root).as_posix()
            thumb_rel = item.thumb_path.relative_to(root).as_posix()
            parts.append(
                f"<div class='variant'><div class='label'>latent={item.latent:g}</div>"
                f"<a href='{html.escape(full_rel)}'><img src='{html.escape(thumb_rel)}' loading='lazy' alt='latent {item.latent:g}'></a></div>"
            )
        parts.append("</div></div>")

    parts.append("</body></html>")
    (root / "index.html").write_text("".join(parts), encoding="utf-8")
    return groups
