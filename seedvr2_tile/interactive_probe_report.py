from __future__ import annotations

import csv
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import noise_sweep as _noise
from .sweep import _fmt_float

_META_MODE = "latent-noise-meta-v1"
_INTERACTIVE_MARKER = 'data-seedvr2-interactive="1"'


def _read_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pre_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"native", "none", "off"}:
        return None
    return float(value)


def _pre_token(value: float | None) -> str:
    return "native" if value is None else f"{_fmt_float(value)}mp"


def _same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= 1e-9


def _index_of_pre(values: Sequence[object], target: float | None) -> int | None:
    for index, raw in enumerate(values, start=1):
        if _same_number(_pre_value(raw), target):
            return index
    return None


def _index_of_float(values: Sequence[object], target: float) -> int | None:
    for index, raw in enumerate(values, start=1):
        if abs(float(raw) - target) <= 1e-9:
            return index
    return None


def _recipe_key(recipe: dict[str, Any]) -> str:
    return json.dumps(recipe, sort_keys=True, separators=(",", ":"))


def _recipe_label(recipe: dict[str, Any]) -> str:
    pre = recipe["pre_megapixels"]
    pre_text = "native" if pre is None else f"{pre:g} MP"
    return (
        f"pre={pre_text} · {recipe['scale']:g}× · px={recipe['pixel_noise']:g} · "
        f"latent={recipe['latent_noise_scale']:g}"
    )


def _run_roots(root: Path) -> list[tuple[float, Path]]:
    manifest = _read_manifest(root)
    if not manifest:
        return []
    if manifest.get("mode") == _META_MODE and manifest.get("kind") == "probe":
        runs: list[tuple[float, Path]] = []
        for item in manifest.get("runs", []):
            path = item.get("path")
            if path:
                runs.append((float(item.get("latent_noise_scale", 0.0)), root / str(path)))
        return runs
    settings = manifest.get("settings") or {}
    return [(float(settings.get("latent_noise_scale", 0.0) or 0.0), root)]


def _common_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    settings = manifest.get("settings") or {}
    keys = (
        "seed",
        "tile",
        "overlap",
        "tile_upscale_resolution",
        "strategy",
        "attention_mode",
        "color_correction",
        "pre_resample",
        "fbcnn",
        "jpeg_quality",
    )
    result = {key: settings[key] for key in keys if key in settings}
    result["model"] = manifest.get("model", "3b")
    result["input_noise_scale"] = float(settings.get("input_noise_scale", 0.0) or 0.0)
    return result


def _crop_path(
    run_root: Path,
    *,
    settings: dict[str, Any],
    row: dict[str, str],
    pre: float | None,
    scale: float,
    pixel_noise: float,
) -> Path | None:
    bucket = row["bucket"]
    pre_values = (settings.get("pre_by_bucket") or {}).get(bucket) or []
    scales = settings.get("scales") or []
    row_index = _index_of_pre(pre_values, pre)
    col_index = _index_of_float(scales, scale)
    if row_index is None or col_index is None:
        return None
    filename = (
        f"{row_index:02d}_pre-{_pre_token(pre)}__"
        f"{col_index:02d}-scale-{_fmt_float(scale)}x.png"
    )
    return (
        run_root
        / "reports"
        / bucket
        / row["source_id"]
        / row["probe_id"]
        / f"noise-{_fmt_float(pixel_noise)}"
        / "crops"
        / filename
    )


def _build_catalog(root: Path) -> dict[str, Any] | None:
    runs = _run_roots(root)
    if not runs:
        return None

    sources: dict[str, dict[str, Any]] = {}
    defaults: dict[str, Any] | None = None
    run_count = 0

    for declared_latent, run_root in runs:
        manifest = _read_manifest(run_root)
        if not manifest:
            continue
        settings = manifest.get("settings") or {}
        latent = float(settings.get("latent_noise_scale", declared_latent) or 0.0)
        input_noise = float(settings.get("input_noise_scale", 0.0) or 0.0)
        if defaults is None:
            defaults = _common_settings(manifest)

        source_meta = {
            str(item.get("source_id")): item
            for item in manifest.get("sources", [])
            if item.get("selected") and item.get("source_id")
        }
        probe_meta: dict[tuple[str, str], dict[str, Any]] = {}
        for source_id, item in source_meta.items():
            for probe in item.get("probes", []):
                probe_meta[(source_id, str(probe.get("probe_id")))] = probe

        results_path = run_root / "results.csv"
        if not results_path.is_file():
            continue
        run_count += 1
        with results_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            source_id = row.get("source_id", "")
            probe_id = row.get("probe_id", "")
            if not source_id or not probe_id or source_id not in source_meta:
                continue
            pre = _pre_value(row.get("pre_mp"))
            scale = float(row.get("scale", 0.0) or 0.0)
            pixel = float(row.get("pixel_noise") or row.get("noise") or 0.0)
            row_latent = float(row.get("latent_noise_scale") or latent)
            row_input = float(row.get("input_noise_scale") or input_noise)
            recipe = {
                "pre_megapixels": pre,
                "scale": scale,
                "pixel_noise": pixel,
                "input_noise_scale": row_input,
                "latent_noise_scale": row_latent,
            }
            key = _recipe_key(recipe)
            crop = _crop_path(
                run_root,
                settings=settings,
                row=row,
                pre=pre,
                scale=scale,
                pixel_noise=pixel,
            )
            if crop is None or not crop.is_file():
                output_scene = row.get("output_scene")
                crop = run_root / output_scene if output_scene else None
            if crop is None or not crop.is_file():
                continue

            meta = source_meta[source_id]
            source = sources.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "source": str(meta.get("source", source_id)),
                    "bucket": str(meta.get("bucket", row.get("bucket", ""))),
                    "width": int(meta.get("width", 0) or 0),
                    "height": int(meta.get("height", 0) or 0),
                    "megapixels": float(meta.get("megapixels", 0.0) or 0.0),
                    "recipes": {},
                    "probes": {},
                    "probe_order": [],
                },
            )
            source["recipes"][key] = recipe
            probe = source["probes"].setdefault(
                probe_id,
                {
                    "probe_id": probe_id,
                    "label": str(probe_meta.get((source_id, probe_id), {}).get("label", row.get("probe_label", probe_id))),
                    "center_x": probe_meta.get((source_id, probe_id), {}).get("center_x"),
                    "center_y": probe_meta.get((source_id, probe_id), {}).get("center_y"),
                    "variants": {},
                },
            )
            if probe_id not in source["probe_order"]:
                source["probe_order"].append(probe_id)
            probe["variants"][key] = {
                "recipe_key": key,
                "recipe": recipe,
                "label": _recipe_label(recipe),
                "image": crop.relative_to(root).as_posix(),
            }

    if not sources:
        return None
    return {
        "format": "seedvr2-interactive-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "defaults": defaults or {},
        "run_count": run_count,
        "sources": sources,
    }


def _preserve_comparison_index(root: Path) -> None:
    source = root / "index.html"
    target = root / "comparison-index.html"
    if not source.is_file():
        return
    text = source.read_text(encoding="utf-8", errors="replace")
    if _INTERACTIVE_MARKER in text:
        return
    shutil.copy2(source, target)


def _write_interactive_html(root: Path, catalog: dict[str, Any]) -> None:
    _preserve_comparison_index(root)
    source_items = sorted(catalog["sources"].values(), key=lambda item: (item["bucket"], item["source"]))
    client_catalog = {
        source["source_id"]: {
            "source_id": source["source_id"],
            "source": source["source"],
            "bucket": source["bucket"],
            "width": source["width"],
            "height": source["height"],
            "megapixels": source["megapixels"],
            "recipes": source["recipes"],
        }
        for source in source_items
    }

    parts = [
        '<!doctype html><html data-seedvr2-interactive="1"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>SeedVR2 interactive comparison</title>',
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:0;background:#f6f7f8;color:#171717}",
        ".page{max-width:1700px;margin:0 auto;padding:22px}.toolbar{position:sticky;top:0;z-index:10;background:rgba(246,247,248,.96);backdrop-filter:blur(8px);padding:12px 0;border-bottom:1px solid #ddd;display:flex;gap:10px;align-items:center;flex-wrap:wrap}",
        "button{font:inherit;padding:8px 13px;border:1px solid #bbb;border-radius:7px;background:white;cursor:pointer}button.primary{background:#111;color:#fff;border-color:#111}button:disabled{opacity:.45;cursor:default}",
        ".meta{color:#666}.source{background:white;border:1px solid #ddd;border-radius:12px;margin:22px 0;padding:18px}.source-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap}.choice{font-weight:600}.probe{margin-top:20px}.variants{display:flex;gap:12px;overflow-x:auto;padding:4px 2px 12px}.variant{flex:0 0 260px;border:2px solid transparent;border-radius:9px;padding:7px;background:#fafafa;cursor:pointer}.variant.selected{border-color:#111;background:#fff}.variant img{display:block;width:246px;height:246px;object-fit:cover;border-radius:5px;background:#eee}.variant .label{font-size:13px;line-height:1.35;margin-top:7px}.variant input{margin-right:6px}.small{font-size:13px}.links{margin-left:auto}code{background:#eee;padding:2px 5px;border-radius:4px}",
        "</style></head><body><div class='page'>",
        "<h1>SeedVR2 interactive comparison</h1>",
        "<p class='meta'>Choose one recipe for each source image. The detail/dark/center selectors are synchronized: selecting a recipe in any probe selects that same recipe in every other probe for the source.</p>",
        "<div class='toolbar'><button class='primary' id='download' disabled>Download selected recipes</button><button id='clear-all'>Clear all</button><span id='count' class='meta'></span><span class='links'><a href='comparison-index.html'>Original comparison report</a></span></div>",
    ]

    for source in source_items:
        sid = html.escape(source["source_id"], quote=True)
        parts.append(f"<article class='source' data-source-card='{sid}'>")
        parts.append("<div class='source-head'><div>")
        parts.append(f"<h2>{html.escape(source['source'])}</h2>")
        parts.append(
            f"<div class='meta'>{source['width']}×{source['height']} · {source['megapixels']:.2f} MP · {html.escape(source['bucket'])}</div>"
        )
        parts.append("</div><div><div class='choice' data-choice-label='" + sid + "'>No recipe selected</div>")
        parts.append(f"<button class='small clear-one' data-clear-source='{sid}'>Clear image</button></div></div>")

        for probe_id in source["probe_order"]:
            probe = source["probes"][probe_id]
            parts.append(f"<section class='probe'><h3>{html.escape(probe['probe_id'])} — {html.escape(probe['label'])}</h3>")
            if probe.get("center_x") is not None and probe.get("center_y") is not None:
                parts.append(f"<div class='meta small'>normalized probe ({float(probe['center_x']):.3f}, {float(probe['center_y']):.3f})</div>")
            parts.append("<div class='variants'>")
            ordered_variants = sorted(
                probe["variants"].values(),
                key=lambda item: (
                    item["recipe"]["latent_noise_scale"],
                    item["recipe"]["pixel_noise"],
                    -1 if item["recipe"]["pre_megapixels"] is None else item["recipe"]["pre_megapixels"],
                    item["recipe"]["scale"],
                ),
            )
            for variant in ordered_variants:
                key = html.escape(variant["recipe_key"], quote=True)
                image = html.escape(variant["image"], quote=True)
                label = html.escape(variant["label"])
                name = html.escape(f"select-{source['source_id']}-{probe_id}", quote=True)
                parts.append(
                    f"<label class='variant' data-variant-source='{sid}' data-variant-key='{key}'>"
                    f"<img src='{image}' loading='lazy' alt='{label}'>"
                    f"<div class='label'><input type='radio' name='{name}' data-source-id='{sid}' data-recipe-key='{key}'>"
                    f"{label}</div></label>"
                )
            parts.append("</div></section>")
        parts.append("</article>")

    embedded = json.dumps({"defaults": catalog["defaults"], "sources": client_catalog}, separators=(",", ":")).replace("</", "<\\/")
    parts.extend(
        [
            "<script>",
            f"const CATALOG={embedded};",
            "const selected={};const storageKey='seedvr2-selection:'+location.pathname;",
            "function allInputs(){return Array.from(document.querySelectorAll('input[data-source-id]'));}",
            "function recipeLabel(r){const p=r.pre_megapixels===null?'native':r.pre_megapixels+' MP';return `pre=${p} · ${r.scale}× · px=${r.pixel_noise} · latent=${r.latent_noise_scale}`;}",
            "function updateUi(){let count=0;for(const sid of Object.keys(CATALOG.sources)){const key=selected[sid];if(key)count++;const label=document.querySelector(`[data-choice-label=\"${CSS.escape(sid)}\"]`);if(label)label.textContent=key?recipeLabel(CATALOG.sources[sid].recipes[key]):'No recipe selected';}document.getElementById('count').textContent=`${count}/${Object.keys(CATALOG.sources).length} images selected`;document.getElementById('download').disabled=count===0;try{localStorage.setItem(storageKey,JSON.stringify(selected));}catch(e){}}",
            "function applySelection(sid,key){if(!CATALOG.sources[sid]||!CATALOG.sources[sid].recipes[key])return;selected[sid]=key;for(const input of allInputs()){if(input.dataset.sourceId!==sid)continue;const on=input.dataset.recipeKey===key;input.checked=on;input.closest('.variant')?.classList.toggle('selected',on);}updateUi();}",
            "function clearSelection(sid){delete selected[sid];for(const input of allInputs()){if(input.dataset.sourceId!==sid)continue;input.checked=false;input.closest('.variant')?.classList.remove('selected');}updateUi();}",
            "for(const input of allInputs()){input.addEventListener('change',()=>{if(input.checked)applySelection(input.dataset.sourceId,input.dataset.recipeKey);});}",
            "for(const button of document.querySelectorAll('[data-clear-source]'))button.addEventListener('click',()=>clearSelection(button.dataset.clearSource));",
            "document.getElementById('clear-all').addEventListener('click',()=>{for(const sid of Object.keys(selected))clearSelection(sid);});",
            "document.getElementById('download').addEventListener('click',()=>{const items=[];for(const sid of Object.keys(CATALOG.sources)){const key=selected[sid];if(!key)continue;const source=CATALOG.sources[sid];items.push({source_id:sid,source:source.source,bucket:source.bucket,width:source.width,height:source.height,megapixels:source.megapixels,recipe:source.recipes[key]});}const doc={format:'seedvr2-selection-v1',created_at:new Date().toISOString(),source_report:location.href,complete:items.length===Object.keys(CATALOG.sources).length,defaults:CATALOG.defaults,selections:items};const blob=new Blob([JSON.stringify(doc,null,2)+'\\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='seedvr2-selections.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});",
            "try{const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');for(const [sid,key] of Object.entries(saved))applySelection(sid,key);}catch(e){}updateUi();",
            "</script></div></body></html>",
        ]
    )
    (root / "index.html").write_text("".join(parts), encoding="utf-8")


def write_interactive_report(root: Path) -> bool:
    root = root.expanduser().resolve()
    catalog = _build_catalog(root)
    if not catalog:
        print(f"Interactive report not generated: no completed probe crops found under {root}")
        return False
    _write_interactive_html(root, catalog)
    print(f"Interactive probe report: {root / 'index.html'}")
    print(f"Original comparison report: {root / 'comparison-index.html'}")
    return True


def probe_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_noise.probe_main(raw) or 0)
    if rc or not raw or "--help" in raw or "-h" in raw:
        return rc
    if len(raw) >= 2 and not raw[1].startswith("-"):
        write_interactive_report(Path(raw[1]))
    return rc


def report_main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    rc = int(_noise.report_main(raw) or 0)
    if rc or not raw or "--help" in raw or "-h" in raw:
        return rc
    if not raw[0].startswith("-"):
        write_interactive_report(Path(raw[0]))
    return rc


if __name__ == "__main__":
    raise SystemExit(probe_main())
