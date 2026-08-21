from __future__ import annotations

import gc
import importlib
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_FBCNN_ROOT = Path.home() / ".cache" / "seedvr2-tile" / "FBCNN"
FBCNN_URL = "https://github.com/jiaxi-jiang/FBCNN.git"
MODEL_NAME = "fbcnn_color.pth"
MODEL_URL = f"https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{MODEL_NAME}"


@dataclass(frozen=True)
class FBCNNOptions:
    enabled: bool = False
    quality: str | int = "auto"
    root: Path | None = None
    device: str = "auto"


def setup_fbcnn(root: Path = DEFAULT_FBCNN_ROOT, ref: str = "main") -> None:
    root = root.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("git"):
        raise RuntimeError("git is required to set up FBCNN")
    if (root / ".git").exists():
        subprocess.run(["git", "-C", str(root), "fetch", "--tags", "origin"], check=True)
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError(f"FBCNN setup target exists and is not a git checkout: {root}")
    else:
        subprocess.run(["git", "clone", FBCNN_URL, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", ref], check=True)
    if ref in {"main", "master"}:
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "origin", ref], check=True)


def resolve_fbcnn_root(explicit: Path | str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("FBCNN_ROOT"):
        candidates.append(Path(os.environ["FBCNN_ROOT"]).expanduser())
    candidates.append(DEFAULT_FBCNN_ROOT)
    for root in candidates:
        if (root / "models" / "network_fbcnn.py").is_file():
            return root.resolve()
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find the official FBCNN checkout. "
        f"Searched: {searched}. Run 'seedvr2-tile setup --fbcnn' or set FBCNN_ROOT."
    )


def normalize_quality(value: str | int) -> str | int:
    if isinstance(value, str):
        if value.lower() == "auto":
            return "auto"
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError("JPEG quality must be 'auto' or an integer from 1 to 100") from exc
    value = int(value)
    if not 1 <= value <= 100:
        raise ValueError("JPEG quality must be in the range 1..100")
    return value


_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def _get_model(root: Path, device_name: str):
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    key = (str(root), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = root / "model_zoo" / MODEL_NAME
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"FBCNN: downloading {MODEL_NAME} -> {model_path}", flush=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)

    # The official architecture imports `models.*`, so put only the official
    # checkout at the front of sys.path while importing it.
    root_str = str(root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    try:
        module = importlib.import_module("models.network_fbcnn")
    finally:
        if added:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass

    model = module.FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R")
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device)
    _MODEL_CACHE[key] = (model, device)
    return model, device


def restore_jpeg_artifacts(image: Image.Image, options: FBCNNOptions) -> tuple[Image.Image, float]:
    """Run official color FBCNN in blind or manually controlled QF mode.

    Returns the restored RGB image and FBCNN's predicted JPEG quality factor.
    Manual `quality` controls reconstruction but the returned value remains the
    network's prediction, matching the reference implementation.
    """
    if not options.enabled:
        raise ValueError("FBCNN is not enabled")
    quality = normalize_quality(options.quality)
    root = resolve_fbcnn_root(options.root)

    import torch

    model, device = _get_model(root, options.device)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    qf_input = None
    if quality != "auto":
        qf_input = torch.tensor([[1.0 - float(quality) / 100.0]], dtype=tensor.dtype, device=device)

    with torch.inference_mode():
        restored, qf_pred = model(tensor, qf_input) if qf_input is not None else model(tensor)

    predicted_quality = float((1.0 - qf_pred.detach()).clamp(0, 1).item() * 100.0)
    restored = restored.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    out = Image.fromarray(np.round(restored * 255.0).astype(np.uint8), mode="RGB")
    return out, predicted_quality


def release_fbcnn() -> None:
    """Release cached FBCNN models before launching SeedVR2."""
    if not _MODEL_CACHE:
        return
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
