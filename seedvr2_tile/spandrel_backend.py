from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str | None = None
    filename: str | None = None
    expected_scale: int | None = None
    expected_architecture: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "realesrgan-x2plus": ModelSpec(
        name="realesrgan-x2plus",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        filename="RealESRGAN_x2plus.pth",
        expected_scale=2,
        expected_architecture="ESRGAN",
    ),
    # SPAN itself is supported natively by Spandrel. The official authors publish
    # checkpoints as a Google Drive archive rather than a stable direct .pth URL,
    # so this alias deliberately requires --model-file or --model-url.
    "span-x2": ModelSpec(
        name="span-x2",
        expected_scale=2,
        expected_architecture="SPAN",
    ),
}


@dataclass(frozen=True)
class LoadedModelInfo:
    name: str
    path: Path
    architecture: str
    scale: int
    device: str
    dtype: str


def _model_cache_dir() -> Path:
    root = os.environ.get("UPSCALER_MODEL_DIR")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "upscale-models"


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading model: {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(target)
    return target


def resolve_model_path(
    model_name: str,
    *,
    model_file: str | None = None,
    model_url: str | None = None,
) -> tuple[Path, ModelSpec | None]:
    spec = MODEL_SPECS.get(model_name)

    if model_file:
        path = Path(model_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        return path, spec

    url = model_url or (spec.url if spec else None)
    if not url:
        known = ", ".join(sorted(MODEL_SPECS))
        if spec is not None:
            raise ValueError(
                f"{model_name} requires --model-file or --model-url for a compatible checkpoint"
            )
        raise ValueError(
            f"unknown model alias {model_name!r}; use --model-file/--model-url "
            f"or one of: {known}"
        )

    parsed = urllib.parse.urlparse(url)
    filename = (
        (spec.filename if spec else None)
        or Path(parsed.path).name
        or f"{model_name}.pth"
    )
    target = _model_cache_dir() / model_name / filename
    if not target.is_file():
        _download(url, target)
    return target, spec


class SpandrelUpscaler:
    def __init__(
        self,
        model_name: str,
        *,
        model_file: str | None = None,
        model_url: str | None = None,
        device: str = "cuda",
        fp16: bool = True,
        tile: int = 512,
        overlap: int = 32,
    ) -> None:
        try:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise RuntimeError(
                "generic upscale backend requires torch and spandrel; "
                "install with `pip install spandrel` in a torch-enabled environment"
            ) from exc

        self.torch = torch
        self.path, spec = resolve_model_path(
            model_name, model_file=model_file, model_url=model_url
        )
        descriptor = ModelLoader().load_from_file(str(self.path))
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError(f"{self.path.name} is not an image-to-image model")

        scale = int(round(float(descriptor.scale)))
        arch_obj = getattr(descriptor, "architecture", None)
        architecture = str(getattr(arch_obj, "id", arch_obj or "unknown"))

        if spec and spec.expected_scale is not None and scale != spec.expected_scale:
            raise ValueError(
                f"{model_name} expected a {spec.expected_scale}x checkpoint, "
                f"but Spandrel detected {scale}x"
            )
        if (
            spec
            and spec.expected_architecture
            and architecture.lower() != spec.expected_architecture.lower()
        ):
            raise ValueError(
                f"{model_name} expected architecture {spec.expected_architecture}, "
                f"but Spandrel detected {architecture}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        self.device = torch.device(device)
        self.scale = scale
        self.tile = int(tile)
        self.overlap = int(overlap)
        if self.tile < 0:
            raise ValueError("tile must be >= 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.tile and self.overlap * 2 >= self.tile:
            raise ValueError("overlap must be less than half the tile size")

        supports_half = bool(getattr(descriptor, "supports_half", True))
        self.use_fp16 = bool(fp16 and self.device.type == "cuda" and supports_half)
        descriptor.to(self.device)
        descriptor.eval()
        if self.use_fp16:
            descriptor.half()
        self.model = descriptor
        self.info = LoadedModelInfo(
            name=model_name,
            path=self.path,
            architecture=architecture,
            scale=scale,
            device=str(self.device),
            dtype="float16" if self.use_fp16 else "float32",
        )

    def _run_tensor(self, tensor):
        with self.torch.inference_mode():
            output = self.model(tensor)
        # Spandrel already clamps image-model outputs to [0, 1] inside
        # inference_mode. Clone here to leave inference tensors at the backend
        # boundary before our tiling/stitch/output code performs normal tensor ops.
        return output.clone()

    def _run_tiled(self, tensor):
        torch = self.torch
        _, channels, height, width = tensor.shape
        scale = self.scale
        if not self.tile or (height <= self.tile and width <= self.tile):
            return self._run_tensor(tensor)

        output = torch.empty(
            (1, channels, height * scale, width * scale),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        core = self.tile
        context = self.overlap

        for y0 in range(0, height, core):
            y1 = min(height, y0 + core)
            ey0 = max(0, y0 - context)
            ey1 = min(height, y1 + context)
            for x0 in range(0, width, core):
                x1 = min(width, x0 + core)
                ex0 = max(0, x0 - context)
                ex1 = min(width, x1 + context)

                tile = tensor[:, :, ey0:ey1, ex0:ex1]
                tile_out = self._run_tensor(tile)

                cy0 = (y0 - ey0) * scale
                cy1 = cy0 + (y1 - y0) * scale
                cx0 = (x0 - ex0) * scale
                cx1 = cx0 + (x1 - x0) * scale

                output[
                    :,
                    :,
                    y0 * scale : y1 * scale,
                    x0 * scale : x1 * scale,
                ] = tile_out[:, :, cy0:cy1, cx0:cx1]
        return output

    def upscale(self, image: Image.Image) -> Image.Image:
        torch = self.torch
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )
        if self.use_fp16:
            tensor = tensor.half()
        output = self._run_tiled(tensor)
        out = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")
