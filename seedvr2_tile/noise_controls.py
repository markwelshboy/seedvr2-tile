from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Sequence

from . import backend

_LATENT_ENV = "SEEDVR2_LATENT_NOISE_SCALE"
_INPUT_ENV = "SEEDVR2_INPUT_NOISE_SCALE"
_ORIGINAL_BUILD_COMMAND = backend.build_command
_HOOK_INSTALLED = False


def parse_values(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        number = float(token)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"noise scale must be in [0, 1], got {number}")
        values.append(number)
    if not values:
        raise ValueError("at least one noise scale is required")
    return tuple(values)


def fmt_value(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return text.replace(".", "p")


def install_backend_noise_hook() -> None:
    """Append Numz-native noise flags to every standalone backend invocation.

    The existing spatial tile/probe engines call backend.run_group(), whose
    function globals resolve backend.build_command at runtime. Replacing that
    one command builder therefore covers direct tile runs, probe runs and the
    full-image sweep without duplicating inference logic.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return

    def build_command(*args, **kwargs):
        cmd = _ORIGINAL_BUILD_COMMAND(*args, **kwargs)
        input_noise = os.environ.get(_INPUT_ENV)
        latent_noise = os.environ.get(_LATENT_ENV)
        if input_noise is not None:
            cmd.extend(["--input_noise_scale", input_noise])
        if latent_noise is not None:
            cmd.extend(["--latent_noise_scale", latent_noise])
        return cmd

    backend.build_command = build_command
    _HOOK_INSTALLED = True


@contextmanager
def upstream_noise(*, latent: float, input_noise: float = 0.0) -> Iterator[None]:
    if not 0.0 <= latent <= 1.0:
        raise ValueError("latent noise scale must be in [0, 1]")
    if not 0.0 <= input_noise <= 1.0:
        raise ValueError("input noise scale must be in [0, 1]")
    install_backend_noise_hook()
    old_latent = os.environ.get(_LATENT_ENV)
    old_input = os.environ.get(_INPUT_ENV)
    os.environ[_LATENT_ENV] = f"{latent:.9g}"
    os.environ[_INPUT_ENV] = f"{input_noise:.9g}"
    try:
        yield
    finally:
        if old_latent is None:
            os.environ.pop(_LATENT_ENV, None)
        else:
            os.environ[_LATENT_ENV] = old_latent
        if old_input is None:
            os.environ.pop(_INPUT_ENV, None)
        else:
            os.environ[_INPUT_ENV] = old_input


def extract_value_option(
    argv: Sequence[str],
    names: set[str],
    *,
    default: str | None = None,
) -> tuple[list[str], str | None]:
    """Remove one value-taking option from argv, supporting --x value/--x=value."""
    out: list[str] = []
    found: str | None = None
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = next((name for name in names if token == name or token.startswith(name + "=")), None)
        if matched is None:
            out.append(token)
            i += 1
            continue
        if found is not None:
            raise SystemExit(f"option specified more than once: {matched}")
        if token == matched:
            if i + 1 >= len(argv):
                raise SystemExit(f"{matched} requires a value")
            found = argv[i + 1]
            i += 2
        else:
            found = token.split("=", 1)[1]
            i += 1
    return out, found if found is not None else default


def replace_option(argv: Sequence[str], old_names: set[str], new_name: str) -> list[str]:
    """Rename a value-taking option while preserving its value syntax."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = next((name for name in old_names if token == name or token.startswith(name + "=")), None)
        if matched is None:
            out.append(token)
            i += 1
            continue
        if token == matched:
            if i + 1 >= len(argv):
                raise SystemExit(f"{matched} requires a value")
            out.extend([new_name, argv[i + 1]])
            i += 2
        else:
            out.append(new_name + "=" + token.split("=", 1)[1])
            i += 1
    return out
