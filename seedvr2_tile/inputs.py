from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceItem:
    path: Path
    relative: Path


def _files_in_directory(path: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return [
        p.resolve()
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]


def _expand_one(spec: str, recursive: bool, extensions: set[str]) -> list[Path]:
    expanded = os.path.expandvars(os.path.expanduser(spec))
    if glob.has_magic(expanded):
        matches = [Path(x) for x in glob.glob(expanded, recursive=True)]
        if not matches:
            raise ValueError(f"input glob matched no files: {spec}")
        files: list[Path] = []
        for match in matches:
            if match.is_dir():
                files.extend(_files_in_directory(match, recursive, extensions))
            elif match.is_file() and match.suffix.lower() in extensions:
                files.append(match.resolve())
        if not files:
            raise ValueError(f"input glob matched no supported images: {spec}")
        return files

    path = Path(expanded).resolve()
    if path.is_file():
        if path.suffix.lower() not in extensions:
            raise ValueError(f"unsupported image extension: {path.suffix}")
        return [path]
    if path.is_dir():
        files = _files_in_directory(path, recursive, extensions)
        if not files:
            raise ValueError(f"no supported images found in {path}")
        return files
    raise FileNotFoundError(path)


def discover_inputs(specs: Iterable[str], *, recursive: bool, extensions: set[str]) -> list[SourceItem]:
    """Expand directories, quoted globs, and shell-expanded file lists.

    Duplicate matches are removed. A common parent is used to preserve relative
    directory structure when practical; if inputs have no useful common parent,
    basenames are used and collisions are rejected.
    """
    all_paths: list[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        for path in _expand_one(str(spec), recursive, extensions):
            if path not in seen:
                seen.add(path)
                all_paths.append(path)
    if not all_paths:
        raise ValueError("no input images specified")

    all_paths.sort()
    if len(all_paths) == 1:
        common_root = all_paths[0].parent
        use_common = True
    else:
        common_root = Path(os.path.commonpath([str(p.parent) for p in all_paths]))
        use_common = common_root != Path(common_root.anchor)

    items: list[SourceItem] = []
    relative_seen: set[Path] = set()
    for path in all_paths:
        relative = path.relative_to(common_root) if use_common else Path(path.name)
        if relative in relative_seen:
            raise ValueError(
                f"multiple inputs map to the same relative output path '{relative}'. "
                "Use a narrower glob/directory grouping."
            )
        relative_seen.add(relative)
        items.append(SourceItem(path=path, relative=relative))
    return items
