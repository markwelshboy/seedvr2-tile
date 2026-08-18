import json
from pathlib import Path

import pytest

from seedvr2_tile.config import load_config
from seedvr2_tile.fbcnn import normalize_quality


def test_fbcnn_quality_accepts_auto_and_absolute_qf():
    assert normalize_quality("auto") == "auto"
    assert normalize_quality("95") == 95
    assert normalize_quality(5) == 5
    with pytest.raises(ValueError):
        normalize_quality(0)
    with pytest.raises(ValueError):
        normalize_quality(101)


def test_nested_fbcnn_config(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "preprocess": {
            "fbcnn": {
                "enabled": True,
                "quality": 92,
                "device": "cuda:0"
            },
            "megapixels": 1.0,
            "noise": 0.01
        }
    }))
    cfg = load_config(path)
    assert cfg["fbcnn_enabled"] is True
    assert cfg["fbcnn_quality"] == 92
    assert cfg["fbcnn_device"] == "cuda:0"
    assert cfg["pre_megapixels"] == 1.0
    assert cfg["noise"] == 0.01
