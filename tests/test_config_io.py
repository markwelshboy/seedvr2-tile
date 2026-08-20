import json

from seedvr2_tile.config import load_config


def test_config_accepts_output_naming_and_input_glob_list(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "io": {
            "input": ["images/*.jpg", "more/*.png"],
            "output": "out",
            "output_mode": "on",
            "output_template": "#basename#_#model#_#scale#_#pre-megapixels#"
        }
    }))
    cfg = load_config(p)
    assert cfg["input"] == ["images/*.jpg", "more/*.png"]
    assert cfg["output_mode"] == "on"
    assert cfg["output_template"].endswith("#pre-megapixels#")
    assert cfg["output"].name == "out"
