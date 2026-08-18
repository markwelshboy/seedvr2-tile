from pathlib import Path

from seedvr2_tile.backend import (
    BackendOptions,
    MODEL_ALIASES,
    build_command,
    build_model_preflight_command,
    resolve_model_name,
)


def test_model_aliases_resolve_and_exact_names_pass_through():
    assert resolve_model_name(None) == MODEL_ALIASES["3b"]
    assert resolve_model_name("3b") == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    assert resolve_model_name("7B-SHARP") == "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    assert resolve_model_name("my_custom.gguf") == "my_custom.gguf"


def test_preflight_uses_selected_model_and_model_dir():
    opts = BackendOptions(
        seedvr2_root=Path("/tmp/seedvr2"),
        dit_model=MODEL_ALIASES["3b"],
        model_dir="/models/seedvr2",
    )
    cmd = build_model_preflight_command(opts)
    assert MODEL_ALIASES["3b"] in cmd
    assert "/models/seedvr2" in cmd
    assert "download_weight" in cmd[2]


def test_inference_command_always_passes_resolved_model():
    opts = BackendOptions(seedvr2_root=Path("/tmp/seedvr2"), dit_model=MODEL_ALIASES["3b"])
    cmd = build_command(opts, input_dir=Path("in"), output_dir=Path("out"), resolution=2048)
    idx = cmd.index("--dit_model")
    assert cmd[idx + 1] == MODEL_ALIASES["3b"]
