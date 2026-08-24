from pathlib import Path

from seedvr2_tile import backend
from seedvr2_tile.noise_controls import install_backend_noise_hook, upstream_noise
from seedvr2_tile.noise_sweep import _prepare_args


def test_backend_hook_passes_numz_input_and_latent_noise_flags(tmp_path: Path):
    install_backend_noise_hook()
    options = backend.BackendOptions(seedvr2_root=tmp_path)
    with upstream_noise(latent=0.03, input_noise=0.0):
        cmd = backend.build_command(
            options,
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            resolution=2048,
        )
    assert cmd[cmd.index("--input_noise_scale") + 1] == "0"
    assert cmd[cmd.index("--latent_noise_scale") + 1] == "0.03"


def test_backend_hook_leaves_command_unchanged_outside_noise_context(tmp_path: Path):
    install_backend_noise_hook()
    options = backend.BackendOptions(seedvr2_root=tmp_path)
    cmd = backend.build_command(
        options,
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        resolution=2048,
    )
    assert "--input_noise_scale" not in cmd
    assert "--latent_noise_scale" not in cmd


def test_sweep_public_pixel_name_maps_to_legacy_rgb_noise_axis():
    argv, latent = _prepare_args(
        [
            "images",
            "results",
            "--pixel-noise-values",
            "0,0.005",
            "--latent-noise-values",
            "0,0.015,0.03,0.05",
        ]
    )
    assert argv == ["images", "results", "--noise-values", "0,0.005"]
    assert latent == (0.0, 0.015, 0.03, 0.05)


def test_legacy_noise_values_remains_supported_as_pixel_noise():
    argv, latent = _prepare_args(["images", "results", "--noise-values", "0.01"])
    assert argv == ["images", "results", "--noise-values", "0.01"]
    assert latent == (0.0,)
