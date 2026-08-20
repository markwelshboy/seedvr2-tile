from seedvr2_tile.cli import _build_parser, _resolve_run_io


def test_shell_expanded_positionals_use_last_path_as_output(tmp_path):
    parser = _build_parser()
    args = parser.parse_args(["run", "a.jpg", "b.jpg", str(tmp_path / "out")])
    inputs, output = _resolve_run_io(args)
    assert inputs == ["a.jpg", "b.jpg"]
    assert output == (tmp_path / "out").resolve()


def test_config_output_makes_all_positionals_inputs(tmp_path):
    parser = _build_parser(defaults={"output": tmp_path / "out"})
    args = parser.parse_args(["run", "a.jpg", "b.jpg"])
    inputs, output = _resolve_run_io(args)
    assert inputs == ["a.jpg", "b.jpg"]
    assert output == (tmp_path / "out").resolve()


def test_output_dir_disambiguates_all_positionals(tmp_path):
    parser = _build_parser()
    args = parser.parse_args(["run", "a.jpg", "b.jpg", "--output-dir", str(tmp_path / "out")])
    inputs, output = _resolve_run_io(args)
    assert inputs == ["a.jpg", "b.jpg"]
    assert output == (tmp_path / "out").resolve()


def test_explicit_3b_fp8_and_fp16_are_accepted():
    parser = _build_parser()
    a = parser.parse_args(["run", "a.jpg", "out", "--model", "3b-fp8"])
    b = parser.parse_args(["run", "a.jpg", "out", "--model", "3b-fp16"])
    assert a.dit_model == "3b-fp8"
    assert b.dit_model == "3b-fp16"
