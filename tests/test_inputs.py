from seedvr2_tile.inputs import discover_inputs

EXTS = {".png", ".jpg", ".jpeg"}


def test_quoted_glob_expands(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("x")
    items = discover_inputs([str(tmp_path / "*.jpg")], recursive=False, extensions=EXTS)
    assert [x.path.name for x in items] == ["a.jpg", "b.jpg"]
    assert [str(x.relative) for x in items] == ["a.jpg", "b.jpg"]


def test_shell_expanded_file_list_is_accepted(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    items = discover_inputs([str(a), str(b)], recursive=False, extensions=EXTS)
    assert {x.path.name for x in items} == {"a.png", "b.png"}


def test_directory_recursive_preserves_relative_structure(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "root.jpg").write_bytes(b"x")
    (nested / "child.jpg").write_bytes(b"x")
    items = discover_inputs([str(tmp_path)], recursive=True, extensions=EXTS)
    rel = {str(x.relative) for x in items}
    assert rel == {"root.jpg", "nested/child.jpg"}


def test_single_recursive_directory_keeps_top_level_relative_path(tmp_path):
    nested = tmp_path / "only" / "nested"
    nested.mkdir(parents=True)
    (nested / "child.jpg").write_bytes(b"x")
    items = discover_inputs([str(tmp_path)], recursive=True, extensions=EXTS)
    assert [str(x.relative) for x in items] == ["only/nested/child.jpg"]
