from seedvr2_tile.spandrel_backend import MODEL_SPECS


def test_span_x2_builtin_uses_2x_span_checkpoint():
    spec = MODEL_SPECS["span-x2"]
    assert spec.expected_scale == 2
    assert spec.expected_architecture == "SPAN"
    assert spec.filename == "2xNomosUni_span_multijpg_ldl.pth"
    assert spec.url is not None
    assert spec.url.endswith("/2xNomosUni_span_multijpg_ldl.pth")
