from PIL import Image

from seedvr2_tile.tiling import make_tiles


def test_tiles_are_fixed_size_and_cover_partial_edges():
    image = Image.new("RGB", (1500, 1100), "white")
    tiles = make_tiles(
        image,
        image_index=0,
        tile_width=1024,
        tile_height=1024,
        padding=64,
        strategy="linear",
    )
    assert len(tiles) == 4
    for spec, tile in tiles:
        assert spec.process_size == (1152, 1152)
        assert tile.size == (1152, 1152)
    assert tiles[-1][0].core_box == (1024, 1024, 1500, 1100)


def test_chess_changes_order_not_coverage():
    image = Image.new("RGB", (2048, 2048), "white")
    linear = make_tiles(image, image_index=0, tile_width=512, tile_height=512, padding=32, strategy="linear")
    chess = make_tiles(image, image_index=0, tile_width=512, tile_height=512, padding=32, strategy="chess")
    assert {x[0].core_box for x in linear} == {x[0].core_box for x in chess}
    assert [x[0].core_box for x in linear] != [x[0].core_box for x in chess]
