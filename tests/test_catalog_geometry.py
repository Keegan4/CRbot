import re

import numpy as np
import pytest

from crbot.catalog import ASSETS, key, load_catalog
from crbot.geometry import (GRID_H, GRID_W, LAYOUT_9x16, ArenaGrid, cell_of, is_our_half,
                            layout_for)


def test_key_normalises_both_vocabularies():
    assert key("hog-rider") == key("hog_rider") == "hogrider"
    assert key("mini-pekka") == key("minipekka")
    assert key("skeleton-evolution") == "skeleton"
    assert key("archers_ev1") == "archers"
    # UI decoration must not turn into the unit it decorates
    assert key("ice-spirit-evolution-symbol") != key("ice-spirit")


def test_every_detector_class_is_known():
    catalog = load_catalog()
    yaml = (ASSETS / "katacr_classes.yaml").read_text()
    names = re.findall(r"^\s+\d+:\s*(\S+)", yaml, re.M)
    assert len(names) == 201
    unresolved = [n for n in names if catalog.is_unit_class(n) and not catalog.is_tower_class(n)
                  and catalog.unit(n) is None and catalog.card(n) is None]
    assert unresolved == []


def test_cards_for_units_and_spells():
    catalog = load_catalog()
    assert {c.name for c in catalog.cards_for("barbarian")} == {"barbarians", "barbarian_barrel", "barbarian_hut"}
    assert [c.name for c in catalog.cards_for("fireball")] == ["fireball"]
    assert catalog.cost("hog-rider") == 4


def test_units_of_includes_spawns():
    units = load_catalog().units_of(["tombstone", "golem"])
    assert {"tombstone", "skeleton", "golem", "golemite"} <= units


def test_tile_pixel_round_trip():
    grid = ArenaGrid.from_layout(LAYOUT_9x16, 720, 1280)
    for tx, ty in [(0, 0), (9, 16), (17.5, 31.5), (3.5, 24.5)]:
        px, py = grid.tile_to_px(tx, ty)
        assert grid.px_to_tile(px, py) == pytest.approx((tx, ty))
    # tile (0,0) corner and tile size from the reference measurements
    assert grid.tile_to_px(0, 0) == pytest.approx((52, 100.8))
    assert grid.tile_size_px == pytest.approx((34, 27.6))


def test_layout_scales_with_resolution():
    small = ArenaGrid.from_layout(LAYOUT_9x16, 720, 1280)
    big = ArenaGrid.from_layout(LAYOUT_9x16, 1080, 1920)
    px, py = small.cell_center_px(4, 20)
    bx, by = big.cell_center_px(4, 20)
    assert (bx, by) == pytest.approx((px * 1.5, py * 1.5))


def test_layout_for_rejects_unknown_aspect():
    assert layout_for(720, 1280).name == "9x16"
    assert layout_for(1080, 1920).name == "9x16"
    with pytest.raises(ValueError):
        layout_for(1280, 720)


def test_affine_fit_recovers_grid():
    truth = ArenaGrid(np.array([[30.0, 0.5, 40.0], [0.0, 25.0, 90.0]]))
    tiles = [(3.5, 7.5), (14.5, 7.5), (9.0, 3.0), (3.5, 24.5), (14.5, 24.5), (9.0, 29.0)]
    pixels = [truth.tile_to_px(*t) for t in tiles]
    grid, rms = ArenaGrid.fit(tiles, pixels)
    assert rms < 1e-6
    assert grid.A == pytest.approx(truth.A)


def test_cells_and_halves():
    assert cell_of(-3, 40) == (0, GRID_H - 1)
    assert cell_of(17.99, 0.2) == (GRID_W - 1, 0)
    assert is_our_half(16.0) and not is_our_half(15.9)
