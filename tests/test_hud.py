from pathlib import Path

import numpy as np
import pytest

from crbot.capture import ImageFolderSource, parse_raw_screencap
from crbot.geometry import LAYOUT_9x16
from crbot.vision.hud import CardReader, ScreenClassifier, read_elixir, read_towers

from .synth import make_frame

DECK = ["hog_rider", "musketeer", "cannon", "fireball", "the_log", "ice_golem", "ice_spirit", "skeletons"]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("elixir", [0.0, 1.0, 3.5, 6.3, 10.0])
def test_elixir_bar(elixir):
    frame = make_frame(elixir=elixir)
    assert read_elixir(frame, LAYOUT_9x16.elixir_bar) == pytest.approx(elixir, abs=0.15)


def test_elixir_at_1080p():
    frame = make_frame(1080, 1920, elixir=4.0)
    assert read_elixir(frame, LAYOUT_9x16.elixir_bar) == pytest.approx(4.0, abs=0.15)


def test_tower_bars():
    towers = {"enemy_left": 1.0, "enemy_right": 0.4, "ally_left": 0.75, "ally_right": 0.1}
    readings = read_towers(make_frame(towers=towers), LAYOUT_9x16)
    for name, value in towers.items():
        assert readings[name].fraction == pytest.approx(value, abs=0.05), name


def test_missing_tower_bar_is_none():
    readings = read_towers(make_frame(towers={"enemy_left": 1.0}), LAYOUT_9x16)
    assert readings["enemy_right"].fraction is None
    assert readings["enemy_right"].missing


def test_hand_from_reference_art():
    reader = CardReader(deck=DECK, user_dir=None)
    frame = make_frame(hand=["hog_rider", "musketeer", "fireball", "ice_spirit"], next_card="the_log",
                       greyed=[2])
    reading = reader.read(frame, LAYOUT_9x16)
    assert reading.cards == ["hog_rider", "musketeer", "fireball", "ice_spirit"]
    assert reading.playable == [True, True, False, True]
    assert reading.next_card == "the_log"


def test_empty_slot():
    reader = CardReader(deck=DECK, user_dir=None)
    frame = make_frame(hand=["hog_rider", "musketeer", "fireball", "ice_spirit"])
    x0, y0, x1, y1 = LAYOUT_9x16.hand[1].to_pixels(720, 1280)
    frame[y0:y1, x0:x1] = (160, 70, 10)            # the flat blue empty panel
    reading = reader.read(frame, LAYOUT_9x16)
    assert reading.cards[1] is None and reading.empty[1]
    assert reading.cards[0] == "hog_rider"


# Real frames from a live match (Hasty-CR's showcase recording, MIT), with
# another bot's overlay drawn on them. Ground truth was read off the images.
LIVE = {
    "live_0000.jpg": (["the_log", "hog_rider", "ice_golem", "cannon"], [True] * 4, "skeletons", 8.2),
    "live_0270.jpg": ([None, "musketeer", "skeletons", "cannon"], [False, False, True, False], None, 2.1),
    "live_0720.jpg": (["fireball", "musketeer", "ice_golem", "cannon"], [False, False, True, False], "hog_rider", 2.1),
}


@pytest.mark.parametrize("name", sorted(LIVE))
def test_live_frames(name):
    import cv2

    image = cv2.imread(str(FIXTURES / name))
    cards, playable, next_card, elixir = LIVE[name]
    reading = CardReader(deck=DECK, user_dir=None).read(image, LAYOUT_9x16)
    assert reading.cards == cards
    assert reading.playable == playable
    if next_card:                      # 0270's next card is hidden by the overlay
        assert reading.next_card == next_card
    assert read_elixir(image, LAYOUT_9x16.elixir_bar) == pytest.approx(elixir, abs=0.3)
    assert ScreenClassifier().classify(image) == "in_game"
    towers = read_towers(image, LAYOUT_9x16)
    assert towers["ally_right"].fraction == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("header", [12, 16])
def test_raw_screencap(header):
    rgba = np.zeros((4, 3, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 10, 20, 30, 255
    head = (3).to_bytes(4, "little") + (4).to_bytes(4, "little") + (1).to_bytes(4, "little")
    head += b"\0" * (header - 12)
    image = parse_raw_screencap(head + rgba.tobytes())
    assert image.shape == (4, 3, 3)
    assert image[0, 0].tolist() == [30, 20, 10]          # BGR


def test_image_folder_source(tmp_path):
    import cv2

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((16, 9, 3), i, np.uint8))
    source = ImageFolderSource(tmp_path, fps=2.0)
    frames = [source.read() for _ in range(4)]
    assert [f.t for f in frames[:3]] == [0.0, 0.5, 1.0]
    assert frames[3] is None
