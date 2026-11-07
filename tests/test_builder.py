"""End to end: synthetic frames through GameStateBuilder with a scripted detector."""

import json

from crbot.capture import Frame
from crbot.geometry import LAYOUT_9x16, ArenaGrid
from crbot.state.builder import GameStateBuilder
from crbot.vision.detector import RawBox

from .synth import make_frame

DECK = ["hog_rider", "musketeer", "cannon", "fireball", "the_log", "ice_golem", "ice_spirit", "skeletons"]
GRID = ArenaGrid.from_layout(LAYOUT_9x16, 720, 1280)


def box_at(name, tx, ty, side=None, w=40, h=50):
    """A detector box whose bottom-centre stands on tile (tx, ty)."""
    px, py = GRID.tile_to_px(tx, ty)
    return RawBox(name, 0.9, (px - w / 2, py - h, px + w / 2, py), side)


class ScriptedDetector:
    def __init__(self):
        self.boxes = []

    def detect(self, image):
        return list(self.boxes)


def run(builder, frames):
    state = None
    for i, (t, image) in enumerate(frames):
        state = builder.process(Frame(image, t, i))
    return state


def test_end_to_end():
    detector = ScriptedDetector()
    builder = GameStateBuilder(detector=detector, deck=DECK)
    hand = ["hog_rider", "musketeer", "fireball", "ice_spirit"]
    towers = {"enemy_left": 1.0, "enemy_right": 1.0, "ally_left": 1.0, "ally_right": 1.0}

    # 2 s of an empty board
    frames = [(0.1 * i, make_frame(hand=hand, next_card="cannon", elixir=7.0, towers=towers)) for i in range(20)]
    state = run(builder, frames)
    assert state.in_battle and state.hand == hand and state.next_card == "cannon"
    assert abs(state.elixir - 7.0) < 0.2

    # We play the Hog Rider: it leaves the hand, elixir drops by 4, and a
    # hog appears at our bridge. The per-frame side guess says "enemy" (it is
    # in their half): the hand change must win.
    new_hand = ["cannon", "musketeer", "fireball", "ice_spirit"]
    detector.boxes = [box_at("hog_rider", 3.5, 15.0)]
    frames = [(2.0 + 0.1 * i, make_frame(hand=new_hand, next_card="the_log", elixir=3.0, towers=towers))
              for i in range(5)]
    state = run(builder, frames)
    hogs = [u for u in state.units if u.name == "hog_rider"]
    assert len(hogs) == 1 and hogs[0].side == 0
    ours = [p for p in state.plays if p.side == 0]
    assert [p.card for p in ours] == ["hog_rider"]
    assert ours[0].x is not None

    # The opponent drops a Knight at their bridge.
    detector.boxes = [box_at("hog_rider", 3.5, 14.0), box_at("knight", 14.5, 12.0)]
    frames = [(2.5 + 0.1 * i, make_frame(hand=new_hand, next_card="the_log", elixir=3.4, towers=towers))
              for i in range(15)]
    state = run(builder, frames)
    theirs = [p for p in state.plays if p.side == 1]
    assert [p.card for p in theirs] == ["knight"]
    assert state.opponent_deck == ["knight"]
    # 5 starting elixir + regen over ~2.6 s - 3 for the Knight
    assert 2.0 < state.enemy_elixir < 4.5

    payload = json.loads(state.to_json())
    assert payload["in_battle"] and payload["plays"]
    assert state.planes().shape == (6, 32, 18)


def test_not_in_battle_without_a_hand():
    builder = GameStateBuilder(deck=DECK)
    image = make_frame()                      # no cards drawn
    state = builder.process(Frame(image, 0.0, 0))
    assert not state.in_battle and state.units == []


def test_unit_inside_standing_tower_is_ignored():
    detector = ScriptedDetector()
    detector.boxes = [box_at("knight", 3.5, 7.5)]          # inside the enemy left princess tower
    builder = GameStateBuilder(detector=detector, deck=DECK)
    image = make_frame(hand=["hog_rider", "musketeer", "fireball", "ice_spirit"])
    run(builder, [(0.1 * i, image) for i in range(3)])
    assert builder.last_detections == []
