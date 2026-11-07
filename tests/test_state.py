import pytest

from crbot.catalog import key, load_catalog
from crbot.state.plays import ElixirEstimate, PlayDetector, elixir_generated
from crbot.state.towers import TowerTracker
from crbot.state.tracker import Tracker
from crbot.vision.hud import BarReading
from crbot.vision.types import Detection

CATALOG = load_catalog()


def det(name, x, y, side=1, conf=0.9, side_conf=0.6):
    unit = CATALOG.unit(name)
    return Detection(name=name, key=key(name), conf=conf, box=(0, 0, 1, 1), tile=(x, y), side=side,
                     side_conf=side_conf, side_method="test",
                     building=bool(unit and unit.building), flying=bool(unit and unit.flying))


# ----------------------------------------------------------------- tracker

def test_tracker_keeps_identity_and_velocity():
    tracker = Tracker()
    ids = set()
    for i in range(6):
        live, new, _ = tracker.update([det("hog_rider", 3.5, 10 + 0.2 * i)], t=0.1 * i)
        ids |= {tr.id for tr in live}
    assert len(ids) == 1
    (track,) = live
    assert track.vy == pytest.approx(2.0, abs=0.3)


def test_tracker_confirms_after_two_hits():
    tracker = Tracker()
    live, new, _ = tracker.update([det("knight", 5, 5)], t=0.0)
    assert live == [] and new == []
    live, new, _ = tracker.update([det("knight", 5, 5.1)], t=0.1)
    assert len(live) == 1 and len(new) == 1


def test_tracker_does_not_merge_different_units():
    tracker = Tracker()
    for i in range(3):
        live, _, _ = tracker.update([det("knight", 3, 10), det("knight", 14, 10)], t=0.1 * i)
    assert len({tr.id for tr in live}) == 2


def test_lost_unit_is_reacquired_with_its_side():
    tracker = Tracker()
    for i in range(3):
        tracker.update([det("ice_golem", 14.8, 17 - 0.1 * i, side=0, side_conf=1.0)], t=0.1 * i)
    (first,) = [tr for tr in tracker.tracks.values()]
    # hidden for 2 s (longer than max_age), reappears across the river with
    # a wrong per-frame side guess
    tracker.update([], t=1.0)
    tracker.update([], t=2.1)
    live, new, _ = tracker.update([det("ice_golem", 14.8, 14.0, side=1, side_conf=0.5)], t=2.3)
    assert new == []
    assert [tr.id for tr in live] == [first.id]
    assert live[0].side == 0


def test_side_hint_applies_to_new_tracks():
    tracker = Tracker()
    hint = lambda d: 0          # noqa: E731
    for i in range(2):
        live, _, _ = tracker.update([det("hog_rider", 3, 15, side=1, side_conf=0.5)], t=0.1 * i, side_hint=hint)
    assert live[0].side == 0


# ------------------------------------------------------------------ plays

def confirm(tracker, detections, t0, frames=2):
    new_all = []
    for i in range(frames):
        live, new, _ = tracker.update(detections, t=t0 + 0.1 * i)
        new_all += new
    return live, new_all


def test_skeleton_army_is_one_play():
    tracker, plays = Tracker(), PlayDetector(CATALOG)
    skeletons = [det("skeleton", 8 + 0.3 * (i % 5), 10 + 0.3 * (i // 5)) for i in range(15)]
    live, new = confirm(tracker, skeletons, 10.0)
    assert plays.update(new, live, 10.1) == []         # still collecting units
    done = plays.update([], live, 11.2)
    assert len(done) == 1
    assert done[0].card == "skeleton_army" and done[0].count == 15 and done[0].cost == 3


def test_single_skeleton_group_resolves_to_skeletons():
    tracker, plays = Tracker(), PlayDetector(CATALOG)
    live, new = confirm(tracker, [det("skeleton", 8, 10), det("skeleton", 8.5, 10), det("skeleton", 8, 10.5)], 5.0)
    plays.update(new, live, 5.1)
    (play,) = plays.update([], live, 6.5)
    assert play.card == "skeletons"


def test_spawned_units_are_not_plays():
    tracker, plays = Tracker(), PlayDetector(CATALOG)
    live, new = confirm(tracker, [det("tombstone", 9, 12)], 1.0)
    plays.update(new, live, 1.1)
    live, new = confirm(tracker, [det("tombstone", 9, 12), det("skeleton", 9, 13)], 5.0)
    plays.update(new, live, 5.1)
    plays.flush()
    assert [p.card for p in plays.plays] == ["tombstone"]


def test_enemy_troop_deep_in_our_half_is_not_a_play():
    tracker, plays = Tracker(), PlayDetector(CATALOG)
    live, new = confirm(tracker, [det("knight", 5, 26, side=1)], 1.0)
    plays.update(new, live, 1.1)
    assert plays.flush() == []


def test_spells_may_land_anywhere():
    tracker, plays = Tracker(), PlayDetector(CATALOG)
    live, new = confirm(tracker, [det("fireball", 3.5, 24.5, side=1)], 1.0)
    plays.update(new, live, 1.1)
    assert [p.card for p in plays.flush()] == ["fireball"]


def test_cycle_tracking():
    plays = PlayDetector(CATALOG)
    deck = plays.decks[1]
    for card in ["hog_rider", "cannon", "fireball", "the_log", "ice_golem"]:
        deck.observe(card)
    assert set(deck.unavailable()) == {"cannon", "fireball", "the_log", "ice_golem"}
    assert "hog_rider" not in deck.unavailable()          # four played since


def test_elixir_regen_phases():
    assert elixir_generated(0, 28) == pytest.approx(10.0)
    assert elixir_generated(110, 130) == pytest.approx(10 / 2.8 + 10 / 1.4)
    estimate = ElixirEstimate()
    estimate.advance(0.0)
    estimate.charge(4, 0.0)
    assert estimate.value == pytest.approx(1.0)
    assert estimate.advance(5.6) == pytest.approx(3.0)
    assert estimate.advance(100.0) == 10.0


# ----------------------------------------------------------------- towers

def test_tower_hp_ignores_occlusion_and_noise():
    towers = TowerTracker()
    for value in [1.0, 1.0, 0.8, 0.8, 0.8, 0.1, 0.8, 0.8]:
        towers.update({"ally_left": BarReading(value, 0.95)})
    towers.update({"ally_left": BarReading(None, 0.6)})       # partly covered: skipped
    assert towers.towers["ally_left"].hp == pytest.approx(0.8)
    assert not towers.towers["ally_left"].destroyed


def test_tower_destroyed_when_bar_gone_after_low_hp():
    towers = TowerTracker(destroyed_after=3)
    for _ in range(5):
        towers.update({"enemy_left": BarReading(0.2, 0.95)})
    for _ in range(3):
        towers.update({"enemy_left": BarReading(None, 0.0)})
    assert towers.towers["enemy_left"].destroyed


def test_healthy_tower_is_not_destroyed_by_missing_bar():
    towers = TowerTracker(destroyed_after=3)
    for _ in range(10):
        towers.update({"enemy_left": BarReading(None, 0.0)})
    assert not towers.towers["enemy_left"].destroyed
