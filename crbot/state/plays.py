"""Infer card plays from units appearing, and estimate the opponent's elixir.

The detector never sees a card being *played*, only units appearing on the
board. A play is inferred when a new unit is confirmed:

* **Spawns are not plays.** A Tombstone makes Skeletons, a Golem splits into
  Golemites, a Furnace makes Fire Spirits. A new unit within `spawn_radius`
  of a same-side unit that can produce it is not counted.
* **Placement must be legal.** A troop card lands in its owner's half (or at
  the bridge). A new enemy unit first seen deep in our half is a unit the
  tracker lost and found again, not a new play. Cards that may be placed
  anywhere (spells, Miner, Goblin Drill, Goblin Barrel...) are exempt.
* **Multi-unit cards are one play.** Same-class units that appear together
  (within `cluster_radius` tiles and `cluster_seconds`) are folded into the
  same play, so Skeleton Army counts once.
* **Ambiguous units** (a Barbarian could be Barbarians, Barbarian Barrel or
  Barbarian Hut) resolve to a card already seen in that player's deck if
  possible, otherwise to the card whose usual unit count is closest.

The resulting play list is a lower bound on what really happened. Spells are
missed unless the detector has classes for spell effects (the KataCR-trained
model does, BuildABot's does not). The elixir estimate is therefore a
ceiling, and should be read as "they have *at most* this much".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..catalog import SPAWNED_BY, CardInfo, Catalog
from ..geometry import RIVER_Y
from .tracker import Track

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
# Seconds per elixir by match time: single for the first 2 minutes, double
# for the third minute and the start of overtime, triple from 4:00.
REGEN_PHASES = ((0.0, 2.8), (120.0, 1.4), (240.0, 0.9))
BRIDGE_MARGIN = 1.5      # tiles past the river a troop may still be deployed


@dataclass
class Play:
    t: float
    side: int
    card: str
    cost: int
    x: float
    y: float
    count: int = 1
    ambiguous: bool = False
    track_ids: List[int] = field(default_factory=list)
    unit_key: str = ""


def regen_seconds(elapsed: float) -> float:
    seconds = REGEN_PHASES[0][1]
    for start, period in REGEN_PHASES:
        if elapsed >= start:
            seconds = period
    return seconds


def elixir_generated(t0: float, t1: float) -> float:
    """Elixir produced between two match times, across phase boundaries."""
    if t1 <= t0:
        return 0.0
    total, t = 0.0, t0
    bounds = [start for start, _ in REGEN_PHASES[1:]] + [float("inf")]
    for (start, period), end in zip(REGEN_PHASES, bounds):
        lo, hi = max(t, start), min(t1, end)
        if hi > lo:
            total += (hi - lo) / period
    return total


class ElixirEstimate:
    """Running estimate of one player's elixir from match time and plays."""

    def __init__(self):
        self.value = START_ELIXIR
        self.at: Optional[float] = None
        self.spent = 0.0

    def advance(self, elapsed: float) -> float:
        if self.at is not None:
            self.value = min(MAX_ELIXIR, self.value + elixir_generated(self.at, elapsed))
        self.at = elapsed
        return self.value

    def charge(self, cost: float, elapsed: float) -> None:
        self.advance(elapsed)
        self.spent += cost
        self.value = max(0.0, self.value - cost)


class CycleTracker:
    """What one player's deck is, and which of its cards cannot be in hand.

    A card returns to hand only after four other cards are played, so a card
    seen fewer than four plays ago is definitely unavailable. The opposite
    ("they have it") cannot be known, because the hand is 4 of the remaining
    cards and the plays we missed are unknown.
    """

    def __init__(self):
        self.deck: List[str] = []
        self.history: List[str] = []

    def observe(self, card: str) -> None:
        if card not in self.deck and len(self.deck) < 8:
            self.deck.append(card)
        self.history.append(card)

    def plays_since(self, card: str) -> Optional[int]:
        for back, played in enumerate(reversed(self.history)):
            if played == card:
                return back
        return None

    def unavailable(self) -> List[str]:
        return [c for c in self.deck if (self.plays_since(c) is not None and self.plays_since(c) < 4)]


class PlayDetector:
    def __init__(self, catalog: Catalog, spawn_radius: float = 3.0,
                 cluster_radius: float = 3.0, cluster_seconds: float = 1.0,
                 sides: Sequence[int] = (0, 1)):
        self.catalog = catalog
        self.sides = tuple(sides)
        self.spawn_radius = spawn_radius
        self.cluster_radius = cluster_radius
        self.cluster_seconds = cluster_seconds
        self.plays: List[Play] = []
        self.decks: Dict[int, CycleTracker] = {0: CycleTracker(), 1: CycleTracker()}
        self._pending: List[Play] = []
        self._candidates: Dict[int, List[CardInfo]] = {}

    def reset(self) -> None:
        self.plays.clear()
        self.decks = {0: CycleTracker(), 1: CycleTracker()}
        self._pending.clear()
        self._candidates.clear()

    def _is_spawn(self, track: Track, live: Sequence[Track], t: float) -> bool:
        parents = SPAWNED_BY.get(track.key)
        if not parents:
            return False
        for other in live:
            if other.id == track.id or other.side != track.side or other.key not in parents:
                continue
            if np.hypot(other.x - track.born_x, other.y - track.born_y) <= self.spawn_radius:
                return True
        return False

    def _legal_spot(self, track: Track, cards: Sequence[CardInfo]) -> bool:
        if any(card.spell for card in cards):
            return True
        if track.side == 1:
            return track.born_y < RIVER_Y + BRIDGE_MARGIN
        return track.born_y >= RIVER_Y - BRIDGE_MARGIN

    def _choose(self, side: int, cards: Sequence[CardInfo], count: int) -> CardInfo:
        known = [c for c in cards if c.name in self.decks[side].deck]
        if len(known) == 1:
            return known[0]
        pool = known or list(cards)
        return min(pool, key=lambda c: (abs(c.units_per_card - count), c.cost))

    def update(self, new_tracks: Sequence[Track], live: Sequence[Track],
               elapsed: float) -> List[Play]:
        """Feed newly confirmed tracks; returns plays finalised this frame.

        A play is held for `cluster_seconds` while the rest of its units
        appear, so that the card is chosen from the final unit count. Plays
        therefore arrive about a second late.
        """
        for track in sorted(new_tracks, key=lambda tr: tr.first_seen):
            if track.kind == "tower" or track.side not in self.sides:
                continue
            cards = self.catalog.cards_for(track.name)
            if not cards or self._is_spawn(track, live, elapsed):
                continue
            # Part of a play still collecting units (the 2nd..15th skeleton)?
            joined = False
            for play in reversed(self._pending):
                if (play.side == track.side and play.unit_key == track.key
                        and np.hypot(play.x - track.born_x, play.y - track.born_y) <= self.cluster_radius):
                    play.count += 1
                    play.track_ids.append(track.id)
                    joined = True
                    break
            if joined or not self._legal_spot(track, cards):
                continue
            self._pending.append(Play(t=elapsed, side=track.side, card="", cost=0,
                                      x=track.born_x, y=track.born_y,
                                      ambiguous=len(cards) > 1, track_ids=[track.id],
                                      unit_key=track.key))
            self._candidates[id(self._pending[-1])] = list(cards)
        return self._finalise(lambda play: elapsed - play.t >= self.cluster_seconds)

    def record(self, play: Play) -> None:
        """Add a play known from elsewhere (our own, read from the hand)."""
        self.plays.append(play)
        self.decks[play.side].observe(play.card)

    def retract(self, cards: Sequence[str], side: int, since: float) -> int:
        """Drop pending plays of these cards (e.g. units that turned out to be
        ours). Returns how many were dropped."""
        keep = []
        for play in self._pending:
            names = {c.name for c in self._candidates[id(play)]}
            if play.side == side and play.t >= since and names & set(cards):
                del self._candidates[id(play)]
                continue
            keep.append(play)
        dropped = len(self._pending) - len(keep)
        self._pending = keep
        return dropped

    def flush(self) -> List[Play]:
        """Finalise everything still pending (end of match or of a recording)."""
        return self._finalise(lambda play: True)

    def _finalise(self, ready) -> List[Play]:
        done: List[Play] = []
        keep: List[Play] = []
        for play in self._pending:
            if not ready(play):
                keep.append(play)
                continue
            cards = self._candidates.pop(id(play))
            card = self._choose(play.side, cards, play.count)
            play.card, play.cost = card.name, card.cost
            self.plays.append(play)
            self.decks[play.side].observe(card.name)
            done.append(play)
        self._pending = keep
        return done
