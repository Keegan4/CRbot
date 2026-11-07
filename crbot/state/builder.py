"""Frame in, game state out.

`GameStateBuilder.process` runs every reader on a frame and updates the
trackers that carry information between frames:

    frame -> screen type -> hand cards -> elixir -> tower HP bars
          -> unit detector -> ally/enemy -> tiles -> tracker -> plays
          -> opponent elixir and cycle
          -> GameState

Our own plays are read from the hand: a card leaving the hand while our
elixir drops by about its cost was played by us. This is exact, covers
spells, and is also the strongest ally/enemy evidence available: a unit of
that card appearing in the next few seconds is ours.

A match starts when a battle screen is first seen, and match time is
measured from that frame. Joining a match part-way (e.g. a recording that
starts late) makes elapsed time, and so the double-elixir switch, wrong. Pass
``elapsed_offset`` in that case.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..capture import Frame
from ..catalog import Catalog, key, load_catalog
from ..geometry import TOWER_TILES, ArenaGrid, Layout, in_arena, layout_for
from ..vision.detector import ALLY, ENEMY, Detector, NullDetector, RawBox
from ..vision.hud import CardReader, ScreenClassifier, read_elixir, read_towers
from ..vision.side import OnnxCropClassifier, SideResolver
from ..vision.types import Detection
from .game_state import GameState, PlayState, TowerHP, UnitState
from .hand import HandTracker
from .plays import ElixirEstimate, Play, PlayDetector
from .towers import TowerTracker
from .tracker import Tracker

BATTLE_END_GRACE = 3.0     # seconds off the battle screen before a match ends
PREEXISTING_SECONDS = 1.0  # units seen this soon after we start watching are not plays
OWN_PLAY_WINDOW = 3.0      # seconds after we play a card in which its units count as ours
OWN_PLAY_LOOKBACK = 1.5    # ...and before: hand voting confirms a play a few frames late


class GameStateBuilder:
    def __init__(self, detector: Optional[Detector] = None,
                 side_classifier: Optional[OnnxCropClassifier] = None,
                 deck: Optional[Iterable[str]] = None,
                 layout: Optional[Layout] = None,
                 catalog: Optional[Catalog] = None,
                 elapsed_offset: float = 0.0):
        self.catalog = catalog or load_catalog()
        self.detector = detector or NullDetector()
        self.deck = [d for d in deck] if deck else None
        self.fixed_layout = layout
        self.elapsed_offset = elapsed_offset
        self.cards = CardReader(deck=self.deck)
        self.screens = ScreenClassifier()
        self.sides = SideResolver(self.catalog, self.deck, side_classifier)
        self.hand = HandTracker()
        self.tracker = Tracker()
        self.towers = TowerTracker()
        self.plays = PlayDetector(self.catalog, sides=(ENEMY,))
        self.enemy_elixir = ElixirEstimate()
        self._prev_hand: List[Optional[str]] = [None] * 4
        self._elixir_history: deque = deque(maxlen=30)
        self._grids: Dict[Tuple[int, int], Tuple[Layout, ArenaGrid]] = {}
        self._battle_start: Optional[float] = None
        self._last_battle_frame: Optional[float] = None
        self.last_detections: List[Detection] = []

    # ------------------------------------------------------------ helpers

    def geometry(self, width: int, height: int) -> Tuple[Layout, ArenaGrid]:
        cached = self._grids.get((width, height))
        if cached is None:
            layout = self.fixed_layout or layout_for(width, height)
            cached = (layout, ArenaGrid.from_layout(layout, width, height))
            self._grids[(width, height)] = cached
        return cached

    def _start_match(self, t: float) -> None:
        self._battle_start = t - self.elapsed_offset
        self.hand.reset()
        self.tracker.reset()
        self.towers.reset()
        self.plays.reset()
        self.enemy_elixir = ElixirEstimate()
        self._prev_hand = [None] * 4
        self._elixir_history.clear()

    def _to_detections(self, image: np.ndarray, raw: List[RawBox], grid: ArenaGrid) -> List[Detection]:
        keep: List[Tuple[RawBox, str, str, Tuple[float, float]]] = []
        for box in raw:
            k = key(box.name)
            if self.catalog.is_tower_class(box.name):
                kind = "tower"
            elif self.catalog.is_unit_class(box.name):
                kind = "unit"
            else:
                continue
            x0, y0, x1, y1 = box.box
            # A sprite stands at the bottom-centre of its box; the centre of
            # the box is around its chest and reads about a tile too far
            # forward. Towers are anchored at their centre.
            foot = ((x0 + x1) / 2, (y0 + y1) / 2) if kind == "tower" else ((x0 + x1) / 2, y1)
            tile = grid.px_to_tile(*foot)
            if not in_arena(*tile, margin=1.0):
                continue
            unit = self.catalog.unit(k)
            if kind == "unit" and unit is None and self.catalog.card(k) is not None:
                kind = "spell"
            elif kind == "unit" and unit is None:
                continue                      # a class we know nothing about
            if kind == "unit" and not unit.flying and self._inside_tower(tile):
                # Nothing walks inside a standing tower. Detectors without
                # tower classes (BuildABot's) sometimes fire on tower art,
                # especially seasonal tower skins.
                continue
            keep.append((box, k, kind, tile))

        resolved = self.sides.resolve(
            image, [b.name for b, *_ in keep], [b.box for b, *_ in keep],
            [tile[1] for *_, tile in keep],
            [None if kind == "tower" else b.side for b, _, kind, _ in keep])
        out: List[Detection] = []
        for (box, k, kind, tile), (side, side_conf, method) in zip(keep, resolved):
            if kind == "tower":
                side, side_conf, method = (0 if tile[1] >= 16 else 1), 1.0, "half"
            unit = self.catalog.unit(k)
            out.append(Detection(name=box.name, key=k, conf=box.conf, box=box.box, tile=tile,
                                 side=side, side_conf=side_conf, side_method=method, kind=kind,
                                 building=bool(unit and unit.building),
                                 flying=bool(unit and unit.flying)))
        return out

    def _inside_tower(self, tile: Tuple[float, float], shrink: float = 0.3) -> bool:
        tx, ty = tile
        for name, (cx, cy) in TOWER_TILES.items():
            state = self.towers.towers.get(name)
            if state is not None and state.destroyed:
                continue
            half = (2.0 if "king" in name else 1.5) - shrink
            if abs(tx - cx) < half and abs(ty - cy) < half:
                return True
        return False

    def _own_plays(self, hand: List[Optional[str]], elixir: float, elapsed: float) -> List[Play]:
        """Cards that just left our hand, confirmed by an elixir drop."""
        self._elixir_history.append((elapsed, elixir))
        played = []
        for before in self._prev_hand:
            if before is None or before in hand:
                continue
            cost = self.catalog.cost(before) or 0
            recent = [e for t, e in self._elixir_history if elapsed - t <= 1.5]
            dropped = (max(recent) - elixir) if recent else cost
            if dropped >= cost - 1.0:
                played.append(Play(t=elapsed, side=ALLY, card=before, cost=cost,
                                   x=float("nan"), y=float("nan")))
        self._prev_hand = list(hand)
        return played

    def _side_hint(self, elapsed: float):
        recent = {p.card for p in self.plays.plays
                  if p.side == ALLY and elapsed - p.t <= OWN_PLAY_WINDOW}

        def hint(det: Detection) -> Optional[int]:
            if recent and any(c.name in recent for c in self.catalog.cards_for(det.name)):
                return ALLY
            return None
        return hint

    @staticmethod
    def _towers_visible(detections: List[Detection]) -> Optional[Dict[str, bool]]:
        towers = [d for d in detections if d.kind == "tower"]
        if not towers:
            return None
        seen = {}
        for name, (tx, ty) in TOWER_TILES.items():
            if "king" in name:
                continue
            seen[name] = any(np.hypot(d.tile[0] - tx, d.tile[1] - ty) < 3.0 for d in towers)
        return seen

    # --------------------------------------------------------------- main

    def process(self, frame: Frame) -> GameState:
        timings: Dict[str, float] = {}
        started = time.perf_counter()
        image = frame.image
        h, w = image.shape[:2]
        layout, grid = self.geometry(w, h)

        screen = self.screens.classify(image)
        reading = self.cards.read(image, layout)
        timings["hud"] = (time.perf_counter() - started) * 1000
        readable = sum(c is not None for c in reading.cards)
        in_battle = screen == "in_game" or (screen == "unknown" and readable >= 2)

        if in_battle:
            if self._battle_start is None:
                self._start_match(frame.t)
            self._last_battle_frame = frame.t
        elif (self._battle_start is not None and self._last_battle_frame is not None
              and frame.t - self._last_battle_frame > BATTLE_END_GRACE):
            self.plays.flush()
            self._battle_start = None

        if self._battle_start is None or not in_battle:
            return GameState(t=frame.t, frame_index=frame.index, screen=screen,
                             in_battle=False, timings_ms=timings)

        elapsed = frame.t - self._battle_start
        elixir = read_elixir(image, layout.elixir_bar)
        hand, next_card = self.hand.update(reading.cards, reading.next_card)
        playable = [bool(hand[i]) and reading.cards[i] == hand[i] and reading.playable[i]
                    for i in range(4)]

        t1 = time.perf_counter()
        raw = self.detector.detect(image)
        timings["detect"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        detections = self._to_detections(image, raw, grid)
        self.last_detections = detections
        towers = self.towers.update(read_towers(image, layout), self._towers_visible(detections))

        for play in self._own_plays(hand, elixir, elapsed):
            self.plays.record(play)
            # The unit may have been seen a moment before the hand settled.
            since = play.t - OWN_PLAY_LOOKBACK
            for tr in sorted(self.tracker.tracks.values(), key=lambda tr: tr.first_seen):
                if tr.first_seen >= since and any(c.name == play.card for c in self.catalog.cards_for(tr.name)):
                    tr.side_votes[ALLY] += 5.0
                    if np.isnan(play.x):
                        play.x, play.y = tr.born_x, tr.born_y
            self.plays.retract([play.card], ENEMY, since)
        live, new, _ = self.tracker.update([d for d in detections if d.kind != "tower"], elapsed,
                                           side_hint=self._side_hint(elapsed))
        # Give our own plays the position where their first unit appeared.
        for tr in new:
            if tr.side != ALLY:
                continue
            names = {c.name for c in self.catalog.cards_for(tr.name)}
            for play in reversed(self.plays.plays):
                if (play.side == ALLY and play.card in names and np.isnan(play.x)
                        and elapsed - play.t <= OWN_PLAY_WINDOW):
                    play.x, play.y = tr.born_x, tr.born_y
                    break
        # Units already on the board when we started watching were not played
        # while we watched: don't count them (matters when joining mid-match).
        new = [tr for tr in new if tr.first_seen >= self.elapsed_offset + PREEXISTING_SECONDS]
        finished = self.plays.update(new, live, elapsed)
        self.enemy_elixir.advance(elapsed)
        for play in finished:
            if play.side == ENEMY:
                self.enemy_elixir.charge(play.cost, elapsed)
        timings["state"] = (time.perf_counter() - t2) * 1000

        units = [UnitState(id=tr.id, name=tr.name, side=tr.side, x=tr.x, y=tr.y, vx=tr.vx, vy=tr.vy,
                           conf=tr.conf, side_conf=tr.side_confidence, building=tr.building,
                           flying=tr.flying, kind=tr.kind, age=elapsed - tr.first_seen,
                           seen_ago=elapsed - tr.last_seen) for tr in live]
        plays = [PlayState(t=p.t, side=p.side, card=p.card, cost=p.cost,
                           x=None if np.isnan(p.x) else p.x, y=None if np.isnan(p.y) else p.y,
                           count=p.count, ambiguous=p.ambiguous) for p in self.plays.plays]
        opponent = self.plays.decks[ENEMY]
        timings["total"] = (time.perf_counter() - started) * 1000
        return GameState(
            t=frame.t, frame_index=frame.index, screen=screen, in_battle=True,
            elapsed=elapsed, elixir=elixir, hand=hand, playable=playable, next_card=next_card,
            towers={n: TowerHP(s.hp, s.destroyed) for n, s in towers.items()},
            units=units, plays=plays, enemy_elixir=round(self.enemy_elixir.value, 2),
            opponent_deck=list(opponent.deck), opponent_unavailable=opponent.unavailable(),
            timings_ms={k: round(v, 1) for k, v in timings.items()},
        )

