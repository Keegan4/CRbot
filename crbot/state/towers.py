"""Smooth princess tower HP over time.

* Readings of a partly covered bar (a unit, spell or emote over it) are
  skipped.
* The HP is the median of the last few clean readings. A few bad frames in
  a row cannot drag it down, and it recovers by itself if they do. Tower HP
  never really rises, but clamping it to "only down" made one burst of bad
  frames permanent, which happened on real footage.
* A missing bar means destroyed *or* covered. A tower is declared destroyed
  only after its bar has been missing for `destroyed_after` consecutive
  frames, and only if it was already low or the detector no longer sees it.

Hasty-CR found one more failure mode. A wrong bar width made every enemy
reading about 12% low, which quietly skewed their reward. The cheap check is
that every tower reads 1.0 at the start of a match. `TowerTracker.start_check`
records whether that held.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from ..vision.hud import BarReading


@dataclass
class TowerState:
    hp: float = 1.0                     # fraction of full
    destroyed: bool = False
    missing: int = 0                    # consecutive frames without a bar
    recent: Deque[float] = field(default_factory=lambda: deque(maxlen=7))


class TowerTracker:
    def __init__(self, names=("enemy_left", "enemy_right", "ally_left", "ally_right"),
                 destroyed_after: int = 6, low_hp: float = 0.35):
        self.names = tuple(names)
        self.destroyed_after = destroyed_after
        self.low_hp = low_hp
        self.towers: Dict[str, TowerState] = {n: TowerState() for n in self.names}
        self.start_readings: Dict[str, float] = {}

    def reset(self) -> None:
        self.towers = {n: TowerState() for n in self.names}
        self.start_readings = {}

    def update(self, readings: Dict[str, BarReading],
               tower_visible: Optional[Dict[str, bool]] = None) -> Dict[str, TowerState]:
        for name, reading in readings.items():
            state = self.towers.setdefault(name, TowerState())
            if state.destroyed:
                continue
            if reading.fraction is None:
                if not reading.missing:
                    continue                    # partly covered: no information
                state.missing += 1
                gone = tower_visible is not None and not tower_visible.get(name, True)
                if state.missing >= self.destroyed_after and (state.hp < self.low_hp or gone):
                    state.destroyed, state.hp = True, 0.0
                continue
            state.missing = 0
            self.start_readings.setdefault(name, reading.fraction)
            state.recent.append(reading.fraction)
            if len(state.recent) >= 3:
                state.hp = float(np.median(state.recent))
        return self.towers

    def start_check(self, tolerance: float = 0.05) -> Dict[str, bool]:
        """Did each tower read (close to) full when first seen?"""
        return {name: abs(value - 1.0) <= tolerance for name, value in self.start_readings.items()}
