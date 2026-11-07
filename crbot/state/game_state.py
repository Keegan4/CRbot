"""The game state: everything perception knows about one moment of a match."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..geometry import GRID_H, GRID_W, cell_of


@dataclass
class UnitState:
    id: int
    name: str
    side: int                    # 0 = ours, 1 = opponent's
    x: float                     # tiles, ground point
    y: float
    vx: float                    # tiles/second
    vy: float
    conf: float
    side_conf: float
    building: bool
    flying: bool
    kind: str                    # unit | spell
    age: float                   # seconds since first seen
    seen_ago: float              # seconds since last detected


@dataclass
class PlayState:
    t: float
    side: int
    card: str
    cost: int
    x: Optional[float]           # None until a unit of the card is seen (and for most spells)
    y: Optional[float]
    count: int
    ambiguous: bool


@dataclass
class TowerHP:
    hp: Optional[float]
    destroyed: bool


@dataclass
class GameState:
    t: float                                  # frame timestamp (s)
    frame_index: int
    screen: str                               # in_game | lobby | end_of_game | unknown
    in_battle: bool
    elapsed: Optional[float] = None           # match time since the battle was first seen
    elixir: Optional[float] = None            # ours, read from the bar
    hand: List[Optional[str]] = field(default_factory=lambda: [None] * 4)
    playable: List[bool] = field(default_factory=lambda: [False] * 4)
    next_card: Optional[str] = None
    towers: Dict[str, TowerHP] = field(default_factory=dict)
    units: List[UnitState] = field(default_factory=list)
    plays: List[PlayState] = field(default_factory=list)   # this match, oldest first
    enemy_elixir: Optional[float] = None      # upper-bound estimate
    opponent_deck: List[str] = field(default_factory=list)
    opponent_unavailable: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        def clean(value):
            if isinstance(value, float):
                return round(value, 3)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        return json.dumps(clean(self.to_dict()), separators=(",", ":"))

    def allies(self) -> List[UnitState]:
        return [u for u in self.units if u.side == 0]

    def enemies(self) -> List[UnitState]:
        return [u for u in self.units if u.side == 1]

    def planes(self) -> np.ndarray:
        """A compact spatial encoding, (6, 32, 18): per side, counts of ground
        troops, air troops and buildings per tile. A starting point for the
        policy's observation; it will grow per-unit features later."""
        out = np.zeros((6, GRID_H, GRID_W), dtype=np.float32)
        for u in self.units:
            col, row = cell_of(u.x, u.y)
            channel = 2 if u.building else (1 if u.flying else 0)
            out[3 * u.side + channel, row, col] += 1.0
        return out
