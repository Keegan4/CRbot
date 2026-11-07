from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """One object in one frame, after side resolution and tile mapping."""
    name: str                                  # detector class name
    key: str                                   # canonical catalog key
    conf: float
    box: Tuple[float, float, float, float]     # frame pixels
    tile: Tuple[float, float]                  # ground point in tiles
    side: int                                  # 0 = ours, 1 = opponent's
    side_conf: float
    side_method: str                           # detector | deck | classifier | half
    kind: str = "unit"                         # unit | spell | tower
    building: bool = False
    flying: bool = False
