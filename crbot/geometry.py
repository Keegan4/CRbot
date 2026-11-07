"""Screen layout and the pixel <-> tile mapping.

Conventions used everywhere in crbot:

* **Pixels**: full captured frame, x to the right, y down.
* **Tiles**: continuous coordinates on the 18 x 32 arena, x to the right
  (0..18), y *down the screen* (0..32). The opponent is at the top, so their
  half is ``y < 16`` and ours is ``y >= 16``. Tile ``(col, row)`` covers
  ``[col, col+1) x [row, row+1)``; its centre is ``(col + 0.5, row + 0.5)``.

Every region is stored as a fraction of the frame. A 720x1280 and a 1080x1920
capture therefore share one layout. A different *aspect ratio* does not:
Clash Royale fits the arena to the screen width on tall phones and to the
height on 9:16 screens, so each aspect ratio needs its own profile. The
9:16 profile below has been used to place taps in two open-source bots
(ClashRoyaleBuildABot, Hasty-CR). Check it on your own screenshots with
``python -m crbot calibrate`` before trusting it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

GRID_W, GRID_H = 18, 32
RIVER_Y = 16.0          # rows >= 16 are our half

# Tower centres in tile coordinates. Princess towers are 3x3, kings 4x4.
TOWER_TILES: Dict[str, Tuple[float, float]] = {
    "enemy_king": (9.0, 3.0),
    "enemy_left": (3.5, 7.5),
    "enemy_right": (14.5, 7.5),
    "ally_left": (3.5, 24.5),
    "ally_right": (14.5, 24.5),
    "ally_king": (9.0, 29.0),
}


@dataclass(frozen=True)
class Rect:
    """A region as fractions of the frame: top-left (x, y), width, height."""
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in integer pixels, clipped to the frame."""
        x0 = int(round(self.x * width))
        y0 = int(round(self.y * height))
        x1 = int(round((self.x + self.w) * width))
        y1 = int(round((self.y + self.h) * height))
        return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.to_pixels(w, h)
        return image[y0:y1, x0:x1]

    def expand(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)


def _px(x: float, y: float, w: float, h: float, ref_w: float, ref_h: float) -> Rect:
    """A rect given in pixels of a reference resolution."""
    return Rect(x / ref_w, y / ref_h, w / ref_w, h / ref_h)


@dataclass(frozen=True)
class Layout:
    """Where everything is on screen, for one aspect ratio."""
    name: str
    aspect: float                       # height / width
    arena: Rect                         # exactly the 18 x 32 tile rectangle
    hand: Tuple[Rect, Rect, Rect, Rect]
    next_card: Rect
    elixir_bar: Rect
    tower_hp: Dict[str, Rect] = field(default_factory=dict)
    verified: bool = False
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(data: dict) -> "Layout":
        rect = lambda d: Rect(**d)  # noqa: E731
        return Layout(
            name=data["name"], aspect=float(data["aspect"]),
            arena=rect(data["arena"]),
            hand=tuple(rect(r) for r in data["hand"]),  # type: ignore[arg-type]
            next_card=rect(data["next_card"]),
            elixir_bar=rect(data["elixir_bar"]),
            tower_hp={k: rect(v) for k, v in data.get("tower_hp", {}).items()},
            verified=bool(data.get("verified", False)),
            notes=data.get("notes", ""),
        )

    @staticmethod
    def load(path: Path) -> "Layout":
        return Layout.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# 9:16 (720x1280, 1080x1920; most emulators). Measurements from
# ClashRoyaleBuildABot's constants (368x652 screenshots and 720x1280 taps),
# which Hasty-CR independently re-derived at 1080x1920.
_REF_W, _REF_H = 368.0, 652.0
LAYOUT_9x16 = Layout(
    name="9x16",
    aspect=16 / 9,
    # Tile (0,0) top-left corner at (52, 100.8) on 720x1280; tiles 34 x 27.6.
    arena=Rect(52 / 720, 100.8 / 1280, 18 * 34 / 720, 32 * 27.6 / 1280),
    hand=tuple(_px(84 + 69 * i, 543, 61, 73, _REF_W, _REF_H) for i in range(4)),  # type: ignore[arg-type]
    next_card=_px(21, 609, 26, 33, _REF_W, _REF_H),
    elixir_bar=_px(100, 628, 250, 15, _REF_W, _REF_H),
    tower_hp={
        "enemy_left": _px(74, 95, 40, 10, _REF_W, _REF_H),
        "enemy_right": _px(266, 95, 40, 10, _REF_W, _REF_H),
        "ally_left": _px(74, 404, 40, 10, _REF_W, _REF_H),
        "ally_right": _px(266, 404, 40, 10, _REF_W, _REF_H),
    },
    verified=True,
    notes="Used for live play by ClashRoyaleBuildABot and Hasty-CR.",
)

# Tall phones (about 2.16:1, e.g. 592x1280). From KataCR's crop constants,
# which were used to cut its detection dataset. The hand and elixir regions
# are converted from its card-table crop; the elixir bar itself was not
# measured there, so it is an estimate. Unverified: calibrate before use.
_T3 = Rect(0.000, 0.808, 1.000, 0.155)          # KataCR "part3": card table


def _in(parent: Rect, x: float, y: float, w: float, h: float) -> Rect:
    return Rect(parent.x + x * parent.w, parent.y + y * parent.h, w * parent.w, h * parent.h)


LAYOUT_TALL = Layout(
    name="tall_2.16",
    aspect=2.165,
    arena=Rect(0.021, 0.073, 0.960, 0.700),
    hand=(_in(_T3, 0.222, 0.0, 0.185, 0.745), _in(_T3, 0.410, 0.0, 0.185, 0.745),
          _in(_T3, 0.600, 0.0, 0.185, 0.745), _in(_T3, 0.785, 0.0, 0.185, 0.745)),
    next_card=_in(_T3, 0.047, 0.590, 0.100, 0.365),
    elixir_bar=_in(_T3, 0.335, 0.72, 0.640, 0.12),
    tower_hp={},
    verified=False,
    notes="From KataCR's dataset crops; elixir bar estimated; no tower HP regions.",
)

LAYOUTS = (LAYOUT_9x16, LAYOUT_TALL)


def layout_for(width: int, height: int, tolerance: float = 0.04) -> Layout:
    """Pick the profile whose aspect ratio matches the frame."""
    aspect = height / max(1, width)
    best = min(LAYOUTS, key=lambda layout: abs(layout.aspect - aspect))
    if abs(best.aspect - aspect) / best.aspect > tolerance:
        raise ValueError(
            f"No screen layout for a {width}x{height} frame (aspect {aspect:.3f}). "
            "Set the emulator to 720x1280 or 1080x1920, or make a layout file "
            "with `python -m crbot calibrate` and pass it with --layout.")
    return best


class ArenaGrid:
    """Affine map between tile coordinates and frame pixels.

    ``pixel = A @ [tx, ty, 1]``. Built from a layout's arena rectangle by
    default; `fit` builds one from matched points (e.g. detected towers) when
    a recording's framing is unknown.
    """

    def __init__(self, matrix: np.ndarray):
        self.A = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
        full = np.vstack([self.A, [0.0, 0.0, 1.0]])
        self.A_inv = np.linalg.inv(full)[:2]

    @classmethod
    def from_layout(cls, layout: Layout, width: int, height: int) -> "ArenaGrid":
        r = layout.arena
        tw = r.w * width / GRID_W
        th = r.h * height / GRID_H
        return cls(np.array([[tw, 0.0, r.x * width], [0.0, th, r.y * height]]))

    @classmethod
    def fit(cls, tiles: Sequence[Tuple[float, float]],
            pixels: Sequence[Tuple[float, float]]) -> Tuple["ArenaGrid", float]:
        """Least-squares affine fit. Returns the grid and RMS error in tiles."""
        if len(tiles) < 3:
            raise ValueError("need at least three matched points")
        src = np.array([[x, y, 1.0] for x, y in tiles])
        dst = np.array(pixels, dtype=np.float64)
        solution, *_ = np.linalg.lstsq(src, dst, rcond=None)
        grid = cls(solution.T)
        back = np.array([grid.px_to_tile(px, py) for px, py in pixels])
        rms = float(np.sqrt(np.mean(np.sum((back - np.array(tiles)) ** 2, axis=1))))
        return grid, rms

    @property
    def tile_size_px(self) -> Tuple[float, float]:
        return float(abs(self.A[0, 0])), float(abs(self.A[1, 1]))

    def tile_to_px(self, tx: float, ty: float) -> Tuple[float, float]:
        x, y = self.A @ np.array([tx, ty, 1.0])
        return float(x), float(y)

    def px_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        x, y = self.A_inv @ np.array([px, py, 1.0])
        return float(x), float(y)

    def cell_center_px(self, col: int, row: int) -> Tuple[float, float]:
        return self.tile_to_px(col + 0.5, row + 0.5)

    def bounds_px(self) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the whole tile rectangle in pixels."""
        corners = np.array([self.tile_to_px(x, y) for x, y in
                            ((0, 0), (GRID_W, 0), (0, GRID_H), (GRID_W, GRID_H))])
        return (float(corners[:, 0].min()), float(corners[:, 1].min()),
                float(corners[:, 0].max()), float(corners[:, 1].max()))


def in_arena(tx: float, ty: float, margin: float = 0.0) -> bool:
    return -margin <= tx < GRID_W + margin and -margin <= ty < GRID_H + margin


def is_our_half(ty: float) -> bool:
    return ty >= RIVER_Y


def cell_of(tx: float, ty: float) -> Tuple[int, int]:
    """Integer cell, clipped into the grid."""
    return (int(np.clip(np.floor(tx), 0, GRID_W - 1)),
            int(np.clip(np.floor(ty), 0, GRID_H - 1)))
