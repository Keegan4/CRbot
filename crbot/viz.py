"""Draw what perception sees on top of the frame, for checking it by eye."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .geometry import GRID_H, GRID_W, ArenaGrid, Layout, Rect
from .state.game_state import GameState
from .vision.types import Detection

ALLY_COLOUR = (255, 160, 40)      # BGR
ENEMY_COLOUR = (60, 60, 240)
REGION_COLOUR = (0, 220, 255)
TEXT_COLOUR = (255, 255, 255)


def _rect(cv2, image, rect: Rect, colour, thickness=1):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = rect.to_pixels(w, h)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, thickness)


def draw_layout(image: np.ndarray, layout: Layout, grid: ArenaGrid, tiles: bool = True) -> np.ndarray:
    """Grid lines and every fixed screen region. Use to check calibration."""
    import cv2

    out = image.copy()
    if tiles:
        for col in range(GRID_W + 1):
            a, b = grid.tile_to_px(col, 0), grid.tile_to_px(col, GRID_H)
            cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (80, 80, 80), 1)
        for row in range(GRID_H + 1):
            colour = (255, 255, 0) if row == 16 else (80, 80, 80)
            a, b = grid.tile_to_px(0, row), grid.tile_to_px(GRID_W, row)
            cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), colour, 1)
    for rect in layout.hand:
        _rect(cv2, out, rect, REGION_COLOUR)
    _rect(cv2, out, layout.next_card, REGION_COLOUR)
    _rect(cv2, out, layout.elixir_bar, (255, 0, 255))
    for rect in layout.tower_hp.values():
        _rect(cv2, out, rect, (0, 255, 0))
    return out


def draw_state(image: np.ndarray, state: GameState, grid: Optional[ArenaGrid] = None,
               detections: Iterable[Detection] = (), layout: Optional[Layout] = None) -> np.ndarray:
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.35, w / 1400)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if layout is not None and grid is not None:
        out = draw_layout(out, layout, grid, tiles=False)

    for det in detections:
        colour = ALLY_COLOUR if det.side == 0 else ENEMY_COLOUR
        x0, y0, x1, y1 = (int(v) for v in det.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 1)

    if grid is not None:
        for unit in state.units:
            colour = ALLY_COLOUR if unit.side == 0 else ENEMY_COLOUR
            px, py = grid.tile_to_px(unit.x, unit.y)
            cv2.circle(out, (int(px), int(py)), 4, colour, -1)
            tip = grid.tile_to_px(unit.x + unit.vx, unit.y + unit.vy)
            cv2.arrowedLine(out, (int(px), int(py)), (int(tip[0]), int(tip[1])), colour, 1, tipLength=0.3)
            label = f"{unit.id}:{unit.name}"
            cv2.putText(out, label, (int(px) + 5, int(py) - 5), font, scale * 0.8, (0, 0, 0), 3)
            cv2.putText(out, label, (int(px) + 5, int(py) - 5), font, scale * 0.8, colour, 1)

    lines = [f"t={state.t:.1f}s  screen={state.screen}  battle={state.in_battle}"]
    if state.in_battle:
        lines.append(f"elapsed={state.elapsed:.1f}s  elixir={state.elixir}  enemy elixir<={state.enemy_elixir}")
        hand = [f"{c or '?'}{'' if p else '(x)'}" for c, p in zip(state.hand, state.playable)]
        lines.append("hand: " + ", ".join(hand) + f"  next: {state.next_card or '?'}")
        towers = "  ".join(f"{k}={'X' if v.destroyed else f'{v.hp:.2f}'}" for k, v in state.towers.items())
        lines.append("towers: " + towers)
        recent = [f"{'them' if p.side else 'us'}:{p.card}@{p.t:.0f}s" for p in state.plays[-4:]]
        lines.append("plays: " + ", ".join(recent))
        if state.opponent_deck:
            lines.append("their deck: " + ", ".join(state.opponent_deck))
        lines.append("ms: " + " ".join(f"{k}={v}" for k, v in state.timings_ms.items()))
    text_scale = scale * 0.8
    line_h = int(28 * text_scale) + 4
    panel_h = line_h * len(lines) + 8
    panel = out[:panel_h].copy()
    panel[:] = (0, 0, 0)
    out[:panel_h] = cv2.addWeighted(out[:panel_h], 0.35, panel, 0.65, 0)
    y = line_h
    for line in lines:
        cv2.putText(out, line, (6, y), font, text_scale, TEXT_COLOUR, 1, cv2.LINE_AA)
        y += line_h
    return out
