"""Build synthetic 9:16 game frames for tests.

The HUD regions (hand, next card, elixir bar, tower HP bars) are painted with
the same geometry and colours the readers expect, using the real reference
card art. An optional arena image is pasted into the 18x32 tile rectangle.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from crbot.catalog import ASSETS
from crbot.geometry import LAYOUT_9x16, Layout
from crbot.vision.hud import ALLY_BAR, ENEMY_BAR

ELIXIR_PURPLE = (230, 60, 200)    # BGR
BAR_EMPTY = (40, 30, 30)


def _paste(cv2, frame, rect, image):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = rect.to_pixels(w, h)
    frame[y0:y1, x0:x1] = cv2.resize(image, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)


def card_image(name: str) -> np.ndarray:
    import cv2

    image = cv2.imread(str(ASSETS / "cards" / f"{name}.jpg"), cv2.IMREAD_COLOR)
    assert image is not None, name
    return image


def make_frame(width: int = 720, height: int = 1280, hand: Sequence[str] = (),
               next_card: Optional[str] = None, elixir: float = 5.0,
               towers: Optional[Dict[str, float]] = None, arena: Optional[np.ndarray] = None,
               greyed: Sequence[int] = (), layout: Layout = LAYOUT_9x16) -> np.ndarray:
    import cv2

    frame = np.full((height, width, 3), (60, 140, 90), dtype=np.uint8)
    if arena is not None:
        _paste(cv2, frame, layout.arena, arena)
    for slot, name in enumerate(hand):
        image = card_image(name)
        if slot in greyed:
            # Live greyed-out cards keep their contrast and lose their colour.
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            image = cv2.cvtColor(np.clip(0.8 * grey + 50, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        _paste(cv2, frame, layout.hand[slot], image)
    if next_card:
        _paste(cv2, frame, layout.next_card, card_image(next_card))

    x0, y0, x1, y1 = layout.elixir_bar.to_pixels(width, height)
    frame[y0:y1, x0:x1] = BAR_EMPTY
    reach = x0 + int(round((x1 - x0) * elixir / 10.0))
    frame[y0:y1, x0:reach] = ELIXIR_PURPLE

    for name, fraction in (towers or {}).items():
        rect = layout.tower_hp[name]
        full, drained = ALLY_BAR if name.startswith("ally") else ENEMY_BAR
        x0, y0, x1, y1 = rect.to_pixels(width, height)
        bar_y0, bar_y1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        split = x0 + int(round((x1 - x0) * fraction))
        frame[bar_y0:bar_y1, x0:x1] = drained[::-1]
        frame[bar_y0:bar_y1, x0:split] = full[::-1]
    return frame
