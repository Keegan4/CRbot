"""Collect hand-card templates from your own game.

Card art changes between game versions, and the bundled reference art (from
ClashRoyaleBuildABot) no longer matches every card. This crops the four hand
slots from a recording or a live feed and groups identical-looking crops.
It writes a few examples of each group to ``<out>/group_NN__<guess>/``.

You then rename each folder to its card name (``musketeer``, ``hog_rider``,
...). A greyed-out card forms its own group: move its images into the same
card folder. Move the result to ``models/cards/``. `CardReader` prefers these
templates over the bundled art.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..geometry import Layout, layout_for
from .hud import CardReader, card_hash, is_empty_slot, is_greyed


@dataclass
class Group:
    centre: np.ndarray
    count: int = 0
    greyed: bool = False
    folder: Optional[Path] = None
    saved: List[Path] = field(default_factory=list)


def collect(frames, out: Path, layout: Optional[Layout] = None, every: float = 0.5,
            threshold: float = 0.3, per_group: int = 5) -> List[Group]:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    groups: List[Group] = []
    guesser = CardReader(user_dir=None)
    last_t = None
    for frame in frames:
        if last_t is not None and frame.t - last_t < every:
            continue
        last_t = frame.t
        lay = layout or layout_for(*frame.size)
        for rect in lay.hand:
            crop = rect.crop(frame.image)
            if is_empty_slot(crop):
                continue
            h = card_hash(crop)
            best, best_d = None, threshold
            for g in groups:
                d = float(np.abs(g.centre - h).mean())
                if d < best_d:
                    best, best_d = g, d
            if best is None:
                best = Group(centre=h, greyed=is_greyed(crop))
                guess = guesser.names[int(np.argmin(guesser.distances(crop)))]
                suffix = "_greyed" if best.greyed else ""
                best.folder = out / f"group_{len(groups):02d}__{guess}{suffix}"
                best.folder.mkdir(exist_ok=True)
                groups.append(best)
            best.count += 1
            if len(best.saved) < per_group:
                path = best.folder / f"{frame.index:06d}_{len(best.saved)}.png"
                cv2.imwrite(str(path), crop)
                best.saved.append(path)
    return groups
