"""Temporal smoothing for the hand.

The card hash is not stable from frame to frame. Animated card art (Hasty-CR
saw Ice Spirit read as Hog Rider and then blank on consecutive frames)
and a finger dragging a card both cause one-frame misreads. A slot's card
is believed once it has won `min_votes` of the last `window` readings.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, List, Optional


class SlotVote:
    def __init__(self, window: int = 4, min_votes: int = 2):
        self.history: Deque[Optional[str]] = deque(maxlen=window)
        self.min_votes = min_votes

    def update(self, value: Optional[str]) -> Optional[str]:
        self.history.append(value)
        counts = Counter(v for v in self.history if v is not None)
        if not counts:
            return None
        best, votes = counts.most_common(1)[0]
        return best if votes >= min(self.min_votes, len(self.history)) else None

    def clear(self) -> None:
        self.history.clear()


class HandTracker:
    def __init__(self, window: int = 4, min_votes: int = 2):
        self.slots = [SlotVote(window, min_votes) for _ in range(4)]
        self.next_slot = SlotVote(window, min_votes)

    def update(self, cards: List[Optional[str]], next_card: Optional[str]):
        stable = [slot.update(card) for slot, card in zip(self.slots, cards)]
        return stable, self.next_slot.update(next_card)

    def reset(self) -> None:
        for slot in self.slots:
            slot.clear()
        self.next_slot.clear()
