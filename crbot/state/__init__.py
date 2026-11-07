"""Turning per-frame readings into a game state that persists across frames."""

from .builder import GameStateBuilder
from .game_state import GameState

__all__ = ["GameStateBuilder", "GameState"]
