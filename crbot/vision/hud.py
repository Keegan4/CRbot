"""Read the fixed screen regions: hand cards, elixir, princess tower HP, and
which screen is showing.

None of these need a trained model. Cards are matched against templates with
a small brightness-normalised greyscale hash. Elixir and tower HP are
measured from bar colours. Each reader gives a raw per-frame value. Smoothing
over time is done in `crbot.state`, because single frames are noisy: animated
card art changes the hash, and units walk in front of tower bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..catalog import ASSETS, key, load_catalog
from ..geometry import Layout, Rect

HASH_SIZE = 8
USER_CARD_DIR = Path("models/cards")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def card_hash(image: np.ndarray, size: int = HASH_SIZE) -> np.ndarray:
    """8x8 greyscale thumbnail, normalised to zero mean and unit variance.

    Normalising removes overall brightness and contrast. A greyed-out
    (unaffordable) card is close to a brightness/contrast change of the same
    card in greyscale, so it still matches its coloured template. Measured on
    live frames: the same card scores 0.03-0.06, a greyed card against its
    coloured template about 0.55, and different cards 0.7 or more.
    """
    import cv2

    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = small.astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def is_empty_slot(crop: np.ndarray, max_std: float = 25.0) -> bool:
    """An empty hand slot is a flat blue panel (greyscale std ~8; cards >55)."""
    import cv2

    if crop.size == 0:
        return True
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) < max_std


def is_greyed(crop: np.ndarray, max_spread: float = 20.0) -> bool:
    """Unaffordable cards are drawn without colour (channel spread ~4-11;
    coloured cards 34 or more)."""
    if crop.size == 0:
        return True
    c = crop.astype(np.int16)
    return float((c.max(axis=2) - c.min(axis=2)).mean()) < max_spread


@dataclass
class HandReading:
    cards: List[Optional[str]]          # 4 slots; None = empty or unreadable
    distances: List[float]
    playable: List[bool]
    empty: List[bool]
    next_card: Optional[str]
    next_distance: float


def load_templates(deck: Optional[Iterable[str]] = None,
                   user_dir: Optional[Path] = USER_CARD_DIR) -> Dict[str, List[np.ndarray]]:
    """Card templates by card name.

    Templates captured from your own game (``models/cards/<card>/*.png``,
    made with ``python -m crbot cards collect``) are preferred. Card art
    changes between game versions, and BuildABot's reference art no longer
    matches every card in the current game. The bundled art is the fallback
    for cards without your own templates.
    """
    import cv2

    catalog = load_catalog()
    wanted = {key(c) for c in deck} if deck else None

    def card_name(stem: str) -> Optional[str]:
        info = catalog.card(stem)
        if info is None or (wanted is not None and key(info.name) not in wanted):
            return None
        return info.name

    templates: Dict[str, List[np.ndarray]] = {}
    if user_dir is not None and Path(user_dir).is_dir():
        for folder in sorted(p for p in Path(user_dir).iterdir() if p.is_dir()):
            name = card_name(folder.name)
            if name is None:
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if image is not None:
                        templates.setdefault(name, []).append(card_hash(image))
    user_cards = set(templates)
    for path in sorted((ASSETS / "cards").glob("*.jpg")):
        name = card_name(path.stem)
        if name is None or name in user_cards:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            templates.setdefault(name, []).append(card_hash(image))
    return templates


class CardReader:
    """Identify the four hand cards and the next card.

    Give it your deck. With eight candidates instead of over a hundred, the
    bundled art is good enough. On live frames it ranked the right card first
    in every case tested, including greyed-out cards. Without a deck, collect
    your own templates. The four slots are assigned jointly (Hungarian
    algorithm), because a hand never holds the same card twice.
    """

    def __init__(self, deck: Optional[Iterable[str]] = None, max_distance: Optional[float] = None,
                 user_dir: Optional[Path] = USER_CARD_DIR):
        self.templates = load_templates(deck, user_dir)
        if not self.templates:
            raise ValueError("no card templates matched the deck")
        self.names = sorted(self.templates)
        # Looser with a known deck: fewer candidates, so less to confuse.
        self.max_distance = max_distance if max_distance is not None else (1.0 if deck else 0.75)
        # Without a deck, also require a clear winner. On a phone screenshot,
        # every real card beat its runner-up by 0.18 or more, and the one
        # wrong guess won by 0.01.
        self.min_margin = 0.0 if deck else 0.08

    def _confident(self, distances: np.ndarray, index: int) -> bool:
        best = float(distances[index])
        if best > self.max_distance:
            return False
        if self.min_margin and len(distances) > 1:
            runner_up = float(np.partition(distances, 1)[1]) if np.argmin(distances) == index else float(distances.min())
            return runner_up - best >= self.min_margin
        return True

    def distances(self, crop: np.ndarray) -> np.ndarray:
        h = card_hash(crop)
        return np.array([min(float(np.abs(t - h).mean()) for t in self.templates[n]) for n in self.names])

    def read(self, image: np.ndarray, layout: Layout) -> HandReading:
        from scipy.optimize import linear_sum_assignment

        crops = [rect.crop(image) for rect in layout.hand]
        empty = [is_empty_slot(c) for c in crops]
        cards: List[Optional[str]] = [None] * 4
        distances = [float("inf")] * 4
        slots = [i for i in range(4) if not empty[i]]
        if slots:
            cost = np.stack([self.distances(crops[i]) for i in slots])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                distances[slots[r]] = float(cost[r, c])
                if self._confident(cost[r], c):
                    cards[slots[r]] = self.names[c]
        playable = [cards[i] is not None and not is_greyed(crops[i]) for i in range(4)]

        next_card, next_distance = None, float("inf")
        next_crop = layout.next_card.crop(image)
        if not is_empty_slot(next_crop):
            in_hand = {c for c in cards if c}
            cost = self.distances(next_crop)
            for index in np.argsort(cost):
                if self.names[index] in in_hand:
                    continue
                next_distance = float(cost[index])
                if self._confident(cost, index):
                    next_card = self.names[index]
                break
        return HandReading(cards, distances, playable, empty, next_card, next_distance)


# ------------------------------------------------------------------ elixir


def read_elixir(image: np.ndarray, bar: Rect, min_red: float = 100.0, window: int = 5) -> float:
    """Elixir (0-10, fractional) from how far the fill reaches along the bar.

    The filled part is magenta (red channel ~200) and the empty part is dark
    blue (red channel ~3), so the red channel alone separates them. The white
    elixir number drawn over the start of the bar also has a high red channel,
    so it does not break the run. A short moving window bridges the thin
    dividers between elixir units.
    """
    crop = bar.crop(image)
    if crop.size == 0:
        return 0.0
    red = np.median(crop[:, :, 2].astype(np.float32), axis=0)      # per column
    filled = (red > min_red).astype(np.float32)
    if len(filled) >= window:
        filled = np.convolve(filled, np.ones(window) / window, mode="same")
    empty = np.nonzero(filled < 0.5)[0]
    reach = len(filled) if len(empty) == 0 else int(empty[0])
    return round(10.0 * reach / len(filled), 2)


# --------------------------------------------------------------- tower HP

# Bar colours (RGB) on the princess towers: the filled part and the drained
# part. From ClashRoyaleBuildABot.
ALLY_BAR = ((111, 208, 252), (63, 79, 112))
ENEMY_BAR = ((224, 35, 93), (90, 49, 68))


@dataclass
class BarReading:
    fraction: Optional[float]       # None when no clean bar was found
    covered: float                  # share of the bar row that has bar colours

    @property
    def missing(self) -> bool:
        """Nothing bar-like at all (destroyed, or completely hidden)."""
        return self.fraction is None and self.covered < 0.3


def read_hp_bar(image: np.ndarray, bar: Rect, colours, full_tol: float = 40.0,
                drained_tol: float = 30.0, min_cover: float = 0.85) -> BarReading:
    """Fraction of a princess tower's HP bar that is still filled.

    Each pixel is classed as "filled colour", "drained colour" or neither.
    The bar is the row where the most pixels are one of the two. In the
    current game UI the HP number is drawn over the lower half of the bar,
    and its dark outline looks like the drained colour. Picking the row with
    the lowest *average* distance therefore chose the number and read 0.
    The fill level is the length of the filled run from the left (the bar
    drains from the right), allowing small gaps.

    ``None`` means no clean bar was found. ``covered`` tells the two cases
    apart: about 0 when there is no bar (destroyed), and in between when
    something partly covers it (a unit, a spell, an emote). Partly covered
    readings are not trusted. Clean bars measured 0.94 or more on live frames.
    """
    import cv2

    crop = bar.crop(image)
    if crop.size == 0:
        return BarReading(None, float("inf"))
    rgb = cv2.blur(crop, (3, 3))[:, :, ::-1].astype(np.float32)
    full, drained = (np.array(c, dtype=np.float32) for c in colours)
    is_full = np.abs(rgb - full).mean(axis=2) < full_tol
    is_drained = np.abs(rgb - drained).mean(axis=2) < drained_tol
    cover = (is_full | is_drained).sum(axis=1) + 0.5 * is_full.sum(axis=1)
    row = int(np.argmax(cover))
    width = crop.shape[1]
    covered = float((is_full[row] | is_drained[row]).mean())
    if covered < min_cover:
        return BarReading(None, covered)
    filled = is_full[row]
    # The run may start a few pixels in, after the bar's frame.
    start = next((i for i in range(min(width, max(3, width // 6))) if filled[i]), None)
    if start is None:
        return BarReading(0.0, covered)
    end, gap = start, 0
    for i in range(start, width):
        if filled[i]:
            end, gap = i + 1, 0
        else:
            gap += 1
            if gap > 2:
                break
    return BarReading(round(min(1.0, end / width), 3), covered)


def read_towers(image: np.ndarray, layout: Layout) -> Dict[str, BarReading]:
    out = {}
    for name, rect in layout.tower_hp.items():
        colours = ALLY_BAR if name.startswith("ally") else ENEMY_BAR
        out[name] = read_hp_bar(image, rect, colours)
    return out


# ------------------------------------------------------------------ screen


@dataclass(frozen=True)
class ScreenRef:
    name: str
    ltrb: Tuple[int, int, int, int]       # in 720x1280 pixels


# Small regions that identify each screen, with reference crops in
# assets/screens. From ClashRoyaleBuildABot.
SCREENS = (
    ScreenRef("in_game", (148, 1254, 163, 1274)),
    ScreenRef("lobby", (424, 126, 506, 181)),
    ScreenRef("end_of_game", (379, 1122, 560, 1201)),
    ScreenRef("bypass_end_of_game", (269, 1123, 451, 1201)),
)


class ScreenClassifier:
    """Which screen is showing: in_game, lobby, end_of_game or unknown.

    Calibrated for 9:16 frames. On other aspect ratios it returns "unknown",
    and the state builder treats any frame with a readable hand as in-game.
    """

    def __init__(self, threshold: float = 30.0):
        import cv2

        self.threshold = threshold
        self.refs = []
        for ref in SCREENS:
            path = ASSETS / "screens" / f"{ref.name}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                self.refs.append((ref, self._hash(image)))

    @staticmethod
    def _hash(image: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(image, (8, 8), interpolation=cv2.INTER_LINEAR).astype(np.float32).ravel()

    def classify(self, image: np.ndarray) -> str:
        h, w = image.shape[:2]
        if abs(h / w - 16 / 9) > 0.06:
            return "unknown"
        best, best_diff = "unknown", self.threshold
        for ref, target in self.refs:
            l, t, r, b = ref.ltrb
            crop = image[int(t * h / 1280):int(b * h / 1280), int(l * w / 720):int(r * w / 720)]
            if crop.size == 0:
                continue
            diff = float(np.abs(self._hash(crop) - target).mean())
            if diff < best_diff:
                best, best_diff = ref.name, diff
        return best
