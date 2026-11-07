"""Card and unit metadata, and name matching across detector vocabularies.

Detectors name things differently. The KataCR dataset writes ``hog-rider`` and
``mini-pekka``; BuildABot writes ``hog_rider`` and ``minipekka``. Every name is
reduced to a canonical key (lower case, letters and digits only) before it is
looked up. Then one table works for both vocabularies, and a new detector
only needs an alias entry where its names really differ.

The metadata in ``assets/cards.json`` and ``assets/units.json`` comes from
ClashRoyaleBuildABot (MIT). See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

ASSETS = Path(__file__).resolve().parent / "assets"

# Detector class names that are scenery, UI or tower furniture, not units.
# Matched on canonical keys.
NON_UNIT_CLASSES = {
    "kingtower", "queentower", "cannoneertower", "daggerduchesstower",
    "daggerduchesstowerbar", "towerbar", "kingtowerbar", "bar", "barlevel",
    "clock", "emote", "text", "elixir", "selected", "skeletonkingbar", "dirt",
    "evolutionsymbol", "icespiritevolutionsymbol", "skeletonkingskill",
    "teslaevolutionshock", "padbelong",
    # Projectiles: they move fast, belong to a unit already on the board and
    # would otherwise be tracked as phantom units.
    "bomb", "axe", "goblinball",
}
TOWER_CLASSES = {"kingtower", "queentower", "cannoneertower", "daggerduchesstower"}

# Detector names that differ from the card/unit table beyond punctuation.
ALIASES = {
    "elixirgolembig": "elixirgolemlarge",
    "elixirgolemmid": "elixirgolemmedium",
    "phoenixbig": "phoenixlarge",
    "goblinbrawler": "brawler",
}
# Note: plain "hog" (in both vocabularies) is the cursed hog a Mother Witch
# creates, not a Hog Rider ("hog-rider") or Royal Hog ("royal-hog").

# Units that appear on their own without a card being played: spawned by a
# building, by another unit, or on death. Key: spawned unit; value: the units
# that can produce it. Used to avoid counting these as plays.
SPAWNED_BY = {
    "skeleton": {"tombstone", "witch", "skeletonking", "graveyard", "skeletonbarrel", "giantskeleton"},
    "speargoblin": {"goblinhut", "goblingiant"},
    "goblin": {"goblindrill", "goblinbarrel"},
    "barbarian": {"barbarianhut", "barbarianbarrel"},
    "firespirit": {"furnace"},
    "golemite": {"golem"},
    "lavapup": {"lavahound"},
    "elixirgolemmedium": {"elixirgolemlarge"},
    "elixirgolemsmall": {"elixirgolemmedium"},
    "phoenixegg": {"phoenixlarge"},
    "phoenixsmall": {"phoenixegg"},
    "bat": {"nightwitch"},
    "hog": {"motherwitch"},
}

# How many units a card puts on the board. Used only to break ties when a
# unit belongs to several cards (a Barbarian could be Barbarians, Barbarian
# Barrel or Barbarian Hut). Balance changes move these, so treat them as hints.
UNITS_PER_CARD = {
    "skeletons": 3, "skeleton_army": 15, "archers": 2, "barbarians": 5,
    "minions": 3, "minion_horde": 6, "goblins": 4, "spear_goblins": 3,
    "goblin_gang": 6, "guards": 3, "bats": 5, "royal_hogs": 4,
    "royal_recruits": 6, "three_musketeers": 3, "elite_barbarians": 2,
    "rascals": 3, "wall_breakers": 2, "skeleton_dragons": 2, "zappies": 3,
    "barbarian_barrel": 1, "goblin_barrel": 3,
}


def key(name: str) -> str:
    """Canonical key: lower case, no punctuation, evolution suffix dropped.

    ``skeleton-evolution`` and ``archers_ev1`` are the same card as the base
    version for every purpose here (cost, side, cycle), so they share a key.
    UI classes such as ``ice-spirit-evolution-symbol`` are checked first so
    they are not mistaken for the unit they decorate.
    """
    name = name.lower()
    flat = re.sub(r"[^a-z0-9]", "", name)
    if flat in NON_UNIT_CLASSES:
        return flat
    name = re.sub(r"[-_ ](evolution|evo|ev\d)$", "", name)
    flat = re.sub(r"[^a-z0-9]", "", name)
    return ALIASES.get(flat, flat)


@dataclass(frozen=True)
class UnitInfo:
    name: str
    building: bool
    flying: bool
    targets: Optional[str]


@dataclass(frozen=True)
class CardInfo:
    name: str
    cost: int
    spell: bool                    # can be placed anywhere (spells, Miner-likes)
    units: tuple
    units_per_card: int = 1


@dataclass
class Catalog:
    units: Dict[str, UnitInfo] = field(default_factory=dict)   # by key
    cards: Dict[str, CardInfo] = field(default_factory=dict)   # by key
    cards_by_unit: Dict[str, List[str]] = field(default_factory=dict)  # unit key -> card keys

    # ------------------------------------------------------------- lookups

    def unit(self, name: str) -> Optional[UnitInfo]:
        return self.units.get(key(name))

    def card(self, name: str) -> Optional[CardInfo]:
        return self.cards.get(key(name))

    def is_unit_class(self, detector_name: str) -> bool:
        k = key(detector_name)
        return k not in NON_UNIT_CLASSES and not k.startswith("pad")

    def is_tower_class(self, detector_name: str) -> bool:
        return key(detector_name) in TOWER_CLASSES

    def cards_for(self, detector_name: str) -> List[CardInfo]:
        """Cards that could have put this object on the board.

        A spell effect detected on the board (``fireball``) maps to its own
        card; a unit maps to every card that contains it.
        """
        k = key(detector_name)
        found = [self.cards[c] for c in self.cards_by_unit.get(k, [])]
        if not found and k in self.cards:
            found = [self.cards[k]]
        return found

    def cost(self, card_name: str) -> Optional[int]:
        info = self.card(card_name)
        return info.cost if info else None

    def units_of(self, cards: Iterable[str]) -> Set[str]:
        """Unit keys that a set of cards can put on the board, spawns included."""
        out: Set[str] = set()
        for name in cards:
            info = self.card(name)
            if info is None:
                continue
            out.add(key(info.name))
            out.update(key(u) for u in info.units)
        # Anything one of those units can spawn also belongs to the same side.
        changed = True
        while changed:
            changed = False
            for child, parents in SPAWNED_BY.items():
                if child not in out and parents & out:
                    out.add(child)
                    changed = True
        return out


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    units_raw = json.loads((ASSETS / "units.json").read_text(encoding="utf-8"))
    cards_raw = json.loads((ASSETS / "cards.json").read_text(encoding="utf-8"))
    catalog = Catalog()
    for name, data in units_raw.items():
        catalog.units[key(name)] = UnitInfo(
            name=name, building=data["category"] == "building",
            flying=bool(data["flying"]), targets=data["targets"])
    for name, data in cards_raw.items():
        if name == "blank":            # the empty hand slot, not a card
            continue
        info = CardInfo(name=name, cost=int(data["cost"]),
                        spell=bool(data["target_anywhere"]),
                        units=tuple(data["units"]),
                        units_per_card=UNITS_PER_CARD.get(name, 1))
        catalog.cards[key(name)] = info
        for unit in data["units"]:
            catalog.cards_by_unit.setdefault(key(unit), []).append(key(name))
    return catalog
