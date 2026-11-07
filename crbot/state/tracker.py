"""Turn per-frame detections into persistent units with identities and velocity.

Detections flicker. A unit is missed for a frame, re-found a tile away, or
labelled the other side for one frame in a crowded fight. The tracker
smooths that out:

* Detections are matched to existing tracks of the same class with the
  Hungarian algorithm, on distance from where each track is *predicted* to
  be. A Hog Rider moves about 2 tiles/s, so matching to its last position
  goes wrong at low frame rates.
* A track is **confirmed** after `confirm_hits` matches, so one-frame false
  positives never reach the state.
* Side is a running vote over every frame the unit was seen, so a single
  misread does not flip it.
* Tracks are dropped after `max_age` seconds unseen (longer for buildings,
  which never move and are often covered by fighting units).

The tracker also reports which tracks were confirmed this frame. Those are
the "new unit appeared" events that play detection is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.types import Detection

MAX_SPEED = 4.0          # tiles/second; fastest units are ~3


@dataclass
class Track:
    id: int
    key: str
    name: str
    x: float
    y: float
    born_x: float
    born_y: float
    first_seen: float
    last_seen: float
    conf: float
    building: bool = False
    flying: bool = False
    kind: str = "unit"
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    confirmed: bool = False
    side_votes: List[float] = field(default_factory=lambda: [0.0, 0.0])
    box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def side(self) -> int:
        return 0 if self.side_votes[0] >= self.side_votes[1] else 1

    @property
    def side_confidence(self) -> float:
        total = sum(self.side_votes)
        return max(self.side_votes) / total if total else 0.0

    def predict(self, t: float) -> Tuple[float, float]:
        dt = max(0.0, t - self.last_seen)
        return self.x + self.vx * dt, self.y + self.vy * dt


class Tracker:
    """See the module docstring.

    ``reacquire`` seconds after a track is dropped, a new detection of the
    same class near where it was heading revives it with its identity and
    side votes. Without this, a unit hidden for two seconds behind a spell
    comes back as a "new" unit. Its side is then re-guessed and it is
    counted as a second play.
    """

    def __init__(self, gate: float = 2.0, confirm_hits: int = 2, max_age: float = 1.5,
                 building_max_age: float = 4.0, smoothing: float = 0.5, reacquire: float = 3.0):
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_age = max_age
        self.building_max_age = building_max_age
        self.smoothing = smoothing
        self.reacquire = reacquire
        self.tracks: Dict[int, Track] = {}
        self.lost: Dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.lost.clear()
        self._next_id = 1

    def _revive(self, det: Detection, t: float) -> Optional[Track]:
        best, best_d = None, None
        for tr in self.lost.values():
            if tr.key != det.key or not tr.confirmed:
                continue
            px, py = tr.predict(t)
            allowed = self.gate + MAX_SPEED * (t - tr.last_seen) * 0.5
            d = float(np.hypot(det.tile[0] - px, det.tile[1] - py))
            if d <= allowed and (best_d is None or d < best_d):
                best, best_d = tr, d
        if best is not None:
            del self.lost[best.id]
            self.tracks[best.id] = best
        return best

    def _expired(self, track: Track, t: float) -> bool:
        limit = self.building_max_age if track.building else self.max_age
        return t - track.last_seen > limit

    def update(self, detections: Sequence[Detection], t: float,
               side_hint: Optional[Callable[[Detection], Optional[int]]] = None
               ) -> Tuple[List[Track], List[Track], List[Track]]:
        """Returns (live confirmed tracks, newly confirmed, removed).

        ``side_hint`` is consulted for brand-new tracks only. It returns a
        side when there is strong outside evidence (e.g. we just played that
        card), which outweighs a per-frame side guess.
        """
        from scipy.optimize import linear_sum_assignment

        removed = [tr for tr in self.tracks.values() if self._expired(tr, t)]
        for tr in removed:
            del self.tracks[tr.id]
            self.lost[tr.id] = tr
        for tr_id in [i for i, tr in self.lost.items() if t - tr.last_seen > self.reacquire]:
            del self.lost[tr_id]

        newly_confirmed: List[Track] = []
        by_key: Dict[str, List[int]] = {}
        for i, det in enumerate(detections):
            by_key.setdefault(det.key, []).append(i)

        for k, det_ids in by_key.items():
            candidates = [tr for tr in self.tracks.values() if tr.key == k]
            matched_dets = set()
            if candidates:
                cost = np.full((len(det_ids), len(candidates)), 1e6)
                for a, di in enumerate(det_ids):
                    det = detections[di]
                    for b, tr in enumerate(candidates):
                        px, py = tr.predict(t)
                        allowed = self.gate + MAX_SPEED * max(0.0, t - tr.last_seen) * 0.5
                        d = float(np.hypot(det.tile[0] - px, det.tile[1] - py))
                        if d <= allowed:
                            cost[a, b] = d
                rows, cols = linear_sum_assignment(cost)
                for a, b in zip(rows, cols):
                    if cost[a, b] >= 1e6:
                        continue
                    det = detections[det_ids[a]]
                    tr = candidates[b]
                    self._update_track(tr, det, t)
                    matched_dets.add(det_ids[a])
                    if not tr.confirmed and tr.hits >= self.confirm_hits:
                        tr.confirmed = True
                        newly_confirmed.append(tr)
            for di in det_ids:
                if di in matched_dets:
                    continue
                det = detections[di]
                revived = self._revive(det, t)
                if revived is not None:
                    self._update_track(revived, det, t)
                    continue
                tr = self._new_track(det, t)
                hint = side_hint(det) if side_hint else None
                if hint is not None:
                    tr.side_votes[hint] += 5.0
                if self.confirm_hits <= 1:
                    tr.confirmed = True
                    newly_confirmed.append(tr)

        live = [tr for tr in self.tracks.values() if tr.confirmed]
        return live, newly_confirmed, removed

    def _new_track(self, det: Detection, t: float) -> Track:
        tr = Track(id=self._next_id, key=det.key, name=det.name, x=det.tile[0], y=det.tile[1],
                   born_x=det.tile[0], born_y=det.tile[1], first_seen=t, last_seen=t,
                   conf=det.conf, building=det.building, flying=det.flying, kind=det.kind,
                   box=det.box)
        tr.side_votes[det.side] += det.side_conf
        self.tracks[tr.id] = tr
        self._next_id += 1
        return tr

    def _update_track(self, tr: Track, det: Detection, t: float) -> None:
        dt = t - tr.last_seen
        if dt > 1e-3 and not tr.building:
            vx = (det.tile[0] - tr.x) / dt
            vy = (det.tile[1] - tr.y) / dt
            a = self.smoothing
            tr.vx = float(np.clip(a * tr.vx + (1 - a) * vx, -MAX_SPEED, MAX_SPEED))
            tr.vy = float(np.clip(a * tr.vy + (1 - a) * vy, -MAX_SPEED, MAX_SPEED))
        tr.x, tr.y = det.tile
        tr.last_seen = t
        tr.hits += 1
        tr.conf = max(det.conf, 0.8 * tr.conf)
        tr.box = det.box
        tr.side_votes[det.side] += det.side_conf
