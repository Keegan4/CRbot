"""Decide whether a detected unit is ours or the opponent's.

Unit tint is a weak cue. Hasty-CR measured a blue-vs-red pixel rule at 52.8%
on units in our own half, the case that matters most. So the evidence is
used in this order, strongest first:

1. **The detector says so.** Models trained with ``--side fold`` predict it.
2. **Deck rule.** When our deck is known, a unit none of our cards can make
   is the opponent's. This is certain, and it settles most cases
   (ClashRoyaleBuildABot uses the same rule).
3. **Crop classifier.** A small model on the unit's crop. It mostly learns
   which way the sprite faces: ours walk up the screen, theirs walk down.
4. **Which half it stands in.** Right most of the time, wrong exactly when a
   push crosses the river. Used only when nothing better is available, and
   reported with low confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..catalog import Catalog, key
from ..geometry import RIVER_Y
from .detector import ALLY, ENEMY


class OnnxCropClassifier:
    """Two-class (ally, enemy) ONNX classifier on a unit crop.

    Accepts NHWC or NCHW input, RGB scaled to [0, 1]. Works with
    ClashRoyaleBuildABot's ``side.onnx`` (16x16 NHWC) and with models
    exported by ``training/train_side_classifier.py``.
    """

    def __init__(self, model_path: Path, labels: Sequence[str] = ("ally", "enemy")):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = list(model_input.shape)
        self.nhwc = shape[-1] == 3
        self.size = int(shape[1] if self.nhwc else shape[2])
        self.dtype = np.float16 if "float16" in model_input.type else np.float32
        self.labels = [label.lower() for label in labels]

    def __call__(self, crops: Sequence[np.ndarray]) -> List[Tuple[int, float]]:
        import cv2

        if not crops:
            return []
        batch = []
        for crop in crops:
            if crop.size == 0:
                crop = np.zeros((self.size, self.size, 3), np.uint8)
            rgb = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
            batch.append(rgb.astype(np.float32) / 255.0)
        x = np.stack(batch)
        if not self.nhwc:
            x = x.transpose(0, 3, 1, 2)
        out: List[Tuple[int, float]] = []
        # Some exported models have a fixed batch size of one.
        for row in x:
            logits = np.asarray(self.session.run(None, {self.input_name: row[None].astype(self.dtype)})[0],
                                dtype=np.float32).reshape(-1)
            if not (np.all(logits >= 0) and abs(float(logits.sum()) - 1.0) < 1e-3):
                exp = np.exp(logits - logits.max())
                logits = exp / exp.sum()
            index = int(np.argmax(logits))
            side = ALLY if self.labels[index].startswith("ally") else ENEMY
            out.append((side, float(logits[index])))
        return out


class SideResolver:
    def __init__(self, catalog: Catalog, my_deck: Optional[Iterable[str]] = None,
                 classifier: Optional[OnnxCropClassifier] = None, min_conf: float = 0.6):
        self.catalog = catalog
        self.classifier = classifier
        self.min_conf = min_conf
        self.my_units: Set[str] = catalog.units_of(my_deck) if my_deck else set()

    def set_deck(self, deck: Iterable[str]) -> None:
        self.my_units = self.catalog.units_of(deck)

    def resolve(self, image: np.ndarray, names: Sequence[str],
                boxes: Sequence[Tuple[float, float, float, float]],
                tiles_y: Sequence[float], given: Sequence[Optional[int]]
                ) -> List[Tuple[int, float, str]]:
        """(side, confidence, method) for each detection."""
        results: List[Optional[Tuple[int, float, str]]] = [None] * len(names)
        pending: List[int] = []
        for i, (name, side) in enumerate(zip(names, given)):
            if side is not None:
                results[i] = (side, 1.0, "detector")
            elif self.my_units and key(name) not in self.my_units:
                results[i] = (ENEMY, 1.0, "deck")
            else:
                pending.append(i)

        if pending and self.classifier is not None:
            h, w = image.shape[:2]
            crops = []
            for i in pending:
                x0, y0, x1, y1 = boxes[i]
                pad_x, pad_y = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
                cx0, cy0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
                cx1, cy1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y))
                crops.append(image[cy0:cy1, cx0:cx1])
            for i, (side, conf) in zip(pending, self.classifier(crops)):
                if conf >= self.min_conf:
                    results[i] = (side, conf, "classifier")

        for i in pending:
            if results[i] is None:
                side = ALLY if tiles_y[i] >= RIVER_Y else ENEMY
                results[i] = (side, 0.5, "half")
        return results  # type: ignore[return-value]
