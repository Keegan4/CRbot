"""Arena object detectors.

Two backends, both returning `RawBox` lists in full-frame pixels:

* `OnnxEndToEndDetector` runs an ONNX model with non-maximum suppression
  built in, which outputs ``[x0, y0, x1, y1, conf, class]`` rows. This is
  the format of ClashRoyaleBuildABot's pretrained unit detector, so
  ``python -m crbot fetch-models`` gives a working detector before you
  train anything. It needs only ``onnxruntime``.
* `UltralyticsDetector` runs a YOLO ``.pt`` (or exported) model trained with
  ``training/train_detector.py`` on the KataCR dataset. It needs
  ``ultralytics``.

Each backend crops the frame before inference, to the region its training
images covered. The two training sets cover different regions, so the crop
is part of the backend's configuration and not of the caller's.

A class name may end in ``@ally`` or ``@enemy`` when the detector was trained
with the side folded into the class (``training/prepare_katacr.py --side
fold``). The suffix is split off here and becomes `RawBox.side`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..catalog import ASSETS
from ..geometry import Rect

ALLY, ENEMY = 0, 1


@dataclass
class RawBox:
    name: str
    conf: float
    box: Tuple[float, float, float, float]     # x0, y0, x1, y1 in frame pixels
    side: Optional[int] = None                 # set only by side-aware models


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[RawBox]: ...


def split_side(name: str) -> Tuple[str, Optional[int]]:
    if name.endswith("@ally"):
        return name[:-5], ALLY
    if name.endswith("@enemy"):
        return name[:-6], ENEMY
    return name, None


def letterbox(image: np.ndarray, height: int, width: int,
              fill: int = 114) -> Tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to (height, width).

    Returns the padded image, the scale applied, and the left/top padding.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    out = np.full((height, width, 3), fill, dtype=np.uint8)
    out[top:top + new_h, left:left + new_w] = resized
    return out, scale, left, top


class OnnxEndToEndDetector:
    """ONNX detector whose output is ``(N, 6)``: x0, y0, x1, y1, conf, class.

    ``crop`` is the part of the frame fed to the model. The default is the
    band ClashRoyaleBuildABot crops to (5%-80% of the height, full width).
    """

    def __init__(self, model_path: Path, class_names: Sequence[str],
                 crop: Rect = Rect(0.0, 0.05, 1.0, 0.75), min_conf: float = 0.35,
                 providers: Optional[Sequence[str]] = None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        wanted = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=list(wanted))
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_h, self.input_w = (int(v) for v in model_input.shape[2:4])
        self.dtype = np.float16 if "float16" in model_input.type else np.float32
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = list(class_names)
        self.crop = crop
        self.min_conf = min_conf

    @classmethod
    def buildabot(cls, model_path: Path, **kwargs) -> "OnnxEndToEndDetector":
        names = json.loads((ASSETS / "buildabot_classes.json").read_text(encoding="utf-8"))
        return cls(model_path, names, **kwargs)

    def detect(self, image: np.ndarray) -> List[RawBox]:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.crop.to_pixels(w, h)
        region = image[y0:y1, x0:x1, ::-1]                       # BGR -> RGB
        padded, scale, left, top = letterbox(np.ascontiguousarray(region), self.input_h, self.input_w)
        tensor = padded.transpose(2, 0, 1)[None].astype(self.dtype) / 255.0
        rows = self.session.run([self.output_name], {self.input_name: tensor.astype(self.dtype)})[0]
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, rows.shape[-1])
        out: List[RawBox] = []
        for bx0, by0, bx1, by1, conf, cls in rows[:, :6]:
            if conf < self.min_conf:
                continue
            index = int(cls)
            if not 0 <= index < len(self.class_names):
                continue
            name, side = split_side(self.class_names[index])
            box = ((bx0 - left) / scale + x0, (by0 - top) / scale + y0,
                   (bx1 - left) / scale + x0, (by1 - top) / scale + y0)
            out.append(RawBox(name, float(conf), tuple(float(v) for v in box), side))
        return out


class UltralyticsDetector:
    """A YOLO model from ``ultralytics`` (``.pt``, or an exported format).

    ``crop`` defaults to the arena rectangle plus a small margin, matching the
    KataCR training images, which are cut to the arena.
    """

    def __init__(self, weights: Path, crop: Optional[Rect] = None, min_conf: float = 0.35,
                 imgsz: int = 640, device: Optional[str] = None, max_det: int = 100):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = dict(self.model.names)
        self.crop = crop
        self.min_conf = min_conf
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det

    def detect(self, image: np.ndarray) -> List[RawBox]:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.crop.to_pixels(w, h) if self.crop else (0, 0, w, h)
        region = np.ascontiguousarray(image[y0:y1, x0:x1])
        kwargs = dict(imgsz=self.imgsz, conf=self.min_conf, verbose=False, max_det=self.max_det)
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(region, **kwargs)[0]
        boxes = result.boxes
        out: List[RawBox] = []
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (bx0, by0, bx1, by1), c, k in zip(xyxy, conf, cls):
            name, side = split_side(str(self.names.get(int(k), k)))
            out.append(RawBox(name, float(c), (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0), side))
        return out


class NullDetector:
    """No unit detection; the rest of the state is still read."""

    def detect(self, image: np.ndarray) -> List[RawBox]:
        return []


def load_detector(spec: Optional[str], arena_crop: Optional[Rect] = None,
                  min_conf: float = 0.35, device: Optional[str] = None) -> Detector:
    """Pick a backend from a path or a keyword.

    ``None`` / ``"none"``      no unit detection
    ``"buildabot"``            models/buildabot/units_M_480x352.onnx
    ``*.onnx``                 end-to-end ONNX model; class names from a
                               sibling ``<model>.names.json`` if present,
                               otherwise BuildABot's class list
    anything else              an ultralytics model
    """
    if spec in (None, "", "none"):
        return NullDetector()
    if spec == "buildabot":
        path = Path("models/buildabot/units_M_480x352.onnx")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m crbot fetch-models` first")
        return OnnxEndToEndDetector.buildabot(path, min_conf=min_conf)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".onnx":
        names_file = path.with_suffix(".names.json")
        if names_file.exists():
            names = json.loads(names_file.read_text(encoding="utf-8"))
            crop = arena_crop or Rect(0.0, 0.0, 1.0, 1.0)
            return OnnxEndToEndDetector(path, names, crop=crop, min_conf=min_conf)
        return OnnxEndToEndDetector.buildabot(path, min_conf=min_conf)
    return UltralyticsDetector(path, crop=arena_crop, min_conf=min_conf, device=device)
