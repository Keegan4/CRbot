"""Train the arena unit detector (Ultralytics YOLO) on the prepared KataCR data.

    pip install ultralytics
    python training/train_detector.py --data datasets/cr_yolo/cr.yaml --model yolov8s.pt
    python -m crbot live --source adb --detector runs/detect/cr/weights/best.pt --deck ...

Reference point: Hasty-CR trained YOLOv8s on this dataset (201 classes) to
0.959 mAP50 (precision 0.949, recall 0.926). That took about 6 hours on an
RTX 4070 Ti SUPER (3 hours for the ``n`` model). mAP becomes meaningful
around epoch 10, so a run that is going nowhere shows early.

Augmentation choices, and why:

* colour jitter **on**: arena skins and the red/blue team tints vary.
* left-right flip **on**: the arena is mirror-symmetric.
* up-down flip, rotation, shear, perspective **off**: the camera never
  changes, and up/down carries meaning (whose side a unit is on, which way
  it faces).

The KataCR dataset's author ended up with two detectors split by object size
(a Skeleton is ~20 px, a King Tower ~200 px). Start with one, measure it, and
split only if small units are the weak spot.

``--export-onnx`` also writes ``<run>/weights/best.onnx`` with NMS built in,
plus ``best.names.json``, so the model runs with only ``onnxruntime``. That
output format depends on the Ultralytics version (``nms=True`` needs 8.3+).
Check it with ``python -m crbot live --detector .../best.onnx`` before
relying on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/cr_yolo/cr.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="starting weights (yolov8n.pt is ~2x faster)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. 0, or cpu")
    parser.add_argument("--name", default="cr")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    if not Path(args.data).exists():
        print(f"missing {args.data}; run training/prepare_katacr.py first")
        return 1
    model = YOLO(args.model)
    kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, project=args.project, name=args.name, exist_ok=True,
        resume=args.resume, patience=args.patience,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, close_mosaic=10, plots=True, val=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    model.train(**kwargs)

    weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"best weights: {weights}")
    if args.export_onnx and weights.exists():
        best = YOLO(str(weights))
        onnx_path = Path(best.export(format="onnx", imgsz=args.imgsz, nms=True))
        names = [best.names[i] for i in sorted(best.names)]
        onnx_path.with_suffix(".names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
        print(f"exported {onnx_path} (+ class names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
