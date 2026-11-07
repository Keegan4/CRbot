"""Convert the KataCR Clash Royale detection dataset into an Ultralytics YOLO layout.

Dataset (MIT): https://github.com/wty-yy/Clash-Royale-Detection-Dataset
~6,900 hand-labelled arena frames, 201 classes. Clone it (about 640 MB):

    git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset datasets/katacr
    python training/prepare_katacr.py --src datasets/katacr/images/part2 --out datasets/cr_yolo

Two things in the source format need fixing, and both fail silently if missed:

* **Twelve columns per label line**: ``class cx cy w h side`` plus six zeros.
  Ultralytics wants exactly five. Column six is the *side* (0 = the bottom
  player = us, 1 = the opponent). That is worth keeping:

  - ``--side drop`` (default): 201 classes. Sides are saved to
    ``sides.json`` for ``train_side_classifier.py``.
  - ``--side fold``: 402 classes, ``name@ally`` / ``name@enemy``. The
    detector then predicts the side itself; crbot reads the suffix.

* **Labels next to images**. Ultralytics looks for labels by swapping
  ``images/`` for ``labels/`` in the path, so the tree is rebuilt. Images are
  hard-linked, not copied.

The frames are crops of the *arena* (KataCR cut the screen to the 18x32 tile
area plus a small margin). crbot crops live frames the same way before
running a model trained on them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List


def read_names(yaml_path: Path) -> Dict[int, str]:
    names, inside = {}, False
    for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("names:"):
            inside = True
            continue
        if inside:
            match = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2)
            elif line.strip() and not line.startswith(" "):
                break
    return names


def flat_name(relative: str) -> str:
    """``./WTY_20240305/1/00810.jpg`` -> ``WTY_20240305_1_00810``. Every episode
    folder restarts numbering, so the path has to be part of the name."""
    return relative.lstrip("./").rsplit(".", 1)[0].replace("/", "_").replace("\\", "_")


def convert(src: Path, out: Path, split: str, listing: Path, fold: bool,
            sides: Dict[str, List[int]]) -> tuple:
    image_dir, label_dir = out / "images" / split, out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    frames = boxes = 0
    for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
        relative = line.strip().split()[0] if line.strip() else ""
        if not relative:
            continue
        image = (src / relative.lstrip("./")).resolve()
        label = image.with_suffix(".txt")
        if not image.exists() or not label.exists():
            continue
        stem = flat_name(relative)
        target = image_dir / f"{stem}{image.suffix}"
        if not target.exists():
            try:
                os.link(image, target)
            except OSError:
                shutil.copyfile(image, target)
        rows, frame_sides = [], []
        for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                values = [float(v) for v in parts[1:5]]
            except ValueError:
                continue
            # One bad coordinate makes Ultralytics drop the whole file.
            if not all(0.0 <= v <= 1.0 for v in values):
                continue
            side = int(parts[5]) if len(parts) > 5 and parts[5] in ("0", "1") else 0
            out_cls = cls * 2 + side if fold else cls
            rows.append(f"{out_cls} " + " ".join(parts[1:5]))
            frame_sides.append(side)
        (label_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        sides[f"{split}/{stem}"] = frame_sides
        frames += 1
        boxes += len(rows)
    return frames, boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="the dataset's images/part2 folder")
    parser.add_argument("--out", type=Path, default=Path("datasets/cr_yolo"))
    parser.add_argument("--side", choices=("drop", "fold"), default="drop")
    args = parser.parse_args()

    names = read_names(args.src / "ClashRoyale_detection.yaml")
    if not names:
        print("could not read class names from ClashRoyale_detection.yaml")
        return 1
    sides: Dict[str, List[int]] = {}
    for split in ("train", "val"):
        listing = args.src / f"{split}_annotation.txt"
        if not listing.exists():
            print(f"missing {listing}")
            return 1
        frames, boxes = convert(args.src, args.out, split, listing, args.side == "fold", sides)
        print(f"{split:5s} {frames:5d} frames {boxes:7d} boxes")

    if args.side == "fold":
        out_names = {}
        for index, name in names.items():
            out_names[index * 2] = f"{name}@ally"
            out_names[index * 2 + 1] = f"{name}@enemy"
    else:
        out_names = names
    lines = [f"path: {args.out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {out_names[i]}" for i in sorted(out_names)]
    (args.out / "cr.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "sides.json").write_text(json.dumps(sides), encoding="utf-8")
    print(f"{len(out_names)} classes -> {args.out / 'cr.yaml'}; sides -> {args.out / 'sides.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
