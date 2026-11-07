"""Train a small ally/enemy classifier on unit crops and export it to ONNX.

    python training/prepare_katacr.py --src ... --out datasets/cr_yolo      # writes sides.json
    pip install torch onnx
    python training/train_side_classifier.py --data datasets/cr_yolo --out models/side.onnx
    python -m crbot live ... --side-model models/side.onnx

Why a separate model: the detector says *what* a unit is, and side is a
different question. Colour does not answer it reliably. Hasty-CR measured a
red-vs-blue rule at 52.8% on units in our own half. The cue a CNN can learn
is orientation (our units face up the screen, theirs face down) together with
the team-coloured level badge. In crbot, this model is the third line of
evidence, after the detector's own side output and the deck and hand rules
(see ``crbot/vision/side.py``).

Crops are 32x32 RGB scaled to [0, 1], NCHW. Labels: 0 = ally (bottom
player), 1 = enemy. Scenery, UI and towers are excluded: towers never leave
their half, so they would inflate the accuracy without teaching anything.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

CROP = 32
EXCLUDE = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "tower-bar",
    "king-tower-bar", "dagger-duchess-tower-bar", "bar", "bar-level", "text", "clock",
    "emote", "elixir", "selected", "skeleton-king-bar", "dirt", "evolution-symbol",
    "ice-spirit-evolution-symbol", "pad_belong",
}


def class_names(yaml_path: Path) -> dict:
    names = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\d+):\s*(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2)
    return names


def build(data: Path, split: str, names: dict, sides: dict, limit: int = 0, pad: float = 0.1):
    import cv2

    xs, ys = [], []
    stems = sorted(p.stem for p in (data / "labels" / split).glob("*.txt"))
    if limit:
        random.Random(0).shuffle(stems)
        stems = stems[:limit]
    for stem in stems:
        frame_sides = sides.get(f"{split}/{stem}")
        image_path = next((data / "images" / split).glob(f"{stem}.*"), None)
        if not frame_sides or image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        rows = (data / "labels" / split / f"{stem}.txt").read_text().split("\n")
        for row, side in zip([r for r in rows if r.strip()], frame_sides):
            cls, cx, cy, bw, bh = row.split()
            name = names.get(int(cls), "")
            if name in EXCLUDE or name.startswith("pad_"):
                continue
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x0, y0 = int(max(0, cx - bw * (0.5 + pad))), int(max(0, cy - bh * (0.5 + pad)))
            x1, y1 = int(min(w, cx + bw * (0.5 + pad))), int(min(h, cy + bh * (0.5 + pad)))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
            xs.append(crop[:, :, ::-1].transpose(2, 0, 1))      # RGB, CHW
            ys.append(int(side))
    return np.stack(xs).astype(np.float32) / 255.0, np.array(ys, dtype=np.int64)


def model():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("datasets/cr_yolo"))
    parser.add_argument("--out", type=Path, default=Path("models/side.onnx"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="use only this many frames (quick runs)")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    names = class_names(args.data / "cr.yaml")
    sides = json.loads((args.data / "sides.json").read_text())
    x_train, y_train = build(args.data, "train", names, sides, args.limit)
    x_val, y_val = build(args.data, "val", names, sides, args.limit // 4 if args.limit else 0)
    print(f"train {len(y_train)} crops ({y_train.mean():.0%} enemy), val {len(y_val)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model().to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=2e-3, total_steps=args.epochs * ((len(y_train) + args.batch - 1) // args.batch))
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    xv, yv = torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device)
    best = 0.0
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(yt))
        for start in range(0, len(yt), args.batch):
            idx = order[start:start + args.batch]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            if random.random() < 0.5:                      # left-right flip keeps the side
                xb = xb.flip(-1)
            optimiser.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimiser.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            correct = sum(int((net(xv[i:i + 1024]).argmax(1) == yv[i:i + 1024]).sum())
                          for i in range(0, len(yv), 1024))
        accuracy = correct / max(1, len(yv))
        print(f"epoch {epoch + 1}: loss {loss.item():.3f} val acc {accuracy:.3f}")
        if accuracy >= best:
            best = accuracy
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(net.cpu().eval(), torch.zeros(1, 3, CROP, CROP), str(args.out),
                              input_names=["crop"], output_names=["logits"],
                              dynamic_axes={"crop": {0: "batch"}, "logits": {0: "batch"}})
            net.to(device)
    print(f"best val accuracy {best:.3f}; model at {args.out} (labels: 0 = ally, 1 = enemy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
