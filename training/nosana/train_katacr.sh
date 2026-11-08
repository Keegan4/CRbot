#!/usr/bin/env bash
# Train the crbot unit detector on the KataCR dataset on a rented GPU (Nosana),
# then serve the results over HTTP so they can be downloaded.
#
# Self-contained: it does not need the crbot repository, only the image
# docker.io/ultralytics/ultralytics (which has PyTorch, CUDA, git and
# Ultralytics). `job.json` embeds this file; regenerate it with
#   python training/nosana/make_job.py
#
# Settings (environment variables):
#   EPOCHS (120)  MODEL (yolov8m.pt)  IMGSZ (640)  BATCH (16)
#   SERVE_MINUTES (60)  how long to keep the download server up
#   WORK (/tmp/work)    STAGE=prepare stops after the dataset conversion
set -euo pipefail

EPOCHS=${EPOCHS:-120}
MODEL=${MODEL:-yolov8m.pt}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
SERVE_MINUTES=${SERVE_MINUTES:-60}
WORK=${WORK:-/tmp/work}
DATASET=${DATASET:-$WORK/katacr/images/part2}
mkdir -p "$WORK" "$WORK/out"
cd "$WORK"

echo "== $(date -u +%FT%TZ) downloading the KataCR dataset (images/part2 only)"
if [ ! -d "$DATASET" ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/wty-yy/Clash-Royale-Detection-Dataset katacr
  (cd katacr && git sparse-checkout set images/part2)
fi

echo "== $(date -u +%FT%TZ) converting labels to YOLO format"
python - "$DATASET" "$WORK/cr_yolo" <<'PY'
# Same conversion as training/prepare_katacr.py (--side drop): keep the
# first five label columns, rebuild images/ and labels/ trees.
import os, re, shutil, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
names = {}
inside = False
for line in (src / "ClashRoyale_detection.yaml").read_text(errors="replace").splitlines():
    if line.startswith("names:"):
        inside = True
        continue
    if inside:
        m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2)
        elif line.strip() and not line.startswith(" "):
            break
for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    frames = boxes = 0
    for line in (src / f"{split}_annotation.txt").read_text(errors="replace").splitlines():
        rel = line.strip().split()[0] if line.strip() else ""
        if not rel:
            continue
        image = src / rel.lstrip("./")
        label = image.with_suffix(".txt")
        if not image.exists() or not label.exists():
            continue
        stem = rel.lstrip("./").rsplit(".", 1)[0].replace("/", "_")
        target = out / "images" / split / f"{stem}{image.suffix}"
        if not target.exists():
            try:
                os.link(image, target)
            except OSError:
                shutil.copyfile(image, target)
        rows = []
        for raw in label.read_text(errors="replace").splitlines():
            p = raw.split()
            if len(p) < 5:
                continue
            try:
                vals = [float(v) for v in p[1:5]]
                int(p[0])
            except ValueError:
                continue
            if all(0.0 <= v <= 1.0 for v in vals):
                rows.append(" ".join(p[:5]))
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + "\n")
        frames += 1
        boxes += len(rows)
    print(f"{split}: {frames} frames, {boxes} boxes", flush=True)
lines = [f"path: {out}", "train: images/train", "val: images/val", "names:"]
lines += [f"  {i}: {names.get(i, f'class_{i}')}" for i in range(max(names) + 1)]
(out / "cr.yaml").write_text("\n".join(lines) + "\n")
print(f"{len(names)} classes", flush=True)
PY

if [ "${STAGE:-}" = "prepare" ]; then
  echo "== STAGE=prepare: stopping before training"
  exit 0
fi

nvidia-smi || true
echo "== $(date -u +%FT%TZ) training $MODEL for up to $EPOCHS epochs"
yolo detect train data="$WORK/cr_yolo/cr.yaml" model="$MODEL" epochs="$EPOCHS" imgsz="$IMGSZ" \
  batch="$BATCH" patience=30 workers=8 project="$WORK/runs" name=cr exist_ok=True \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0 shear=0 perspective=0 flipud=0 fliplr=0.5 \
  mosaic=1.0 close_mosaic=10 plots=True

RUN="$WORK/runs/cr"
echo "== $(date -u +%FT%TZ) exporting ONNX"
yolo export model="$RUN/weights/best.pt" format=onnx imgsz="$IMGSZ" nms=True || \
  echo "ONNX export failed; best.pt is still usable with --detector best.pt"
python - "$RUN/weights/best.pt" <<'PY'
import json, sys
from pathlib import Path
from ultralytics import YOLO
m = YOLO(sys.argv[1])
names = [m.names[i] for i in sorted(m.names)]
Path(sys.argv[1]).with_suffix(".names.json").write_text(json.dumps(names, indent=1))
PY

echo "== final validation metrics"
tail -n 3 "$RUN/results.csv" || true
tar czf "$WORK/out/cr_detector.tgz" -C "$RUN" weights results.csv $(cd "$RUN" && ls *.png 2>/dev/null) || \
  tar czf "$WORK/out/cr_detector.tgz" -C "$RUN" weights
ls -la "$WORK/out"

echo "== $(date -u +%FT%TZ) serving $WORK/out on port 8000 for $SERVE_MINUTES minutes"
echo "== download: <service URL>/cr_detector.tgz  (then stop the job to save credits)"
cd "$WORK/out"
timeout "$((SERVE_MINUTES * 60))" python -m http.server 8000 || true
echo "== done"
