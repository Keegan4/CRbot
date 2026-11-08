"""Write training/nosana/job.json from train_katacr.sh.

    python training/nosana/make_job.py                       # full run -> job.json
    python training/nosana/make_job.py --epochs 1 --serve-minutes 20 --out job_test.json

The job runs the script with bash inside the Ultralytics image on one GPU and
exposes port 8000, where the script serves the trained model when it is done.
Field names follow Nosana's job definition schema (version "0.1").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--serve-minutes", type=int, default=60)
    parser.add_argument("--image", default="docker.io/ultralytics/ultralytics:latest")
    parser.add_argument("--vram", type=int, default=16, help="minimum GPU memory in GB")
    parser.add_argument("--out", default="job.json")
    args = parser.parse_args()

    script = (HERE / "train_katacr.sh").read_text(encoding="utf-8")
    settings = (f"export EPOCHS={args.epochs} MODEL={args.model} BATCH={args.batch} "
                f"SERVE_MINUTES={args.serve_minutes}\n")
    job = {
        "version": "0.1",
        "type": "container",
        "meta": {"trigger": "dashboard", "system_requirements": {"required_vram": args.vram}},
        "ops": [{
            "id": "train-cr-detector",
            "type": "container/run",
            "args": {
                "image": args.image,
                "gpu": True,
                "shm_size": "8g",
                "expose": 8000,
                "cmd": ["/bin/bash", "-c", settings + script],
            },
        }],
    }
    out = HERE / args.out
    out.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
