"""Command line: ``python -m crbot <command>``.

    live           read the game state from a feed, show / record / log it
    image          read the state of one screenshot
    calibrate      draw the screen layout on a screenshot to check alignment
    record         save frames from a feed, e.g. to label for training
    cards-collect  collect hand-card templates from your own game
    fetch-models   download ClashRoyaleBuildABot's pretrained ONNX models
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path


def _deck(value):
    return [c.strip() for c in value.split(",") if c.strip()] if value else None


def cmd_live(args) -> int:
    import cv2

    from .capture import LatestFrame, open_source
    from .geometry import Layout
    from .state.builder import GameStateBuilder
    from .viz import draw_state

    layout = Layout.load(args.layout) if args.layout else None
    source = open_source(args.source, realtime=args.realtime, step=args.step)
    if args.threaded:
        source = LatestFrame(source)

    first = source.read()
    if first is None:
        print("source produced no frames", file=sys.stderr)
        return 1
    builder = GameStateBuilder(deck=_deck(args.deck), layout=layout, elapsed_offset=args.elapsed_offset)
    lay, grid = builder.geometry(*first.size)
    _setup_models(builder, lay, args)

    log = open(args.log, "a", encoding="utf-8") if args.log else None
    writer = None
    frame = first
    count, started = 0, time.monotonic()
    try:
        while frame is not None:
            state = builder.process(frame)
            count += 1
            if log:
                log.write(state.to_json() + "\n")
            if args.show or args.record:
                overlay = draw_state(frame.image, state, grid, builder.last_detections, lay)
                if args.record:
                    if writer is None:
                        h, w = overlay.shape[:2]
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.record_fps, (w, h))
                    writer.write(overlay)
                if args.show:
                    cv2.imshow("crbot", overlay)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            if args.print_every and count % args.print_every == 0:
                rate = count / max(1e-6, time.monotonic() - started)
                hand = ",".join(c or "?" for c in state.hand)
                print(f"[{count} frames, {rate:.1f} fps] battle={state.in_battle} elixir={state.elixir} "
                      f"hand={hand} units={len(state.units)} enemy_elixir={state.enemy_elixir}")
            if args.max_frames and count >= args.max_frames:
                break
            frame = source.read()
    finally:
        source.close()
        if log:
            log.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


def _setup_models(builder, layout, args) -> None:
    from .vision.detector import load_detector
    from .vision.side import OnnxCropClassifier

    builder.detector = load_detector(args.detector, arena=layout.arena,
                                     min_conf=args.min_conf, device=args.device)
    side_model = args.side_model
    if side_model is None and args.detector == "buildabot" and Path("models/buildabot/side.onnx").exists():
        side_model = "models/buildabot/side.onnx"
    if side_model:
        builder.sides.classifier = OnnxCropClassifier(Path(side_model))


def cmd_image(args) -> int:
    """One-shot perception on a single screenshot: no tracking over time."""
    import cv2

    from .capture import Frame
    from .geometry import Layout
    from .state.builder import GameStateBuilder
    from .viz import draw_state

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 1
    layout = Layout.load(args.layout) if args.layout else None
    builder = GameStateBuilder(deck=_deck(args.deck), layout=layout)
    builder.tracker.confirm_hits = 1          # a single frame has to be enough
    h, w = image.shape[:2]
    lay, grid = builder.geometry(w, h)
    _setup_models(builder, lay, args)
    state = builder.process(Frame(image, 0.0, 0))

    print(f"{w}x{h}, layout '{lay.name}', screen={state.screen}, in battle={state.in_battle}")
    if state.in_battle:
        cards = ", ".join("unrecognised" if c is None else f"{c}{'' if p else ' (greyed)'}"
                          for c, p in zip(state.hand, state.playable))
        print(f"hand: {cards}   next: {state.next_card or '?'}")
        print(f"elixir: {state.elixir}")
        print("towers: " + ", ".join(f"{k}={v.hp:.2f}" for k, v in state.towers.items()))
        print(f"units ({len(builder.last_detections)}):")
        for d in sorted(builder.last_detections, key=lambda d: d.tile[1]):
            print(f"  {d.name:16s} {'ours  ' if d.side == 0 else 'theirs'} tile=({d.tile[0]:4.1f},{d.tile[1]:4.1f}) "
                  f"conf={d.conf:.2f} side by {d.side_method} ({d.side_conf:.2f})")
    if args.out:
        cv2.imwrite(args.out, draw_state(image, state, grid, builder.last_detections, lay))
        print(f"overlay -> {args.out}")
    if args.json:
        Path(args.json).write_text(state.to_json() + "\n", encoding="utf-8")
    return 0


def cmd_calibrate(args) -> int:
    import cv2

    from .capture import open_source
    from .geometry import ArenaGrid, Layout, layout_for
    from .viz import draw_layout

    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            print(f"could not read {args.image}", file=sys.stderr)
            return 1
    else:
        source = open_source(args.source)
        frame = source.read()
        source.close()
        if frame is None:
            print("source produced no frames", file=sys.stderr)
            return 1
        image = frame.image
    h, w = image.shape[:2]
    layout = Layout.load(args.layout) if args.layout else layout_for(w, h)
    grid = ArenaGrid.from_layout(layout, w, h)
    cv2.imwrite(args.out, draw_layout(image, layout, grid))
    print(f"{w}x{h} frame, layout '{layout.name}' (verified={layout.verified}) -> {args.out}")
    if args.save_layout:
        Path(args.save_layout).write_text(layout.to_json() + "\n", encoding="utf-8")
        print(f"layout written to {args.save_layout}; edit it and pass --layout to use it")
    return 0


def cmd_record(args) -> int:
    import cv2

    from .capture import open_source

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = open_source(args.source)
    saved, last = 0, None
    try:
        while args.max_frames == 0 or saved < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            if last is not None and frame.t - last < args.every:
                continue
            last = frame.t
            cv2.imwrite(str(out / f"{int(time.time() * 1000)}_{frame.index:06d}.png"), frame.image)
            saved += 1
    finally:
        source.close()
    print(f"saved {saved} frames to {out}")
    return 0


def cmd_cards_collect(args) -> int:
    from .capture import frames, open_source
    from .geometry import Layout
    from .vision.collect import collect

    layout = Layout.load(args.layout) if args.layout else None
    groups = collect(frames(open_source(args.source, step=args.step)), Path(args.out), layout=layout,
                     every=args.every, threshold=args.threshold)
    for g in sorted(groups, key=lambda g: -g.count):
        print(f"{g.count:6d} crops  {g.folder}")
    print(f"\n{len(groups)} groups in {args.out}. Rename each folder to its card name "
          "(merge greyed groups into the same folder) and move them to models/cards/.")
    return 0


# Pinned to a commit so the files cannot change underneath us.
BUILDABOT_COMMIT = "f89ecbf309e698cb9fccdfb64a6ff2a930e6541e"
BUILDABOT_MODELS = {
    "units_M_480x352.onnx": "4cf422370650b1d18647cffc05e02bf75fdc2fbdb1f8ef983f4b8d6f8ea908b3",
    "side.onnx": "686bc75eae2ddc872b556534aeac8b4f5c6e891df76eef12e9fc91d406ab2583",
}


def cmd_fetch_models(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = (f"https://raw.githubusercontent.com/Pbatch/ClashRoyaleBuildABot/{BUILDABOT_COMMIT}"
            "/clashroyalebuildabot/models/")
    for name, digest in BUILDABOT_MODELS.items():
        target = out / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f"{target} already present")
            continue
        print(f"downloading {name} ...")
        data = urllib.request.urlopen(base + name, timeout=120).read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            print(f"checksum mismatch for {name}: {actual}", file=sys.stderr)
            return 1
        target.write_bytes(data)
        print(f"  -> {target} ({len(data) / 1e6:.1f} MB)")
    print("ClashRoyaleBuildABot models are MIT licensed; see THIRD_PARTY_NOTICES.md")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="crbot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="read game state from a feed")
    live.add_argument("--source", required=True,
                      help="adb, adb:SERIAL, device:N, a video file, a stream URL or an image folder")
    live.add_argument("--detector", default="none",
                      help="'buildabot', a .onnx/.pt model path, or 'none' (default)")
    live.add_argument("--side-model", help="ONNX ally/enemy crop classifier")
    live.add_argument("--deck", help="your 8 cards, comma separated (e.g. hog_rider,musketeer,...)")
    live.add_argument("--layout", help="layout JSON from `calibrate --save-layout`")
    live.add_argument("--min-conf", type=float, default=0.35)
    live.add_argument("--device", help="inference device for ultralytics models (e.g. 0, cpu)")
    live.add_argument("--elapsed-offset", type=float, default=0.0,
                      help="match seconds already played when the feed starts")
    live.add_argument("--show", action="store_true", help="show an overlay window")
    live.add_argument("--record", help="write the overlay to this .mp4")
    live.add_argument("--record-fps", type=float, default=10.0)
    live.add_argument("--log", help="append one JSON game state per line to this file")
    live.add_argument("--realtime", action="store_true", help="replay video files at wall-clock speed")
    live.add_argument("--step", type=int, default=1, help="use every n-th frame of a video file")
    live.add_argument("--threaded", action="store_true", help="always process the newest frame (live feeds)")
    live.add_argument("--max-frames", type=int, default=0)
    live.add_argument("--print-every", type=int, default=10)
    live.set_defaults(func=cmd_live)

    img = sub.add_parser("image", help="read the state of a single screenshot")
    img.add_argument("image")
    img.add_argument("--detector", default="buildabot")
    img.add_argument("--side-model")
    img.add_argument("--deck")
    img.add_argument("--layout")
    img.add_argument("--min-conf", type=float, default=0.35)
    img.add_argument("--device")
    img.add_argument("--out", help="write the overlay image here")
    img.add_argument("--json", help="write the GameState JSON here")
    img.set_defaults(func=cmd_image)

    cal = sub.add_parser("calibrate", help="draw the layout on a frame to check alignment")
    cal.add_argument("--image", help="a screenshot")
    cal.add_argument("--source", default="adb", help="or grab one frame from a feed")
    cal.add_argument("--layout", help="layout JSON to check instead of the built-in one")
    cal.add_argument("--out", default="calibration.png")
    cal.add_argument("--save-layout", help="also write the layout as editable JSON")
    cal.set_defaults(func=cmd_calibrate)

    rec = sub.add_parser("record", help="save frames from a feed")
    rec.add_argument("--source", default="adb")
    rec.add_argument("--out", default="recordings/frames")
    rec.add_argument("--every", type=float, default=0.5, help="seconds between saved frames")
    rec.add_argument("--max-frames", type=int, default=0)
    rec.set_defaults(func=cmd_record)

    cards = sub.add_parser("cards-collect", help="collect hand-card templates from your own game")
    cards.add_argument("--source", required=True, help="a recording or live feed (see `live --source`)")
    cards.add_argument("--out", default="models/cards_unlabeled")
    cards.add_argument("--layout", help="layout JSON")
    cards.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    cards.add_argument("--step", type=int, default=1, help="read every n-th frame of a video")
    cards.add_argument("--threshold", type=float, default=0.3, help="hash distance that starts a new group")
    cards.set_defaults(func=cmd_cards_collect)

    fetch = sub.add_parser("fetch-models", help="download BuildABot's pretrained ONNX models")
    fetch.add_argument("--out", default="models/buildabot")
    fetch.set_defaults(func=cmd_fetch_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
