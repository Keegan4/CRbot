# CRbot

Reads the game state of a Clash Royale match from a live video feed, as the
first stage of a deep-learning bot. Every frame becomes a structured
`GameState`: units with positions, sides and velocities, your hand and
elixir, tower HP, the plays both players made, and an estimate of the
opponent's elixir and deck.

> **Terms of Service.** Supercell's terms forbid bots and automation on live
> ladder play. Accounts that use them get banned. Use this on recordings,
> in friendly battles against your own second account, or in Training Camp.

This stage only *reads* the game. It never taps the screen.

## Quick start

```bash
pip install -e ".[onnx,dev]"
python -m crbot fetch-models              # BuildABot's pretrained detector (MIT, ~31 MB)

# A recorded match (9:16, e.g. 720x1280 or 1080x1920), every 6th frame:
python -m crbot live --source match.mp4 --step 6 --detector buildabot \
    --deck hog_rider,musketeer,cannon,fireball,the_log,ice_golem,ice_spirit,skeletons \
    --record overlay.mp4 --log states.jsonl

# One screenshot (prints the state, writes an overlay):
python -m crbot image screenshot.png --out overlay.png

# An emulator over ADB (set it to 720x1280 or 1080x1920), with a live window:
python -m crbot live --source adb --threaded --show --detector buildabot --deck ...
```

`--source` takes `adb`, `adb:SERIAL`, `device:N` (e.g. scrcpy's
`--v4l2-sink` or an OBS virtual camera), a stream URL, a video file or a
folder of images. Always pass `--deck` (card names as in
`crbot/assets/cards.json`). It makes card reading and ally/enemy decisions
much more reliable.

Each line of `states.jsonl` is one `GameState` (see
`crbot/state/game_state.py`):

```json
{"t": 12.0, "screen": "in_game", "in_battle": true, "elapsed": 12.0, "elixir": 2.09,
 "hand": ["fireball", "musketeer", "ice_golem", "cannon"], "playable": [false, false, true, false],
 "next_card": "hog_rider",
 "towers": {"ally_left": {"hp": 0.89, "destroyed": false}, "...": "..."},
 "units": [{"id": 9, "name": "ice_golem", "side": 0, "x": 14.8, "y": 13.1, "vx": 0.0, "vy": -1.1, "...": "..."}],
 "plays": [{"t": 9.0, "side": 0, "card": "ice_spirit", "cost": 1, "x": 1.2, "y": 26.6, "...": "..."}],
 "enemy_elixir": 4.2, "opponent_deck": ["giant", "..."], "opponent_unavailable": ["..."]}
```

Coordinates are tiles on the 18 x 32 arena: `x` from 0 (left) to 18, and `y`
from 0 (top, the opponent's back row) to 32. Our half is `y >= 16`. `side` is
0 for us and 1 for the opponent.

## How it works

```
frame ─┬─ screen type (in game / lobby / end)
       ├─ hand cards + next card ── vote over frames ──┐
       ├─ elixir bar                                   ├─ our plays (card left hand + elixir drop)
       ├─ princess tower HP bars ── median, occlusion ─┤
       └─ unit detector ── ally/enemy ── tiles ── tracker ── enemy plays ── opponent elixir + cycle
                                                                                     │
                                                                                 GameState
```

| Part | Method | File |
|---|---|---|
| Screen layout | Regions stored as fractions of the frame, so one profile covers 720x1280 and 1080x1920. Measurements from ClashRoyaleBuildABot and checked on live 9:16 frames | `crbot/geometry.py` |
| Hand | Brightness-normalised 8x8 greyscale hash against card templates. Slots are assigned jointly (a hand never holds a card twice). Empty and greyed-out cards are detected separately | `crbot/vision/hud.py` |
| Elixir | How far the magenta fill reaches along the bar (red channel) | `crbot/vision/hud.py` |
| Tower HP | Bar colours. Picks the row with the most bar-coloured pixels, rejects partly covered bars, takes a rolling median | `crbot/vision/hud.py`, `crbot/state/towers.py` |
| Units | ONNX detector (BuildABot's, or your own) or an Ultralytics YOLO model. Ground point is the bottom-centre of the box | `crbot/vision/detector.py` |
| Ally / enemy | Strongest evidence first: detector output → "not in our deck" → "we just played that card" → crop classifier → which half | `crbot/vision/side.py`, `crbot/state/builder.py` |
| Tracking | Hungarian matching on predicted position, per class. Confirmation after 2 hits, re-identification of units lost for up to 3 s | `crbot/state/tracker.py` |
| Plays | Ours: a card leaving the hand with a matching elixir drop. Theirs: new units, with spawns, illegal positions and multi-unit cards handled | `crbot/state/builder.py`, `crbot/state/plays.py` |
| Opponent | Elixir from the regeneration schedule minus observed plays (an upper bound). Deck and "can't be in hand" from the 8-card cycle | `crbot/state/plays.py` |

## Measured so far

Checked on frames from a real match, cropped from Hasty-CR's showcase
recording, which has another bot's overlay drawn on top:

* **Hand cards:** 44/44 slots correct over 11 frames with `--deck`, including
  greyed-out cards and empty slots. Without a deck, Musketeer is misread,
  because the bundled art is older than the current card art (see "Card
  templates" below).
* **Next card, playable/greyed flags, in-game screen:** correct on every frame
  where they were visible.
* **Elixir:** fractional readings that agree with which cards are greyed out.
* **Our tower HP:** 0.89 against a true 3304/3668 = 0.90.
* **Our plays:** read correctly from hand changes.
* **Speed:** about 10 frames/s end to end on a 4-core CPU with the BuildABot
  detector (~65 ms of the ~70 ms per frame). The fixed-region readers take
  about 4 ms.

Three of those frames are regression tests (`tests/test_hud.py`). Run
everything with `python -m pytest`.

**Not yet good enough:**

* **The BuildABot detector** has no tower classes. It fires on tower archers
  and seasonal tower skins (ground units inside a standing tower are
  filtered out, flying ones are not). Its side classifier flips on some
  units. Enemy play inference and the opponent elixir estimate are only as
  good as the detector. **Train your own detector** (below) for real use.
* **Enemy tower HP bars** are implemented but not validated: the overlay in
  the test footage covers them. Check them with `calibrate` on clean
  frames.
* **No match timer reading.** Match time counts from the first battle frame
  seen. Pass `--elapsed-offset` if the feed starts mid-match, or the
  double-elixir switch will be off.
* **No unit HP.** Units are reported without hit points.
* **King tower HP** is not read.

## Train your own detector

The KataCR dataset (MIT) has ~6,900 hand-labelled arena frames with 201
classes, including towers and spell effects. Spell effects let the opponent
model see enemy spells as plays.

```bash
git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset datasets/katacr
python training/prepare_katacr.py --src datasets/katacr/images/part2 --out datasets/cr_yolo
pip install -e ".[train]"
python training/train_detector.py --data datasets/cr_yolo/cr.yaml --model yolov8s.pt --export-onnx
python training/train_side_classifier.py --data datasets/cr_yolo --out models/side.onnx
python -m crbot live --source adb --detector runs/detect/cr/weights/best.pt --side-model models/side.onnx --deck ...
```

`prepare_katacr.py --side fold` trains the detector to predict ally/enemy
itself (402 classes) instead. Expect several hours on one consumer GPU. Hasty-CR
reported 0.959 mAP50 with YOLOv8s on this data.

## Card templates from your own game

Card art changes between game versions. Collect templates from a recording
of your own matches:

```bash
python -m crbot cards-collect --source my_match.mp4 --step 6
# rename each models/cards_unlabeled/group_NN__<guess> folder to the card name
# (put greyed-out groups in the same folder), then:
mv models/cards_unlabeled models/cards
```

`CardReader` uses `models/cards/<card>/` before the bundled art.

## Other aspect ratios

Layouts are per aspect ratio. Clash Royale fits the arena to the screen width
on tall phones and to the height on 9:16 screens. There are two profiles:
9:16 (emulators; used by two other bots for live play) and 2.17:1 (phones such
as 942x2048 or 1080x2340, measured on one screenshot, where all four tower HP
bars read within 0.02 of the true values). To check or adjust one:

```bash
python -m crbot calibrate --image screenshot.png --out check.png --save-layout my_layout.json
# edit my_layout.json until check.png lines up, then pass --layout my_layout.json
```

## Layout

```
crbot/
  capture.py        frame sources (adb, video, device, folder) + latest-frame thread
  geometry.py       screen layouts, tile <-> pixel mapping
  catalog.py        card/unit metadata and detector name matching
  vision/           per-frame readers: detector, side, hud (cards/elixir/towers/screen), collect
  state/            tracker, hand voting, tower smoothing, play inference, GameState builder
  viz.py            overlay drawing
  __main__.py       CLI: live, calibrate, record, cards-collect, fetch-models
training/           dataset conversion and model training
tests/              unit tests and real-frame regression tests
```

Third-party material and licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
