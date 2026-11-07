"""Frame sources: an emulator over ADB, a video file or stream, a camera
device (e.g. scrcpy's v4l2 sink or an OBS virtual camera), or a folder of
images.

Every source yields `Frame` objects holding a BGR ``uint8`` image and a
timestamp in seconds. Live sources use a monotonic clock. Video files use the
position in the file, so a recorded match replays with its real timing.

For live play, wrap a source in `LatestFrame`. It reads on a background thread
and always hands back the newest frame. If inference is slower than capture,
stale frames are dropped instead of building up a growing delay.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Union

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Frame:
    image: np.ndarray          # H x W x 3, BGR, uint8
    t: float                   # seconds
    index: int

    @property
    def size(self):
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(Protocol):
    def read(self) -> Optional[Frame]: ...
    def close(self) -> None: ...


def frames(source: FrameSource) -> Iterator[Frame]:
    """Iterate a source until it runs out, closing it afterwards."""
    try:
        while True:
            frame = source.read()
            if frame is None:
                return
            yield frame
    finally:
        source.close()


# --------------------------------------------------------------------- adb


def parse_raw_screencap(raw: bytes) -> np.ndarray:
    """Decode ``adb exec-out screencap`` output (no ``-p``) into BGR.

    The raw format is a little-endian header (width, height, pixel format and,
    on Android 10+, a colour space) followed by RGBA pixels. Skipping PNG
    encoding on the device roughly halves capture time.
    """
    if len(raw) < 12:
        raise ValueError("screencap output too short")
    width = int.from_bytes(raw[0:4], "little")
    height = int.from_bytes(raw[4:8], "little")
    size = width * height * 4
    header = len(raw) - size
    if header not in (12, 16) or width <= 0 or height <= 0:
        raise ValueError(f"unexpected screencap layout ({len(raw)} bytes for {width}x{height})")
    rgba = np.frombuffer(raw, dtype=np.uint8, count=size, offset=header)
    rgba = rgba.reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, 2::-1])     # RGBA -> BGR


class AdbSource:
    """Screenshots from an Android device or emulator over ADB.

    Typically 15-40 ms per frame on a local emulator, which is plenty for
    decisions every 0.5 s. For higher rates, stream with scrcpy into a v4l2
    or virtual camera device and use `VideoSource` instead.
    """

    def __init__(self, serial: Optional[str] = None, adb: str = "adb", timeout: float = 3.0):
        self.adb = shutil.which(adb) or adb
        self.serial = serial
        self.timeout = timeout
        self.index = 0

    def _command(self) -> List[str]:
        cmd = [self.adb]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + ["exec-out", "screencap"]

    def read(self) -> Optional[Frame]:
        result = subprocess.run(self._command(), capture_output=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"adb screencap failed: {result.stderr.decode(errors='replace').strip()}")
        image = parse_raw_screencap(result.stdout)
        frame = Frame(image=image, t=time.monotonic(), index=self.index)
        self.index += 1
        return frame

    def close(self) -> None:
        pass


# ------------------------------------------------------------------- video


class VideoSource:
    """A video file, a stream URL, or a capture device index (via OpenCV).

    ``realtime=True`` on a file drops frames to keep pace with the wall clock,
    which reproduces what a live feed would deliver to a slow pipeline.
    ``step`` reads every n-th frame of a file (e.g. 6 for 10 fps from 60 fps).
    """

    def __init__(self, target: Union[str, int], realtime: bool = False, step: int = 1):
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(target)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open video source {target!r}")
        self.is_file = isinstance(target, str) and Path(target).exists()
        self.realtime = realtime and self.is_file
        self.step = max(1, int(step))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.index = 0
        self._started: Optional[float] = None

    def read(self) -> Optional[Frame]:
        cv2 = self.cv2
        if self.realtime:
            if self._started is None:
                self._started = time.monotonic()
            target = int((time.monotonic() - self._started) * self.fps)
            current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            if target > current + 1:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        elif self.step > 1 and self.index > 0:
            for _ in range(self.step - 1):
                if not self.capture.grab():
                    return None
        ok, image = self.capture.read()
        if not ok or image is None:
            return None
        if self.is_file:
            t = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            t = time.monotonic()
        frame = Frame(image=image, t=t, index=self.index)
        self.index += 1
        return frame

    def close(self) -> None:
        self.capture.release()


class ImageFolderSource:
    """Images in name order, timestamped at a nominal frame rate."""

    def __init__(self, folder: Union[str, Path], fps: float = 2.0):
        import cv2

        self.cv2 = cv2
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise RuntimeError(f"no images in {folder}")
        self.fps = fps
        self.index = 0

    def read(self) -> Optional[Frame]:
        if self.index >= len(self.paths):
            return None
        image = self.cv2.imread(str(self.paths[self.index]), self.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {self.paths[self.index]}")
        frame = Frame(image=image, t=self.index / self.fps, index=self.index)
        self.index += 1
        return frame

    def close(self) -> None:
        pass


class LatestFrame:
    """Read a live source on a thread; `read` returns the newest frame only."""

    def __init__(self, source: FrameSource):
        self.source = source
        self._frame: Optional[Frame] = None
        self._error: Optional[BaseException] = None
        self._done = False
        self._seen = -1
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._done:
                frame = self.source.read()
                with self._cond:
                    if frame is None:
                        self._done = True
                    else:
                        self._frame = frame
                    self._cond.notify_all()
        except BaseException as exc:          # surfaced on the next read
            with self._cond:
                self._error = exc
                self._done = True
                self._cond.notify_all()

    def read(self) -> Optional[Frame]:
        with self._cond:
            while (self._frame is None or self._frame.index == self._seen) and not self._done:
                self._cond.wait(timeout=5.0)
            if self._error is not None:
                raise RuntimeError("capture failed") from self._error
            if self._frame is None or self._frame.index == self._seen:
                return None
            self._seen = self._frame.index
            return self._frame

    def close(self) -> None:
        self._done = True
        self._thread.join(timeout=5.0)
        self.source.close()


def open_source(spec: str, realtime: bool = False, step: int = 1) -> FrameSource:
    """Build a source from a short string.

    ``adb`` or ``adb:SERIAL``      emulator/device screenshots
    ``device:N``                   capture device N (scrcpy v4l2 sink, OBS camera)
    ``rtsp://...`` / ``http://...`` network stream
    a folder                       images in name order
    anything else                  a video file
    """
    if spec == "adb" or spec.startswith("adb:"):
        serial = spec.split(":", 1)[1] if ":" in spec else None
        return AdbSource(serial=serial or None)
    if spec.startswith("device:"):
        return VideoSource(int(spec.split(":", 1)[1]))
    path = Path(spec)
    if path.is_dir():
        return ImageFolderSource(path)
    return VideoSource(spec, realtime=realtime, step=step)
