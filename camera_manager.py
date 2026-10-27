"""
camera_manager.py - threaded camera acquisition with measured FPS.

Design rules (from the spec):
  * The selected index/backend is used exactly. If it fails, raise CameraError.
    There is NO fallback to another camera.
  * FPS is measured from real frame timestamps, never assumed.
  * A frame-size mismatch is an error, because the firmware's centre/tolerance
    checks assume 640x480.

Run:   not run directly; used by main.py and the diagnostics.
       Use diagnostics/camera_test.py to probe cameras.
Deps:  opencv-python, numpy, config.py
Hardware-specific: CameraConfig.index / backend / fourcc.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import CameraConfig  # keep before cv2 import (sets MSMF env var)
import cv2
import numpy as np

BACKENDS = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW, "DEFAULT": cv2.CAP_ANY}
FOURCCS = {"MJPG": "MJPG", "YUY2": "YUY2", "DEFAULT": None}


class CameraError(RuntimeError):
    """Raised when the selected camera cannot be opened or stops working."""


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray       # private copy, safe to draw on
    frame_id: int
    timestamp: float        # time.monotonic()


@dataclass(frozen=True)
class CameraStatus:
    connected: bool
    error: str
    fps: float
    frame_count: int
    width: int
    height: int
    backend: str
    index: int


def _apply_settings(cap, fourcc_name, width, height, fps):
    code = FOURCCS.get(fourcc_name)
    if code:  # FOURCC first, then size, then rate
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends


class CameraManager:
    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._running = False
        self._error = ""
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._stamps = deque(maxlen=max(2, cfg.fps_window))
        self._size = (0, 0)

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        """Open the configured camera. Raises CameraError with a clear message."""
        self.close()
        cfg = self._cfg
        if cfg.backend not in BACKENDS:
            raise CameraError(f"Unknown backend '{cfg.backend}'. Choose from {list(BACKENDS)}.")

        cap = cv2.VideoCapture(cfg.index, BACKENDS[cfg.backend])
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Camera index {cfg.index} could not be opened with backend {cfg.backend}. "
                "Check the USB cable, close other apps using the camera, and run "
                "diagnostics/camera_test.py to find the right index. "
                "No other camera was tried.")

        _apply_settings(cap, cfg.fourcc, cfg.width, cfg.height, cfg.target_fps)

        deadline = time.monotonic() + cfg.open_timeout_s
        first = None
        while time.monotonic() < deadline:
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                first = f
                break
            time.sleep(0.05)
        if first is None:
            cap.release()
            raise CameraError(
                f"Camera {cfg.index} ({cfg.backend}) opened but delivered no frames within "
                f"{cfg.open_timeout_s:.1f} s. Try another backend or FOURCC.")

        h, w = first.shape[:2]
        if (w, h) != (cfg.width, cfg.height):
            cap.release()
            raise CameraError(
                f"Camera {cfg.index} returned {w}x{h} but {cfg.width}x{cfg.height} is required "
                "(the firmware assumes this frame size). Try another FOURCC/backend.")

        with self._cond:
            self._cap = cap
            self._error = ""
            self._frame = first
            self._frame_id = 1
            self._frame_ts = time.monotonic()
            self._stamps.clear()
            self._stamps.append(self._frame_ts)
            self._size = (w, h)
            self._running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- thread
    def _run(self) -> None:
        cap = self._cap
        failures = 0
        while not self._stop.is_set():
            try:
                ok, frame = cap.read()
            except cv2.error:
                ok, frame = False, None
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= self._cfg.max_consecutive_failures:
                    with self._cond:
                        self._error = "Camera stopped delivering frames (unplugged or in use elsewhere?)"
                    break
                time.sleep(0.01)
                continue
            failures = 0
            now = time.monotonic()
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._frame_ts = now
                self._stamps.append(now)
                self._cond.notify_all()
        with self._cond:
            self._running = False
            self._cond.notify_all()

    # ------------------------------------------------------------- consumers
    def wait_for_frame(self, last_id: int, timeout: float = 0.5) -> Optional[FramePacket]:
        """Block until a frame newer than last_id arrives. Returns None on timeout/stop."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._frame_id <= last_id and self._running:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._frame is None or self._frame_id <= last_id:
                return None
            return FramePacket(self._frame.copy(), self._frame_id, self._frame_ts)

    def get_latest(self) -> Optional[FramePacket]:
        with self._cond:
            if self._frame is None:
                return None
            return FramePacket(self._frame.copy(), self._frame_id, self._frame_ts)

    def fps(self) -> float:
        """Measured capture FPS over the last fps_window frames (0.0 if stalled)."""
        with self._cond:
            if len(self._stamps) < 2 or time.monotonic() - self._stamps[-1] > 1.0:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def status(self) -> CameraStatus:
        fps = self.fps()
        with self._cond:
            return CameraStatus(
                connected=self._running and not self._error,
                error=self._error,
                fps=fps,
                frame_count=self._frame_id,
                width=self._size[0],
                height=self._size[1],
                backend=self._cfg.backend,
                index=self._cfg.index,
            )

    @property
    def is_running(self) -> bool:
        with self._cond:
            return self._running

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        with self._cond:
            self._running = False
            self._frame = None
            self._cond.notify_all()
        self._thread = None


def probe_camera(index, backend="MSMF", fourcc="MJPG", width=640, height=480, duration_s=3.0):
    """
    One-shot diagnostic used by camera_test.py. Never raises.
    Returns a dict: opened, frames, fps, width, height, error.
    """
    result = {"index": index, "backend": backend, "fourcc": fourcc, "opened": False,
              "frames": 0, "fps": 0.0, "width": 0, "height": 0, "error": ""}
    cap = None
    try:
        cap = cv2.VideoCapture(index, BACKENDS[backend])
        if not cap.isOpened():
            result["error"] = "could not open"
            return result
        _apply_settings(cap, fourcc, width, height, 30.0)
        # discard warm-up frames
        for _ in range(5):
            cap.read()
        stamps = []
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                stamps.append(time.monotonic())
                result["height"], result["width"] = f.shape[:2]
        result["opened"] = True
        result["frames"] = len(stamps)
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            result["fps"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        elif not stamps:
            result["error"] = "opened but no frames"
    except Exception as exc:  # diagnostic must not crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
    return result
