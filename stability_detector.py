"""
stability_detector.py - host-side "same place for N seconds" detector.

Three DIFFERENT conditions are tracked (they are not interchangeable):
  1. stable in image : the balloon centre stays within position_tolerance_px of an anchor
                       (image_stable_s). A balloon sitting still at the frame edge is stable
                       but is NOT centred.
  2. centred         : the balloon centre is within center_tolerance_px of the frame centre,
                       i.e. the pan/tilt has actually put it where the mechanism expects it.
  3. ready           : stable AND centred continuously for required_s. Only this is reported
                       as ready. The ESP32 repeats the same test itself and has the last word.

Rules:
  * Timing uses time.monotonic() (or an injected `now`).
  * No timer starts until min_consecutive_detections reliable detections are seen.
  * A missing detection NEVER counts as stability. A gap up to dropout_grace_s is tolerated
    (timers keep running and the gap counts as elapsed time, exactly like the firmware);
    a longer gap resets everything.
  * Moving beyond position_tolerance_px restarts the timer at the new position.

Run:   python stability_detector.py      (self-test with a simulated clock; Phase 7)
Deps:  config.py only.
"""
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from config import StabilityConfig


@dataclass(frozen=True)
class StabilityInfo:
    status: str          # NO_TARGET / ACQUIRING / NOT_CENTERED / STABILIZING / READY
    consecutive: int
    image_stable_s: float
    centered: bool
    locked_s: float      # continuous stable AND centred time (drives activation)
    required_s: float
    ready: bool
    progress: float      # 0..1


class StabilityDetector:
    def __init__(self, cfg: StabilityConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._consecutive = 0
        self._last_seen: Optional[float] = None
        self._anchor: Optional[Tuple[float, float]] = None
        self._anchor_start: Optional[float] = None
        self._lock_start: Optional[float] = None
        self._centered = False

    def update(self, found: bool, cx, cy, frame_w: int, frame_h: int, now: Optional[float] = None) -> StabilityInfo:
        now = time.monotonic() if now is None else now
        c = self.cfg
        if found and cx is not None and cy is not None:
            self._consecutive += 1
            self._last_seen = now
            reliable = self._consecutive >= c.min_consecutive_detections or self._anchor is not None
            if reliable:
                self._update_position(cx, cy, frame_w, frame_h, now)
        else:
            self._consecutive = 0
            if self._anchor is not None and (self._last_seen is None or now - self._last_seen > c.dropout_grace_s):
                self.reset()
        return self.info(now)

    def _update_position(self, cx, cy, fw, fh, now) -> None:
        c = self.cfg
        if self._anchor is None or math.hypot(cx - self._anchor[0], cy - self._anchor[1]) > c.position_tolerance_px:
            self._anchor = (cx, cy)
            self._anchor_start = now
            self._lock_start = None          # moved: restart timing at the new position
        self._centered = (abs(cx - fw / 2.0) <= c.center_tolerance_px and
                          abs(cy - fh / 2.0) <= c.center_tolerance_px)
        if self._centered or not c.require_centered:
            if self._lock_start is None:
                self._lock_start = now
        else:
            self._lock_start = None

    def info(self, now: Optional[float] = None) -> StabilityInfo:
        now = time.monotonic() if now is None else now
        req = self.cfg.required_s
        image_s = (now - self._anchor_start) if self._anchor_start is not None else 0.0
        locked = (now - self._lock_start) if self._lock_start is not None else 0.0
        ready = self._lock_start is not None and locked >= req
        if self._anchor is None:
            status = "ACQUIRING" if self._consecutive > 0 else "NO_TARGET"
        elif self._lock_start is None:
            status = "NOT_CENTERED"
        elif ready:
            status = "READY"
        else:
            status = "STABILIZING"
        return StabilityInfo(status, self._consecutive, image_s, self._centered, locked, req, ready,
                             min(1.0, locked / req) if req > 0 else 0.0)


def _self_test() -> int:
    cfg = StabilityConfig(required_s=5.0, dropout_grace_s=0.3, min_consecutive_detections=5)
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        print(("PASS  " if cond else "FAIL  ") + name)
        ok_all = ok_all and cond

    dt = 1 / 30.0
    d = StabilityDetector(cfg)
    t = 0.0
    for _ in range(int(4.8 / dt)):
        i = d.update(True, 322, 241, 640, 480, t); t += dt
    check("not ready at ~4.8 s", not i.ready)
    for _ in range(int(0.6 / dt)):
        i = d.update(True, 322, 241, 640, 480, t); t += dt
    check("ready after >5 s centred and still", i.ready)

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(int(3 / dt)):
        i = d.update(True, 320, 240, 640, 480, t); t += dt
    i = d.update(True, 380, 240, 640, 480, t); t += dt
    check("timer restarts when balloon moves beyond tolerance", i.locked_s < 0.1 and not i.ready)

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(int(3 / dt)):
        i = d.update(True, 320, 240, 640, 480, t); t += dt
    for _ in range(int(0.2 / dt)):
        i = d.update(False, None, None, 640, 480, t); t += dt
    i = d.update(True, 320, 240, 640, 480, t); t += dt
    check("0.2 s dropout is tolerated", i.locked_s > 2.9)

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(int(3 / dt)):
        i = d.update(True, 320, 240, 640, 480, t); t += dt
    for _ in range(int(0.5 / dt)):
        i = d.update(False, None, None, 640, 480, t); t += dt
    check("0.5 s dropout resets the timer", i.locked_s == 0.0 and i.status == "NO_TARGET")

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(int(6 / dt)):
        i = d.update(True, 600, 100, 640, 480, t); t += dt
    check("still but NOT centred is never ready", (not i.ready) and i.image_stable_s > 5.0 and i.status == "NOT_CENTERED")

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(int(6 / dt)):
        i = d.update(False, None, None, 640, 480, t); t += dt
    check("no detection is never 'stable'", not i.ready and i.status == "NO_TARGET")

    d = StabilityDetector(cfg); t = 0.0
    for _ in range(3):
        i = d.update(True, 320, 240, 640, 480, t); t += dt
    check("timer waits for consecutive detections", i.status == "NO_TARGET" or i.status == "ACQUIRING")
    print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
