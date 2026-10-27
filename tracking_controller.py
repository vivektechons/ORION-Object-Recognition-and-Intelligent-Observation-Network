"""
tracking_controller.py - pan/tilt controller. Two modes (config tracking.control_mode):

"bearing" (default) - latency-compensated absolute aiming
    Problem with the old incremental P controller: the camera image is ~100-150 ms old by the time it is
    processed, and the head keeps moving during that time. Stepping by "kp * (old error)" therefore
    over-corrects, so the gain had to be kept low (slow) to stay stable. Raising the gain only produces
    oscillation (verified in tests/ simulation in docs/fix_verification.md).
    Fix: convert the pixel error into an ANGLE, add the head angle that was valid when the frame was
    captured, and obtain the target's absolute bearing:
        bearing = head_angle_at(t_frame - loop_delay) + direction * error_px / px_per_deg
    The bearing does not depend on where the head is NOW, so stale measurements no longer cause overshoot.
    Each update then moves a fraction (follow_gain) of the remaining angle toward the bearing, limited by
    max_speed_dps and the angle limits.

"incremental" - the previous controller (kept for A/B comparison).

Sign convention (image coordinates, origin top-left):
    error_x = target_x - frame_center_x   (> 0: target is RIGHT of centre)
    error_y = target_y - frame_center_y   (> 0: target is BELOW centre)
    pan_direction / tilt_direction = +1 or -1: whether a LARGER servo angle moves the camera toward
    +x / +y. If the head moves AWAY from the target on an axis, flip that axis' direction.

Safety: hard min/max angle clamp, speed cap, deadband, hold position when the target is lost.

Run:   python tracking_controller.py     (self-test, no hardware)
Deps:  config.py only.
"""
import collections
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from config import TrackingConfig


@dataclass(frozen=True)
class TrackingOutput:
    pan: float
    tilt: float
    changed: bool                 # True when a new command should be sent
    error_x: Optional[float]      # raw pixel error (None if no target)
    error_y: Optional[float]
    filtered_x: Optional[float]
    filtered_y: Optional[float]
    in_deadband: bool
    at_limit: bool                # a commanded axis is pinned at its angle limit
    bearing_pan: Optional[float] = None    # estimated absolute target bearing in servo degrees (bearing mode)
    bearing_tilt: Optional[float] = None


class TrackingController:
    def __init__(self, cfg: TrackingConfig, frame_w: int = 640, frame_h: int = 480,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._cx0 = frame_w / 2.0
        self._cy0 = frame_h / 2.0
        self._clock = clock
        self._pan = cfg.pan_home
        self._tilt = cfg.tilt_home
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._bp: Optional[float] = None       # filtered bearing
        self._bt: Optional[float] = None
        self._next = 0.0
        self._last_cmd_t: Optional[float] = None
        self._hist: collections.deque = collections.deque(maxlen=256)   # (time, pan, tilt) of commanded angles
        self._hist.append((clock(), self._pan, self._tilt))

    # ------------------------------------------------------------ properties
    @property
    def pan(self) -> float:
        return self._pan

    @property
    def tilt(self) -> float:
        return self._tilt

    def _clamp_pan(self, v: float) -> float:
        return max(self.cfg.pan_min, min(self.cfg.pan_max, v))

    def _clamp_tilt(self, v: float) -> float:
        return max(self.cfg.tilt_min, min(self.cfg.tilt_max, v))

    def set_angles(self, pan: float, tilt: float) -> None:
        """Manual / resync. Always clamped to the configured limits."""
        self._pan = round(self._clamp_pan(pan), 2)
        self._tilt = round(self._clamp_tilt(tilt), 2)
        self._hist.append((self._clock(), self._pan, self._tilt))

    def home(self) -> None:
        self.set_angles(self.cfg.pan_home, self.cfg.tilt_home)
        self.reset_filter()

    def reset_filter(self) -> None:
        self._fx = self._fy = None
        self._bp = self._bt = None

    # --------------------------------------------------------------- helpers
    def _angles_at(self, t: float) -> Tuple[float, float]:
        """Commanded head angles at time t (linear interpolation over the command history)."""
        h = self._hist
        if t <= h[0][0]:
            return h[0][1], h[0][2]
        prev = h[0]
        for cur in h:
            if cur[0] >= t:
                span = cur[0] - prev[0]
                f = 0.0 if span <= 1e-9 else (t - prev[0]) / span
                return prev[1] + f * (cur[1] - prev[1]), prev[2] + f * (cur[2] - prev[2])
            prev = cur
        return h[-1][1], h[-1][2]

    # ---------------------------------------------------------------- update
    def update(self, found: bool, cx: Optional[float], cy: Optional[float],
               t_frame: Optional[float] = None) -> TrackingOutput:
        c = self.cfg
        now = self._clock()
        if not found or cx is None or cy is None:
            self.reset_filter()
            self._hist.append((now, self._pan, self._tilt))
            return TrackingOutput(self._pan, self._tilt, False, None, None, None, None, False, False)
        if c.control_mode == "incremental":
            return self._update_incremental(cx, cy, now)
        return self._update_bearing(cx, cy, now, now if t_frame is None else t_frame)

    # -------------------------------------------------------- bearing control
    def _update_bearing(self, cx: float, cy: float, now: float, t_frame: float) -> TrackingOutput:
        c = self.cfg
        ex, ey = cx - self._cx0, cy - self._cy0
        pan_ref, tilt_ref = self._angles_at(t_frame - c.loop_delay_s)
        bp = pan_ref + c.pan_direction * ex / c.camera_px_per_deg
        bt = tilt_ref + c.tilt_direction * ey / c.camera_px_per_deg
        a = c.error_filter_alpha
        self._bp = bp if self._bp is None else a * bp + (1 - a) * self._bp
        self._bt = bt if self._bt is None else a * bt + (1 - a) * self._bt
        err_pan = self._bp - self._pan             # remaining angle to the target bearing, in degrees
        err_tilt = self._bt - self._tilt
        in_db = (abs(err_pan) * c.camera_px_per_deg <= c.deadband_px and
                 abs(err_tilt) * c.camera_px_per_deg <= c.deadband_px)
        if now < self._next:
            return TrackingOutput(self._pan, self._tilt, False, ex, ey, ex, ey, in_db, False, self._bp, self._bt)
        dt = 1.0 / c.command_rate_hz if self._last_cmd_t is None else min(max(now - self._last_cmd_t, 1e-3), 0.2)
        self._last_cmd_t = now
        self._next = now + 0.9 / c.command_rate_hz
        max_step = min(c.max_speed_dps * dt, c.max_step_deg)

        def step(err: float) -> float:
            if abs(err) * c.camera_px_per_deg <= c.deadband_px:
                return 0.0
            return max(-max_step, min(max_step, c.follow_gain * err))

        wp, wt = self._pan + step(err_pan), self._tilt + step(err_tilt)
        new_pan, new_tilt = round(self._clamp_pan(wp), 2), round(self._clamp_tilt(wt), 2)
        at_limit = (new_pan != round(wp, 2)) or (new_tilt != round(wt, 2))
        changed = abs(new_pan - self._pan) > 1e-6 or abs(new_tilt - self._tilt) > 1e-6
        self._pan, self._tilt = new_pan, new_tilt
        self._hist.append((now, self._pan, self._tilt))
        return TrackingOutput(new_pan, new_tilt, changed, ex, ey, ex, ey, in_db, at_limit, self._bp, self._bt)

    # ----------------------------------------------------- incremental (legacy)
    def _delta(self, err: float, kp: float, direction: int) -> float:
        db = self.cfg.deadband_px
        if abs(err) <= db:
            return 0.0
        mag = min((abs(err) - db) * kp, self.cfg.max_step_deg)
        return direction * (mag if err > 0 else -mag)

    def _update_incremental(self, cx: float, cy: float, now: float) -> TrackingOutput:
        c = self.cfg
        ex, ey = cx - self._cx0, cy - self._cy0
        a = c.error_filter_alpha
        self._fx = ex if self._fx is None else a * ex + (1 - a) * self._fx
        self._fy = ey if self._fy is None else a * ey + (1 - a) * self._fy
        in_db = abs(self._fx) <= c.deadband_px and abs(self._fy) <= c.deadband_px
        if now < self._next:
            return TrackingOutput(self._pan, self._tilt, False, ex, ey, self._fx, self._fy, in_db, False)
        self._next = now + 1.0 / c.command_rate_hz
        wanted_pan = self._pan + self._delta(self._fx, c.kp_pan, c.pan_direction)
        wanted_tilt = self._tilt + self._delta(self._fy, c.kp_tilt, c.tilt_direction)
        new_pan = round(self._clamp_pan(wanted_pan), 2)
        new_tilt = round(self._clamp_tilt(wanted_tilt), 2)
        at_limit = (new_pan != round(wanted_pan, 2)) or (new_tilt != round(wanted_tilt, 2))
        changed = abs(new_pan - self._pan) > 1e-6 or abs(new_tilt - self._tilt) > 1e-6
        self._pan, self._tilt = new_pan, new_tilt
        self._hist.append((now, self._pan, self._tilt))
        return TrackingOutput(new_pan, new_tilt, changed, ex, ey, self._fx, self._fy, in_db, at_limit)


def _self_test() -> int:
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        print(("PASS  " if cond else "FAIL  ") + name)
        ok_all = ok_all and cond

    for mode in ("bearing", "incremental"):
        t = [0.0]
        cfg = TrackingConfig(error_filter_alpha=1.0, control_mode=mode)
        tc = TrackingController(cfg, clock=lambda: t[0])
        t[0] += 1
        o = tc.update(True, 500, 240)
        check(f"[{mode}] target right => pan_direction=+1 increases pan", o.changed and o.pan > cfg.pan_home)
        t[0] += 1
        o = tc.update(True, 320, 400)
        check(f"[{mode}] target below => tilt_direction=+1 increases tilt", o.changed and o.tilt > cfg.tilt_home)
        t[0] += 1
        before = (tc.pan, tc.tilt)
        o = tc.update(False, None, None)
        check(f"[{mode}] target lost => hold position", (not o.changed) and (tc.pan, tc.tilt) == before)
        lim = TrackingController(cfg, clock=lambda: t[0])
        for _ in range(400):
            t[0] += 0.05
            lim.update(True, 640, 480)
        check(f"[{mode}] angle limits never exceeded", lim.pan <= cfg.pan_max and lim.tilt <= cfg.tilt_max)
        inv = TrackingController(TrackingConfig(pan_direction=-1, error_filter_alpha=1.0, control_mode=mode),
                                 clock=lambda: t[0])
        t[0] += 1
        o = inv.update(True, 500, 240)
        check(f"[{mode}] pan_direction=-1 reverses the move", o.pan < 90.0)

    # bearing mode: speed cap independent of update rate, deadband, rate limit
    t = [0.0]
    cfg = TrackingConfig(error_filter_alpha=1.0, max_speed_dps=100.0, follow_gain=1.0)
    tc = TrackingController(cfg, clock=lambda: t[0])
    t[0] = 1.0
    tc.update(True, 640, 240)
    t[0] = 1.0 + 1 / 30
    o = tc.update(True, 640, 240)
    check("[bearing] step <= max_speed_dps * dt", abs(o.pan - cfg.pan_home) <= 100.0 / 30 * 2 + 1e-6)
    t[0] += 1
    tc2 = TrackingController(cfg, clock=lambda: t[0])
    o = tc2.update(True, 330, 245)
    check("[bearing] inside deadband => no command", (not o.changed) and o.in_deadband)
    print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
