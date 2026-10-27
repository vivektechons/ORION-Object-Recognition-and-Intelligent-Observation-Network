"""
safety_manager.py - host-side interlocks. This is the SECOND safety layer only:
the ESP32 firmware independently validates every actuation and the physical e-stop
cuts servo power. Python can only REQUEST activation.

Activation is allowed only when ALL of these are true (evaluate() lists failures):
  serial link up | telemetry fresh | ESP32 itself reports READY with a challenge |
  no e-stop | PCA9685 OK | user explicitly armed | camera running at a sane FPS |
  tracking enabled | target currently detected | host stability test passed |
  no activation request in flight | one-shot latch clear.

One-shot latch (fixed in this version): v1.0 set the latch the moment ACT was QUEUED. If the ESP32
rejected that ACT (stale READY, wrong challenge, interlock lost) the latch stayed set and the host
never tried again until the object disappeared for 1.5 s: the "works only once" symptom. The latch is
now set ONLY when the ESP32 answers ACK to the ACT, and it is cleared as soon as the ESP32 reports it
has left ACTIVATING/LOCKOUT (the ESP32 decides when to re-arm). A rejected ACT just allows a retry
with the new challenge after a short cool-down.

Run:   python safety_manager.py     (self-test)
Deps:  config.py, serial_communication.py (EspStatus only)
"""
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from config import SafetyConfig
from serial_communication import EspStatus


@dataclass(frozen=True)
class SafetyInputs:
    link_connected: bool
    status: Optional[EspStatus]
    status_age_s: float
    camera_ok: bool
    camera_fps: float
    tracking_enabled: bool
    target_found: bool
    host_ready: bool
    now: float
    act_state: str = "NONE"       # NONE / PENDING / CONFIRMED / REJECTED / TIMEOUT (what the ESP32 answered to ACT)
    act_seq: int = 0
    act_age_s: float = 1e9


@dataclass(frozen=True)
class SafetyDecision:
    allow_activate: bool
    reasons: Tuple[str, ...]
    warning: str
    level: str          # ok / info / warn / critical
    app_state: str


class SafetyManager:
    ARM_GRACE_S = 1.5   # time allowed for the ESP32 to confirm ARM before intent is dropped

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._user_armed = False
        self._arm_grace_until = 0.0
        self._latched = False
        self._latched_seq = 0
        self._latch_t = 0.0

    @property
    def user_armed(self) -> bool:
        with self._lock:
            return self._user_armed

    def note_armed(self, armed: bool, now: float = 0.0) -> None:
        with self._lock:
            self._user_armed = armed
            self._arm_grace_until = now + self.ARM_GRACE_S if armed else 0.0
            self._latched = False

    def can_arm(self, i: SafetyInputs) -> Tuple[bool, str]:
        if not i.link_connected:
            return False, "Connect the ESP32 first"
        st = i.status
        if st is None or i.status_age_s > self.cfg.status_max_age_s:
            return False, "No fresh ESP32 telemetry"
        if st.estop or st.state == "ESTOP":
            return False, "Emergency stop is active: release it, then press RESET"
        if st.state == "FAULT":
            return False, f"ESP32 fault ({st.reason}): fix it, then press RESET"
        if st.state != "DISARMED":
            return False, f"ESP32 is {st.state}, not DISARMED"
        if not st.pca_ok:
            return False, "PCA9685 not responding"
        if not i.camera_ok:
            return False, "Connect the camera first"
        return True, "OK"

    def evaluate(self, i: SafetyInputs) -> SafetyDecision:
        st = i.status
        stale = st is None or i.status_age_s > self.cfg.status_max_age_s
        with self._lock:
            # sync intent with what the ESP32 really reports
            if (not stale and st.state in ("DISARMED", "ESTOP", "FAULT")
                    and i.now >= self._arm_grace_until):
                self._user_armed = False
                self._latched = False
            # the latch follows what the ESP32 ANSWERED, never what we merely queued
            if i.act_state == "CONFIRMED" and i.act_seq != self._latched_seq:
                self._latched, self._latched_seq, self._latch_t = True, i.act_seq, i.now
            if (self._latched and not stale and st.state not in ("ACTIVATING", "LOCKOUT")
                    and i.now - self._latch_t > 0.6):
                self._latched = False                   # the ESP32 has re-armed itself
            armed, latched = self._user_armed, self._latched

        reasons = []
        if not i.link_connected:
            reasons.append("ESP32 not connected")
        if stale:
            reasons.append("ESP32 telemetry stale")
        else:
            if st.state != "READY":
                reasons.append(f"ESP32 state is {st.state}")
            if st.challenge == 0:
                reasons.append("no ESP32 challenge")
            if st.estop:
                reasons.append("e-stop active")
            if not st.pca_ok:
                reasons.append("PCA9685 fault")
        if not armed:
            reasons.append("not armed by user")
        if not i.camera_ok:
            reasons.append("camera not running")
        elif i.camera_fps < self.cfg.min_fps_for_activation:
            reasons.append(f"camera FPS {i.camera_fps:.1f} too low")
        if not i.tracking_enabled:
            reasons.append("tracking disabled")
        if not i.target_found:
            reasons.append("no target")
        if not i.host_ready:
            reasons.append("stability test not passed")
        if i.act_state == "PENDING" and i.act_age_s < 1.5:
            reasons.append("activation request in flight")
        if i.act_state == "REJECTED" and i.act_age_s < 0.25:
            reasons.append("retry cool-down after the ESP32 rejected ACT")
        if latched:
            reasons.append("one-shot latch (waiting for the ESP32 to re-arm)")
        allow = len(reasons) == 0

        if not i.link_connected:
            app_state = "NO LINK"
        elif stale:
            app_state = "NO TELEMETRY"
        else:
            app_state = st.state

        if not stale and (st.estop or st.state == "ESTOP"):
            warning, level = "EMERGENCY STOP ACTIVE: release the e-stop switch, then press RESET", "critical"
        elif not stale and st.state == "FAULT":
            warning, level = f"SAFETY FAULT: {st.reason}. Fix the cause, then press RESET", "critical"
        elif not i.link_connected:
            warning, level = "ESP32 NOT CONNECTED: actuator disabled", "warn"
        elif stale:
            warning, level = "ESP32 telemetry lost: actuator disabled", "warn"
        elif st.state == "DISARMED":
            warning, level = "SYSTEM DISARMED: actuator disabled", "warn"
        elif st.state == "READY":
            warning, level = "READY: activation can occur", "critical"
        elif st.state == "ACTIVATING":
            warning, level = "ACTIVATING", "critical"
        elif st.state == "LOCKOUT":
            warning, level = "LOCKOUT: move the object away or press Re-arm", "info"
        else:
            warning, level = "ARMED: tracking", "info"
        return SafetyDecision(allow, tuple(reasons), warning, level, app_state)


def _self_test() -> int:
    import time
    from config import SafetyConfig as SC
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        print(("PASS  " if cond else "FAIL  ") + name)
        ok_all = ok_all and cond

    def stat(state="READY", estop=False, chal=12345, pca=True, t=None):
        return EspStatus(state, 90, 90, 20, estop, state != "DISARMED", 100, 5000, 5000, chal, 0, pca, "NONE",
                         time.monotonic() if t is None else t)

    def inp(st, now=100.0, **kw):
        base = dict(link_connected=True, status=st, status_age_s=0.1, camera_ok=True, camera_fps=30.0,
                    tracking_enabled=True, target_found=True, host_ready=True, now=now)
        base.update(kw)
        return SafetyInputs(**base)

    sm = SafetyManager(SC())
    check("starts NOT armed => no activation", not sm.evaluate(inp(stat())).allow_activate)
    sm.note_armed(True, 100.0)
    check("all conditions met => allowed", sm.evaluate(inp(stat())).allow_activate)
    check("e-stop blocks", not sm.evaluate(inp(stat(estop=True))).allow_activate)
    check("no challenge blocks", not sm.evaluate(inp(stat(chal=0))).allow_activate)
    check("stale telemetry blocks", not sm.evaluate(inp(stat(), status_age_s=5.0)).allow_activate)
    check("tracking disabled blocks", not sm.evaluate(inp(stat(), tracking_enabled=False)).allow_activate)
    check("host stability not passed blocks", not sm.evaluate(inp(stat(), host_ready=False)).allow_activate)
    check("low FPS blocks", not sm.evaluate(inp(stat(), camera_fps=1.0)).allow_activate)
    check("link down blocks", not sm.evaluate(inp(stat(), link_connected=False)).allow_activate)
    # --- activation bookkeeping: driven by the ESP32's answer, not by queueing ---
    d = sm.evaluate(inp(stat(), now=100.5, act_state="PENDING", act_seq=7, act_age_s=0.1))
    check("activation in flight blocks a second ACT", not d.allow_activate)
    d = sm.evaluate(inp(stat(), now=100.9, act_state="REJECTED", act_seq=7, act_age_s=0.1))
    check("ESP32 rejection does NOT latch (the old 'works only once' bug)", "one-shot latch (waiting for the ESP32 to re-arm)" not in d.reasons)
    d = sm.evaluate(inp(stat(), now=101.4, act_state="REJECTED", act_seq=7, act_age_s=0.6))
    check("after the cool-down a retry is allowed", d.allow_activate)
    d = sm.evaluate(inp(stat("ACTIVATING"), now=102.0, act_state="CONFIRMED", act_seq=8, act_age_s=0.05))
    check("ESP32 ACK latches", "one-shot latch (waiting for the ESP32 to re-arm)" in d.reasons)
    d = sm.evaluate(inp(stat("LOCKOUT"), now=103.0, act_state="CONFIRMED", act_seq=8, act_age_s=1.0))
    check("stays latched while the ESP32 is in LOCKOUT", not d.allow_activate)
    d = sm.evaluate(inp(stat("READY"), now=104.0, act_state="CONFIRMED", act_seq=8, act_age_s=2.0))
    check("latch released when the ESP32 has re-armed and is READY again", d.allow_activate)
    d = sm.evaluate(inp(stat("READY"), now=104.2, act_state="CONFIRMED", act_seq=8, act_age_s=2.2))
    check("stale CONFIRMED of the old request does not re-latch", d.allow_activate)
    sm.evaluate(inp(stat("DISARMED"), now=110.0))
    check("ESP32 DISARMED clears user arm", not sm.user_armed)
    d = sm.evaluate(inp(stat("ESTOP", estop=True), now=111.0))
    check("e-stop shows critical warning", d.level == "critical")
    print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
