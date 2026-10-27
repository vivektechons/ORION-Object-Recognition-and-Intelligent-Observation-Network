"""
main.py - application controller and entry point.

Wires camera, detector, tracking, stability, serial link and safety together.
A processing thread runs the vision/control loop; the Tk GUI (main thread) only reads
immutable snapshots and calls the thread-safe methods of Application.

Run:   python main.py
       python main.py --index 1 --backend MSMF --port COM5 --config config.json
Deps:  opencv-python, numpy, pyserial, tkinter (bundled with Python), all project modules.
Hardware-specific: camera index/backend (config or flags), ESP32 COM port.
The application ALWAYS starts DISARMED with tracking OFF.
"""
import argparse
import csv
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Tuple

from config import AppConfig, ConfigError, load_config  # must precede cv2 import
import cv2

from camera_manager import BACKENDS, CameraError, CameraManager
from round_object_detector import draw_overlay, make_detector
from safety_manager import SafetyDecision, SafetyInputs, SafetyManager
from serial_communication import SerialCommunicator, list_serial_ports
from stability_detector import StabilityDetector
from tracking_controller import TrackingController


# ------------------------------------------------------------------ logging
class EventLog:
    """Thread-safe in-memory message list (for the GUI) that also writes to the log file."""

    def __init__(self, maxlen: int = 300):
        self._items: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("orion")

    def add(self, level: str, msg: str) -> None:
        level = level.upper()
        self._logger.log(getattr(logging, level, logging.INFO), msg)
        with self._lock:
            self._items.append((time.strftime("%H:%M:%S"), level, msg))

    def recent(self, n: int = 50) -> List[Tuple[str, str, str]]:
        with self._lock:
            return list(self._items)[-n:]


def setup_logging() -> Path:
    log_dir = Path(__file__).with_name("logs")
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "orion.log"
    root = logging.getLogger("orion")
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        fh = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(sh)
    return path


# ----------------------------------------------------------------- snapshot
@dataclass
class AppSnapshot:
    frame_id: int = 0
    ppm: Optional[bytes] = None            # PPM image bytes for Tk PhotoImage
    app_state: str = "NO LINK"
    warning: str = "SYSTEM DISARMED: actuator disabled"
    level: str = "warn"
    detection_status: str = "NO_TARGET"
    target_text: str = "-"
    error_text: str = "-"
    stability_status: str = "NO_TARGET"
    stable_s: float = 0.0
    required_s: float = 5.0
    progress: float = 0.0
    pan_cmd: float = 90.0
    tilt_cmd: float = 90.0
    pan_rep: Optional[float] = None
    tilt_rep: Optional[float] = None
    actuator_text: str = "-"
    fps: float = 0.0
    proc_ms: float = 0.0
    detect_ms: float = 0.0
    camera_text: str = "Camera: not connected"
    camera_ok: bool = False
    serial_text: str = "ESP32: not connected"
    serial_state: str = "DISCONNECTED"
    tracking_enabled: bool = False
    armed: bool = False
    estop: bool = False
    esp_state: str = "-"


# -------------------------------------------------------------- application
class Application:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.log = EventLog()
        self.camera = CameraManager(cfg.camera)
        self.detector = make_detector(cfg.detector)       # round (any colour) by default; "red" = legacy
        self.tracker = TrackingController(cfg.tracking, cfg.camera.width, cfg.camera.height)
        self.stability = StabilityDetector(cfg.stability)
        self.comm = SerialCommunicator(cfg.serial, self.log.add)
        self.safety = SafetyManager(cfg.safety)
        self.comm.set_required_stability_s(cfg.stability.required_s)

        self._lock = threading.RLock()
        self._tracking = False                       # always starts OFF
        self._snapshot = AppSnapshot(required_s=cfg.stability.required_s)
        self._stop = threading.Event()
        self._down = False
        self._last_cam_error = ""
        self._proc_ms = 0.0
        self._perf_t = time.monotonic()
        self._perf_frames = 0
        self._perf_found = 0
        self._detect_ms = 0.0
        self._csv = None
        self._csv_file = None
        self._last_act_logged = 0
        self._last_loop_error = 0.0
        self._thread = threading.Thread(target=self._loop, name="processing", daemon=True)
        self._thread.start()
        self.log.add("INFO", "Application started: DISARMED, tracking OFF")

    # --------------------------------------------------------- GUI-facing API
    @staticmethod
    def backend_names() -> List[str]:
        return list(BACKENDS)

    @staticmethod
    def serial_ports() -> List[Tuple[str, str]]:
        return list_serial_ports()

    def get_snapshot(self) -> AppSnapshot:
        return self._snapshot

    def recent_messages(self, n: int = 50):
        return self.log.recent(n)

    def connect_camera(self, index: int, backend: str) -> Tuple[bool, str]:
        """Blocking (up to the camera open timeout): call from a worker thread."""
        try:
            self.cfg.camera.index = int(index)
            self.cfg.camera.backend = backend
            self.cfg.validate()
        except (ValueError, ConfigError) as exc:
            self.log.add("ERROR", f"Invalid camera settings: {exc}")
            return False, str(exc)
        self.set_tracking(False)
        try:
            self.camera.open()
        except CameraError as exc:
            self.log.add("ERROR", str(exc))
            return False, str(exc)
        self.detector.reset_tracking()
        self.stability.reset()
        self._last_cam_error = ""
        self.log.add("INFO", f"Camera {index} ({backend}) opened")
        return True, "Camera connected"

    def disconnect_camera(self) -> None:
        self.set_tracking(False)
        self.camera.close()
        self.log.add("INFO", "Camera disconnected")

    def connect_serial(self, port: str) -> Tuple[bool, str]:
        self.cfg.serial.port = port
        ok, msg = self.comm.connect(port)
        self.log.add("INFO" if ok else "ERROR", msg)
        return ok, msg

    def disconnect_serial(self) -> None:
        self.set_tracking(False)
        self.safety.note_armed(False)
        self.comm.disconnect()
        self.log.add("INFO", "ESP32 disconnected")

    def set_tracking(self, enabled: bool) -> None:
        with self._lock:
            self._tracking = bool(enabled)
        if not enabled:
            self.stability.reset()
            self.tracker.reset_filter()
            self._close_csv()
        else:
            self._open_csv()
        self.log.add("INFO", f"Tracking {'ENABLED' if enabled else 'DISABLED'}")

    def set_required_stability(self, seconds: float) -> Tuple[bool, str]:
        if self.safety.user_armed:
            return False, "Disarm before changing the stability time"
        if not (0.1 <= seconds <= 60.0):
            return False, "Stability time must be 0.1 .. 60 s"
        self.cfg.stability.required_s = float(seconds)
        self.comm.set_required_stability_s(seconds)
        self.log.add("INFO", f"Stability time set to {seconds:.1f} s")
        return True, "OK"

    def _inputs(self, now: float, tracking: bool, found: bool, host_ready: bool) -> SafetyInputs:
        cam = self.camera.status()
        act_state, act_seq, act_age, _code = self.comm.act_info()
        return SafetyInputs(
            link_connected=self.comm.is_connected(), status=self.comm.get_status(),
            status_age_s=self.comm.status_age(), camera_ok=cam.connected, camera_fps=cam.fps,
            tracking_enabled=tracking, target_found=found, host_ready=host_ready, now=now,
            act_state=act_state, act_seq=act_seq, act_age_s=act_age)

    def arm(self) -> Tuple[bool, str]:
        now = time.monotonic()
        with self._lock:
            tracking = self._tracking
        ok, why = self.safety.can_arm(self._inputs(now, tracking, False, False))
        if not ok:
            self.log.add("WARNING", f"ARM refused: {why}")
            return False, why
        self.stability.reset()
        ok, msg = self.comm.arm()
        if not ok:
            self.log.add("WARNING", f"ARM refused: {msg}")
            return False, msg
        self.safety.note_armed(True, now)
        self.log.add("WARNING", "System ARMED by user")
        return True, "ARM requested"

    def disarm(self) -> None:
        self.safety.note_armed(False)
        self.stability.reset()
        self.comm.disarm()
        self.log.add("INFO", "System DISARMED by user")

    def emergency_stop(self) -> None:
        self.set_tracking(False)
        self.safety.note_armed(False)
        self.comm.emergency_stop()

    def reset(self) -> Tuple[bool, str]:
        ok, msg = self.comm.reset()
        if ok:
            self.safety.note_armed(False)
            st = self.comm.get_status()
            if st is not None:
                self.tracker.set_angles(st.pan, st.tilt)
            self.log.add("INFO", "RESET requested (refused by the ESP32 while the physical e-stop is pressed)")
        else:
            self.log.add("WARNING", f"RESET refused: {msg}")
        return ok, msg

    def rearm(self) -> Tuple[bool, str]:
        """Explicit re-arm after an actuation (the ESP32 also re-arms by itself, see firmware REARM_* settings)."""
        ok, msg = self.comm.rearm()
        self.stability.reset()
        self.log.add("INFO" if ok else "WARNING", "Re-arm requested" if ok else f"Re-arm refused: {msg}")
        return ok, msg

    def go_home(self) -> None:
        self.tracker.home()
        self.comm.set_angles(self.tracker.pan, self.tracker.tilt)
        self.log.add("INFO", "Pan/tilt commanded to home")

    # ------------------------------------------------------ tracking CSV log
    CSV_COLUMNS = ["t_s", "capture_fps", "found", "cx", "cy", "radius", "score", "err_x", "err_y",
                   "bearing_pan", "bearing_tilt", "cmd_pan", "cmd_tilt", "rep_pan", "rep_tilt",
                   "esp_state", "stable_s", "detect_ms", "proc_ms", "act_state", "act_count"]

    def _open_csv(self) -> None:
        if self._csv is not None:
            return
        try:
            d = Path(__file__).with_name("logs")
            d.mkdir(exist_ok=True)
            path = d / time.strftime("tracking_%Y%m%d_%H%M%S.csv")
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(self.CSV_COLUMNS)
            self._csv_t0 = time.monotonic()
            self.log.add("INFO", f"Tracking CSV log: {path}")
        except OSError as exc:
            self._csv = self._csv_file = None
            self.log.add("WARNING", f"Cannot open tracking CSV log: {exc}")

    def _close_csv(self) -> None:
        f, self._csv, self._csv_file = self._csv_file, None, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass

    def _write_csv(self, t, fps, det, out, stab) -> None:
        if self._csv is None:
            return
        st = self.comm.get_status()
        act_state = self.comm.act_info()[0]
        f = lambda v, n=2: "" if v is None else round(float(v), n)
        try:
            self._csv.writerow([
                round(t - self._csv_t0, 3), f(fps, 1), int(det.found), f(det.cx, 1), f(det.cy, 1),
                f(getattr(det, "radius", None), 1), f(det.score, 3),
                f(out.error_x, 1) if out else "", f(out.error_y, 1) if out else "",
                f(out.bearing_pan) if out else "", f(out.bearing_tilt) if out else "",
                f(self.tracker.pan), f(self.tracker.tilt),
                f(st.pan, 1) if st else "", f(st.tilt, 1) if st else "", st.state if st else "",
                f(stab.locked_s), f(self._detect_ms, 1), f(self._proc_ms, 1), act_state, st.act_count if st else ""])
            if int(t * 10) % 20 == 0:
                self._csv_file.flush()
        except (OSError, ValueError):
            self._close_csv()

    def _log_act_results(self) -> None:
        state, seq, age, code = self.comm.act_info()
        if seq != self._last_act_logged and state in ("CONFIRMED", "REJECTED", "TIMEOUT"):
            self._last_act_logged = seq
            st = self.comm.get_status()
            if state == "CONFIRMED":
                self.log.add("CRITICAL", f"ESP32 confirmed activation (total {st.act_count if st else '?'})")
            elif state == "REJECTED":
                self.log.add("WARNING", f"ESP32 rejected ACT ({code}); it will retry with the new challenge if still READY")
            else:
                self.log.add("WARNING", "ESP32 did not answer ACT within 1.5 s")

    # -------------------------------------------------------- processing loop
    def _loop(self) -> None:
        last_id = 0
        while not self._stop.is_set():
            try:
                if not self.camera.is_running:
                    self._idle()
                    time.sleep(0.05)
                    continue
                pkt = self.camera.wait_for_frame(last_id, 0.25)
                if pkt is None:
                    self._idle(keep_camera_text=True)
                    continue
                last_id = pkt.frame_id
                self._process(pkt)
            except Exception as exc:  # keep the loop alive; report at most once per second
                now = time.monotonic()
                if now - self._last_loop_error > 1.0:
                    self._last_loop_error = now
                    self.log.add("ERROR", f"Processing error: {type(exc).__name__}: {exc}")
                time.sleep(0.05)

    def _camera_text(self, cam) -> str:
        if cam.connected:
            return f"Camera {cam.index} {cam.backend} {cam.width}x{cam.height} @ {cam.fps:.1f} FPS"
        return f"Camera ERROR: {cam.error}" if cam.error else "Camera: not connected"

    def _watch_camera_error(self, cam) -> None:
        if cam.error and cam.error != self._last_cam_error:
            self._last_cam_error = cam.error
            self.log.add("CRITICAL", cam.error)
            self.set_tracking(False)

    def _idle(self, keep_camera_text: bool = False) -> None:
        now = time.monotonic()
        cam = self.camera.status()
        self._watch_camera_error(cam)
        self.stability.reset()
        self.comm.set_target(False, 0, 0, 0)
        with self._lock:
            tracking = self._tracking
        decision = self.safety.evaluate(self._inputs(now, tracking, False, False))
        self._publish(None, 0, decision, cam, None, None, self.stability.info(now), tracking)

    def _process(self, pkt) -> None:
        t0 = time.monotonic()
        res = self.detector.detect(pkt.frame)
        self._detect_ms = 0.9 * self._detect_ms + 0.1 * ((time.monotonic() - t0) * 1000.0) if self._detect_ms else (time.monotonic() - t0) * 1000.0
        det = res.detection
        h, w = pkt.frame.shape[:2]
        with self._lock:
            tracking = self._tracking

        if tracking:
            stab = self.stability.update(det.found, det.cx, det.cy, w, h, t0)
        else:
            self.stability.reset()
            stab = self.stability.info(t0)

        # the ESP32 only ever hears about a target while tracking is enabled
        self.comm.set_target(det.found and tracking, det.cx or 0, det.cy or 0, int(stab.locked_s * 1000))

        out = None
        if tracking:
            out = self.tracker.update(det.found, det.cx, det.cy, t_frame=pkt.timestamp)
            if out.changed:
                self.comm.set_angles(out.pan, out.tilt)

        cam = self.camera.status()
        self._watch_camera_error(cam)
        decision = self.safety.evaluate(self._inputs(t0, tracking, det.found, stab.ready))
        if decision.allow_activate:
            ok, msg = self.comm.activate()
            if ok:
                self.log.add("CRITICAL", "ACTIVATION requested (all interlocks satisfied); waiting for the ESP32's answer")
            else:
                self.log.add("WARNING", f"Activation not sent: {msg}")
        self._log_act_results()

        extra = f"stable {stab.locked_s:.1f}/{stab.required_s:.1f}s [{stab.status}]"
        view = draw_overlay(pkt.frame, det, decision.app_state, cam.fps,
                            self.cfg.tracking.deadband_px, self.cfg.stability.center_tolerance_px, extra)
        self._proc_ms = 0.9 * self._proc_ms + 0.1 * ((time.monotonic() - t0) * 1000.0) if self._proc_ms else (time.monotonic() - t0) * 1000.0
        self._publish(view, pkt.frame_id, decision, cam, det, out, stab, tracking)
        self._write_csv(t0, cam.fps, det, out, stab)
        self._perf_log(cam.fps, det.found)

    def _perf_log(self, fps: float, found: bool) -> None:
        self._perf_frames += 1
        self._perf_found += 1 if found else 0
        now = time.monotonic()
        if now - self._perf_t >= 5.0:
            self.log.add("DEBUG", f"perf: capture_fps={fps:.1f} processed_fps={self._perf_frames / (now - self._perf_t):.1f} "
                                  f"proc_ms={self._proc_ms:.1f} detected_in={100.0 * self._perf_found / self._perf_frames:.0f}% of frames")
            self._perf_t, self._perf_frames, self._perf_found = now, 0, 0

    def _publish(self, view, frame_id, decision: SafetyDecision, cam, det, out, stab, tracking) -> None:
        ppm = None
        if view is not None:
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            ppm = b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()
        st = self.comm.get_status()
        info = self.comm.info()
        target_text = error_text = "-"
        det_status = "NO_TARGET"
        if det is not None:
            det_status = det.status
            if det.found:
                r = getattr(det, "radius", 0.0)
                target_text = f"x={det.cx:.0f}  y={det.cy:.0f}  r={r:.0f}  score {det.score:.2f}" if r else \
                              f"x={det.cx:.0f}  y={det.cy:.0f}  (area {det.area:.0f})"
        if out is not None and out.error_x is not None:
            error_text = f"ex={out.error_x:+.0f}  ey={out.error_y:+.0f}"
        if st is not None and info.status_age_s < 2.0:
            actuator = {"ACTIVATING": "ACTIVATING", "LOCKOUT": "LOCKOUT (returned to rest)",
                        "READY": "READY (waiting)"}.get(st.state, f"REST ({st.act:.0f} deg)")
            if st.state == "ESTOP":
                actuator = "DISABLED (e-stop)"
            if st.state == "DISARMED":
                actuator = f"DISABLED (disarmed), {st.act:.0f} deg"
        else:
            actuator = "UNKNOWN (no telemetry)"
        if st is not None:
            act_state = self.comm.act_info()[0]
            actuator += f"  |  pops {st.act_count}"
            if act_state in ("PENDING", "REJECTED", "TIMEOUT"):
                actuator += f"  |  ACT {act_state.lower()}"
        if info.conn_state == "CONNECTED":
            serial_text = f"ESP32: connected on {info.port} (telemetry {info.status_age_s:.1f}s ago, rx {info.rx}, err {info.errs})"
        elif info.conn_state == "ERROR":
            serial_text = f"ESP32 ERROR: {info.error}"
        elif info.conn_state == "CONNECTING":
            serial_text = f"ESP32: connecting to {info.port}..."
        else:
            serial_text = "ESP32: not connected"
        snap = AppSnapshot(
            frame_id=frame_id, ppm=ppm, app_state=decision.app_state, warning=decision.warning,
            level=decision.level, detection_status=det_status, target_text=target_text,
            error_text=error_text, stability_status=stab.status, stable_s=stab.locked_s,
            required_s=stab.required_s, progress=stab.progress, pan_cmd=self.tracker.pan,
            tilt_cmd=self.tracker.tilt, pan_rep=None if st is None else st.pan,
            tilt_rep=None if st is None else st.tilt, actuator_text=actuator, fps=cam.fps,
            proc_ms=self._proc_ms, detect_ms=self._detect_ms, camera_text=self._camera_text(cam), camera_ok=cam.connected,
            serial_text=serial_text, serial_state=info.conn_state, tracking_enabled=tracking,
            armed=self.safety.user_armed, estop=bool(st.estop) if st else False,
            esp_state=st.state if st else "-")
        self._snapshot = snap

    # --------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        if self._down:
            return
        self._down = True
        self.log.add("INFO", "Shutting down safely")
        try:
            self.set_tracking(False)
            self.safety.note_armed(False)
            self.comm.disarm()
        except Exception:
            pass
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._close_csv()
        self.comm.disconnect()
        self.camera.close()


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="ORION — Object Recognition and Intelligent Observation Network")
    ap.add_argument("--config", help="path to a JSON config file (default: config.json if present)")
    ap.add_argument("--index", type=int, help="camera index override")
    ap.add_argument("--backend", choices=list(BACKENDS), help="camera backend override")
    ap.add_argument("--port", help="ESP32 COM port to preselect, e.g. COM5")
    args = ap.parse_args()

    log_path = setup_logging()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1
    if args.index is not None:
        cfg.camera.index = args.index
    if args.backend:
        cfg.camera.backend = args.backend
    if args.port:
        cfg.serial.port = args.port
    try:
        cfg.validate()
    except ConfigError as exc:
        print(exc)
        return 1

    import tkinter as tk
    from gui import ControlPanel

    app = Application(cfg)
    app.log.add("INFO", f"Log file: {log_path}")
    root = tk.Tk()
    ControlPanel(root, app, cfg)
    try:
        root.mainloop()
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
