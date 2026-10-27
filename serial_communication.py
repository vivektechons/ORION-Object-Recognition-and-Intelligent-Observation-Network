"""
serial_communication.py - framed, checksummed, sequenced link to the ESP32 firmware.

Frame:   $BODY*XX\n      XX = 2-digit hex XOR of every character of BODY
Host->ESP32 bodies (seq is strictly increasing, assigned under the write lock):
    HELLO,seq   HB,seq   STATUS,seq   SET,seq,pan,tilt   TGT,seq,det,cx,cy,hostStableMs
    ARM,seq   DISARM,seq   ACT,seq,challenge   REARM,seq   STOP,seq   RESET,seq   CFG,seq,stableMs
ESP32->host:  ACK,seq,STATE   ERR,seq,CODE   STAT,...(14 fields)   BOOT,name,version

Threading: one I/O thread owns reading, the heartbeat, target reports and queued commands.
Public methods are thread-safe. STOP and DISARM are written IMMEDIATELY from the calling
thread (they never wait in a queue) and STOP is retried until the ESP32 reports ESTOP.
connect() returns at once; the port is opened and the handshake runs in the I/O thread,
so the GUI never blocks.

Run:   not run directly (see diagnostics/serial_test.py)
Deps:  pyserial, config.py
Hardware-specific: SerialConfig.port (COM port of the ESP32), baud (must match firmware).
"""
import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import serial
from serial.tools import list_ports

from config import SerialConfig

FRAME_W = 640   # must match firmware FRAME_W
FRAME_H = 480   # must match firmware FRAME_H

DISCONNECTED = "DISCONNECTED"
CONNECTING = "CONNECTING"
CONNECTED = "CONNECTED"
ERROR = "ERROR"


# ------------------------------------------------------------------ framing
def checksum(body: str) -> int:
    x = 0
    for b in body.encode("latin-1", errors="replace"):
        x ^= b
    return x


def build_frame(body: str) -> bytes:
    return f"${body}*{checksum(body):02X}\n".encode("ascii")


def parse_frame(line: str) -> Optional[List[str]]:
    """Return the comma-split fields of a valid frame, or None if malformed / bad checksum."""
    line = line.strip()
    if not line.startswith("$"):
        return None
    star = line.rfind("*")
    if star < 0 or len(line) - star != 3:
        return None
    body = line[1:star]
    try:
        rx = int(line[star + 1:], 16)
    except ValueError:
        return None
    if checksum(body) != rx:
        return None
    return body.split(",")


def list_serial_ports() -> List[Tuple[str, str]]:
    return [(p.device, p.description) for p in list_ports.comports()]


# --------------------------------------------------------------- data types
@dataclass(frozen=True)
class EspStatus:
    state: str
    pan: float
    tilt: float
    act: float
    estop: bool
    armed: bool
    hb_age_ms: int
    stable_ms: int
    stable_req_ms: int
    challenge: int
    act_count: int
    pca_ok: bool
    reason: str
    rx_time: float          # time.monotonic() when received


@dataclass(frozen=True)
class CommInfo:
    conn_state: str
    port: str
    error: str
    status_age_s: float
    tx: int
    rx: int
    bad_frames: int
    acks: int
    errs: int
    last_esp_error: str


@dataclass(frozen=True)
class TargetReport:
    det: bool
    cx: float
    cy: float
    host_stable_ms: int
    t: float


def _parse_stat(f: List[str], now: float) -> Optional[EspStatus]:
    if len(f) != 14:
        return None
    try:
        return EspStatus(f[1], float(f[2]), float(f[3]), float(f[4]), f[5] == "1", f[6] == "1",
                         int(f[7]), int(f[8]), int(f[9]), int(f[10]), int(f[11]), f[12] == "1", f[13], now)
    except ValueError:
        return None


class SerialCommunicator:
    def __init__(self, cfg: SerialConfig, log: Optional[Callable[[str, str], None]] = None):
        self.cfg = cfg
        self._log_cb = log or (lambda level, msg: None)
        self._lock = threading.RLock()
        self._wlock = threading.Lock()
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._conn = DISCONNECTED
        self._error = ""
        self._port = ""
        self._seq = 0
        self._status: Optional[EspStatus] = None
        self._pending: "collections.OrderedDict[int, Tuple[str, float]]" = collections.OrderedDict()
        self._ctrl: collections.deque = collections.deque()
        self._target: Optional[TargetReport] = None
        self._angles: Optional[Tuple[float, float]] = None
        self._angles_dirty = False
        self._angles_sent_t = 0.0
        self._stop_req = False
        self._stop_sent_t = 0.0
        self._hello_seq: Optional[int] = None
        self._hello_ok = False
        self._stable_ms = 5000
        self._rxbuf = bytearray()
        self._last_log: dict = {}
        self._tx = self._rx = self._bad = self._acks = self._errs = 0
        self._last_esp_error = ""
        self._act_seq = 0                   # sequence number of the last ACT frame actually written
        self._act_state = "NONE"            # NONE / PENDING / CONFIRMED / REJECTED
        self._act_code = ""
        self._act_t = 0.0

    # ----------------------------------------------------------------- misc
    def _log(self, level: str, msg: str, key: Optional[str] = None, interval: float = 0.0) -> None:
        if key is not None and interval > 0:
            now = time.monotonic()
            if now - self._last_log.get(key, -1e9) < interval:
                return
            self._last_log[key] = now
        try:
            self._log_cb(level, msg)
        except Exception:
            pass

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._conn = ERROR
            self._error = msg
        self._log("ERROR", msg)

    def _wait(self, seconds: float) -> bool:
        return not self._stop_evt.wait(seconds)

    # ------------------------------------------------------------- low level
    def _send(self, name: str, args=()) -> int:
        """Assign the next sequence number and write one frame. Raises on I/O errors."""
        with self._wlock:
            ser = self._ser
            if ser is None or not ser.is_open:
                raise serial.SerialException("port is not open")
            self._seq += 1
            seq = self._seq
            body = ",".join([name, str(seq), *[str(a) for a in args]])
            ser.write(build_frame(body))
            with self._lock:
                self._pending[seq] = (name, time.monotonic())
                while len(self._pending) > 64:
                    self._pending.popitem(last=False)
                self._tx += 1
            return seq

    def _send_safe(self, name: str, args=()) -> bool:
        """Public-thread send: on I/O failure mark the link as ERROR instead of raising."""
        try:
            self._send(name, args)
            return True
        except (serial.SerialException, OSError) as exc:
            self._set_error(f"Serial write failed: {exc}")
            self._stop_evt.set()
            return False

    def _read_available(self) -> None:
        ser = self._ser
        if ser is None:
            return
        n = ser.in_waiting
        data = ser.read(n if n else 1)          # blocks up to the 20 ms port timeout
        if data:
            self._rxbuf.extend(data)
        while True:
            idx = self._rxbuf.find(b"\n")
            if idx < 0:
                break
            line = bytes(self._rxbuf[:idx])
            del self._rxbuf[:idx + 1]
            self._handle_line(line)
        if len(self._rxbuf) > 1024:
            self._rxbuf.clear()

    def _handle_line(self, raw: bytes) -> None:
        text = raw.decode("latin-1", errors="replace")
        pos = text.rfind("$")
        if pos < 0:
            return                              # boot-ROM noise, not a frame
        f = parse_frame(text[pos:])
        if f is None:
            with self._lock:
                self._bad += 1
            self._log("WARNING", "Corrupted frame from ESP32 ignored", "badframe", 5.0)
            return
        with self._lock:
            self._rx += 1
        kind = f[0]
        now = time.monotonic()
        if kind == "STAT":
            st = _parse_stat(f, now)
            if st is not None:
                with self._lock:
                    self._status = st
        elif kind == "ACK" and len(f) >= 3:
            try:
                seq = int(f[1])
            except ValueError:
                return
            with self._lock:
                self._acks += 1
                self._pending.pop(seq, None)
            if seq == self._hello_seq:
                self._hello_ok = True
            if seq == self._act_seq and seq != 0:
                with self._lock:
                    self._act_state, self._act_code, self._act_t = "CONFIRMED", "", now
        elif kind == "ERR" and len(f) >= 3:
            try:
                seq = int(f[1])
            except ValueError:
                return
            with self._lock:
                self._errs += 1
                name = self._pending.pop(seq, ("?", 0.0))[0]
                if seq != 0:
                    self._last_esp_error = f"{name}: {f[2]}"
            if seq == self._act_seq and seq != 0:
                with self._lock:
                    self._act_state, self._act_code, self._act_t = "REJECTED", f[2], now
            if seq == 0:
                self._log("WARNING", f"ESP32 reported a bad frame from the host ({f[2]})", "esp_bad", 5.0)
            else:
                self._log("WARNING", f"ESP32 rejected {name}: {f[2]}", f"rej_{name}_{f[2]}", 5.0)
        elif kind == "BOOT":
            ver = f[2] if len(f) > 2 else "?"
            if self._conn == CONNECTED:
                self._log("WARNING", "ESP32 rebooted (brownout or reset). It is DISARMED; arm again if intended.")
            else:
                self._log("INFO", f"ESP32 firmware {ver} reported boot")

    # ---------------------------------------------------------- connection
    def connect(self, port: str) -> Tuple[bool, str]:
        if not port:
            return False, "No serial port selected"
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "Serial link already connecting or connected"
            self._conn = CONNECTING
            self._error = ""
            self._port = port
            self._stop_evt.clear()
            self._status = None
            self._ctrl.clear()
            self._pending.clear()
            self._stop_req = False
            self._target = None
            self._hello_seq = None
            self._hello_ok = False
            self._rxbuf.clear()
            self._thread = threading.Thread(target=self._run, args=(port,), name="serial-io", daemon=True)
            t = self._thread
        t.start()
        return True, f"Connecting to {port}..."

    def disconnect(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)

    def is_connected(self) -> bool:
        with self._lock:
            return self._conn == CONNECTED

    def _run(self, port: str) -> None:
        ser = None
        try:
            try:
                ser = serial.Serial(port=port, baudrate=self.cfg.baud, timeout=0.02,
                                    write_timeout=self.cfg.write_timeout_s)
            except (serial.SerialException, OSError, ValueError) as exc:
                self._set_error(f"Cannot open {port}: {exc}. Is another program (Arduino Serial Monitor) using it?")
                return
            with self._lock:
                self._ser = ser
            self._log("INFO", f"Opened {port} at {self.cfg.baud} baud; waiting {self.cfg.connect_settle_s:.1f} s for the ESP32")
            if not self._wait(self.cfg.connect_settle_s):
                return
            ser.reset_input_buffer()
            self._rxbuf.clear()
            if not self._handshake():
                if not self._stop_evt.is_set():
                    self._set_error("ESP32 did not answer HELLO. Check the COM port, baud rate and that the firmware is flashed.")
                return
            with self._lock:
                self._conn = CONNECTED
            self._log("INFO", "ESP32 link established (HELLO acknowledged)")
            self._loop()
        except (serial.SerialException, OSError) as exc:
            if not self._stop_evt.is_set() or self._conn != DISCONNECTED:
                self._set_error(f"Serial connection lost: {exc}")
        finally:
            self._close_port(ser)

    def _handshake(self) -> bool:
        for _ in range(3):
            if self._stop_evt.is_set():
                return False
            self._hello_ok = False
            self._hello_seq = self._send("HELLO")
            deadline = time.monotonic() + self.cfg.ack_timeout_s
            while time.monotonic() < deadline and not self._stop_evt.is_set():
                self._read_available()
                if self._hello_ok:
                    return True
        return False

    def _close_port(self, ser) -> None:
        if ser is not None:
            try:
                if ser.is_open:
                    try:
                        self._send("DISARM")        # best effort; the firmware watchdog also disarms
                    except Exception:
                        pass
                    ser.close()
            except Exception:
                pass
        with self._lock:
            self._ser = None
            if self._conn != ERROR:
                self._conn = DISCONNECTED

    def _loop(self) -> None:
        hb_int = self.cfg.heartbeat_interval_s
        tgt_int = 1.0 / self.cfg.target_rate_hz
        next_hb = next_tgt = 0.0
        while not self._stop_evt.is_set():
            self._read_available()
            now = time.monotonic()

            if self._stop_req:                       # STOP has priority and is retried until confirmed
                st = self._status
                if st is not None and st.state == "ESTOP" and st.rx_time >= self._stop_sent_t:
                    self._stop_req = False
                elif now - self._stop_sent_t >= 0.2:
                    self._stop_sent_t = now
                    self._send("STOP")

            with self._lock:
                items = list(self._ctrl)
                self._ctrl.clear()
            for name, args in items:
                if self._stop_req and name in ("ACT", "ARM", "REARM"):
                    continue
                seq = self._send(name, args)
                if name == "ACT":                   # remember which frame it was, to learn what the ESP32 answered
                    with self._lock:
                        self._act_seq, self._act_state, self._act_code, self._act_t = seq, "PENDING", "", time.monotonic()

            if now >= next_hb:
                self._send("HB")
                next_hb = now + hb_int
            if now >= next_tgt:
                self._send_target(now)
                next_tgt = now + tgt_int
            self._send_angles(now)

    def _send_target(self, now: float) -> None:
        t = self._target
        if t is None or now - t.t > 0.5 or not t.det:
            self._send("TGT", (0, 0, 0, 0))          # stale or no detection never counts as a target
            return
        cx = int(max(0, min(FRAME_W, round(t.cx))))
        cy = int(max(0, min(FRAME_H, round(t.cy))))
        self._send("TGT", (1, cx, cy, int(max(0, min(600000, t.host_stable_ms)))))

    def _send_angles(self, now: float) -> None:
        with self._lock:
            a = self._angles
            dirty = self._angles_dirty
            st = self._status
        if a is None:
            return
        age = now - self._angles_sent_t
        keepalive = age >= 1.0 and not (st is not None and st.state in ("ESTOP", "FAULT"))
        if (dirty and age >= 0.03) or keepalive:
            self._send("SET", (f"{a[0]:.1f}", f"{a[1]:.1f}"))
            with self._lock:
                if self._angles == a:
                    self._angles_dirty = False
            self._angles_sent_t = now

    # ------------------------------------------------------------ public API
    def set_required_stability_s(self, seconds: float) -> None:
        with self._lock:
            self._stable_ms = int(max(100, min(60000, round(seconds * 1000))))

    def set_target(self, det: bool, cx: float, cy: float, host_stable_ms: int) -> None:
        with self._lock:
            self._target = TargetReport(bool(det), float(cx or 0), float(cy or 0), int(host_stable_ms), time.monotonic())

    def set_angles(self, pan: float, tilt: float) -> None:
        with self._lock:
            self._angles = (round(pan, 1), round(tilt, 1))
            self._angles_dirty = True

    def arm(self) -> Tuple[bool, str]:
        with self._lock:
            if self._conn != CONNECTED:
                return False, "ESP32 not connected"
            if self._stop_req:
                return False, "Emergency stop is pending"
            self._ctrl.append(("CFG", (self._stable_ms,)))
            self._ctrl.append(("ARM", ()))
        return True, "ARM requested"

    def rearm(self) -> Tuple[bool, str]:
        """Explicit re-arming after an actuation (ESP32 only accepts it in LOCKOUT)."""
        with self._lock:
            if self._conn != CONNECTED:
                return False, "ESP32 not connected"
            if self._stop_req:
                return False, "Emergency stop is pending"
            self._ctrl.append(("REARM", ()))
        return True, "REARM requested"

    def act_info(self) -> Tuple[str, int, float, str]:
        """(state, seq, age_s, reject_code) of the last activation request.
        state: NONE / PENDING / CONFIRMED / REJECTED / TIMEOUT (no answer within 1.5 s)."""
        with self._lock:
            state, seq, t, code = self._act_state, self._act_seq, self._act_t, self._act_code
        age = time.monotonic() - t if t else 1e9
        if state == "PENDING" and age > 1.5:
            state = "TIMEOUT"
        return state, seq, age, code

    def disarm(self) -> bool:
        with self._lock:
            ok = self._conn == CONNECTED
            self._ctrl = collections.deque((n, a) for n, a in self._ctrl if n not in ("ACT", "ARM"))
        return ok and self._send_safe("DISARM")

    def activate(self) -> Tuple[bool, str]:
        """Request one actuation. Only possible while the ESP32 itself reports READY with a challenge."""
        with self._lock:
            if self._conn != CONNECTED:
                return False, "ESP32 not connected"
            if self._stop_req:
                return False, "Emergency stop pending"
            st = self._status
            if st is None or time.monotonic() - st.rx_time > self.cfg.status_timeout_s:
                return False, "ESP32 telemetry is stale"
            if st.state != "READY" or st.challenge == 0:
                return False, f"ESP32 is not READY (state {st.state})"
            self._ctrl.append(("ACT", (st.challenge,)))
        return True, "ACT queued"

    def emergency_stop(self) -> bool:
        with self._lock:
            connected = self._conn == CONNECTED
            self._ctrl.clear()
            if connected:
                self._stop_req = True
        if not connected:
            self._log("CRITICAL", "No ESP32 link: use the HARDWARE e-stop switch")
            return False
        self._stop_sent_t = time.monotonic()
        self._log("CRITICAL", "EMERGENCY STOP sent")
        return self._send_safe("STOP")

    def reset(self) -> Tuple[bool, str]:
        with self._lock:
            if self._conn != CONNECTED:
                return False, "ESP32 not connected"
            self._stop_req = False
            self._ctrl.append(("RESET", ()))
        return True, "RESET requested"

    def get_status(self) -> Optional[EspStatus]:
        with self._lock:
            return self._status

    def status_age(self) -> float:
        with self._lock:
            st = self._status
        return float("inf") if st is None else time.monotonic() - st.rx_time

    def info(self) -> CommInfo:
        age = self.status_age()
        with self._lock:
            return CommInfo(self._conn, self._port, self._error, age, self._tx, self._rx,
                            self._bad, self._acks, self._errs, self._last_esp_error)
