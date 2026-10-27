"""
config.py - all tunable settings as plain dataclasses (no global mutable state).

Run:  python config.py                       -> validates defaults and prints them as JSON
      python config.py --write config.json   -> writes a starter config file
Deps: standard library only.

HARDWARE-SPECIFIC values to check (all marked ASSUMPTION):
  camera.index, camera.backend, tracking.*_direction, tracking angle limits,
  serial.port.

Values that must match the firmware are listed in README.md.
"""
import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Tuple

# Speeds up Windows MSMF camera opening. Must be set before cv2 is imported,
# which is why every module imports config first.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

BACKEND_NAMES = ("MSMF", "DSHOW", "DEFAULT")
FOURCC_NAMES = ("MJPG", "YUY2", "DEFAULT")


class ConfigError(ValueError):
    """Raised for an invalid or unreadable configuration."""


@dataclass
class CameraConfig:
    index: int = 1                     # ASSUMPTION: external webcam is often 1 (0 = laptop camera). Verify with camera_test.py
    backend: str = "MSMF"              # MSMF / DSHOW / DEFAULT (DSHOW measured ~8 FPS on your camera)
    fourcc: str = "MJPG"               # MJPG / YUY2 / DEFAULT
    width: int = 640
    height: int = 480
    target_fps: float = 30.0           # requested only; actual FPS is always measured
    fps_window: int = 30               # frames used for the FPS estimate
    max_consecutive_failures: int = 30
    open_timeout_s: float = 5.0


@dataclass
class DetectorConfig:
    # Red wraps around the OpenCV hue range (0-179), so two ranges are used.
    lower_red_1: Tuple[int, int, int] = (0, 120, 70)
    upper_red_1: Tuple[int, int, int] = (10, 255, 255)
    lower_red_2: Tuple[int, int, int] = (170, 120, 70)
    upper_red_2: Tuple[int, int, int] = (180, 255, 255)
    blur_ksize: int = 5                # odd
    open_ksize: int = 5                # odd
    open_iterations: int = 1
    close_ksize: int = 9               # odd
    close_iterations: int = 2
    min_area_px: float = 400.0
    max_area_fraction: float = 0.5     # reject blobs bigger than this fraction of the frame
    min_circularity: float = 0.60      # 4*pi*area / perimeter^2
    min_aspect: float = 0.5            # bbox width / height limits
    max_aspect: float = 2.0
    min_fill_ratio: float = 0.55       # contour area / bbox area (ellipse ~0.785)
    size_reference_area_px: float = 6000.0
    # weights: circularity, fill ratio, aspect, size (normalised internally)
    score_weights: Tuple[float, float, float, float] = (0.35, 0.20, 0.20, 0.25)
    temporal_radius_px: float = 100.0
    temporal_bonus: float = 0.10
    temporal_forget_frames: int = 5
    min_score: float = 0.55            # heuristic threshold, NOT a probability

    # ---- colour-agnostic round-object detector (round_object_detector.py) ----
    mode: str = "round"                # "round" = any circular object, any colour; "red" = legacy red-balloon detector
    round_min_radius_px: int = 12      # radius search range only; objects are NOT scored by size
    round_max_radius_px: int = 235
    round_work_scale: float = 0.5      # candidate search runs at this fraction of the frame size (speed)
    round_hough_param2: float = 0.70   # HOUGH_GRADIENT_ALT circle "perfectness" for CANDIDATES (verification decides)
    round_edge_delta_e: float = 3.5    # min Lab colour step across the rim for a rim sector to count as supported
    round_min_support: float = 0.45    # fraction of visible rim sectors that must lie on the fitted circle (hand-held objects hide part of the rim)
    round_min_coverage: float = 0.50   # fraction of 12 angular bins that must contain rim points
    round_min_score: float = 0.50      # heuristic quality threshold (NOT a probability)
    round_max_candidates: int = 10     # candidates verified per frame
    round_temporal_bonus: float = 0.15 # score bonus for a circle near last frame's circle (sticks to the target)


@dataclass
class TrackingConfig:
    deadband_px: float = 20.0          # must stay below stability.center_tolerance_px
    kp_pan: float = 0.03               # degrees per pixel of error (conservative)
    kp_tilt: float = 0.03
    pan_direction: int = 1             # ASSUMPTION: +1 or -1, found in Phase 5 calibration
    tilt_direction: int = -1            # ASSUMPTION: +1 or -1, found in Phase 5 calibration
    max_step_deg: float = 4.5          # per-command cap in degrees (legacy; max_speed_dps is the main limit)
    error_filter_alpha: float = 0.6    # 1.0 = no low-pass filtering
    command_rate_hz: float = 30.0      # one command per camera frame (was 20: 1 of 3 measurements was discarded)
    pan_min: float = 45.0              # ASSUMPTION: must be inside firmware limits and your bracket's free range
    pan_max: float = 135.0
    tilt_min: float = 60.0
    tilt_max: float = 120.0
    pan_home: float = 90.0
    tilt_home: float = 90.0
    # ---- latency-compensated "bearing" controller (see tracking_controller.py) ----
    control_mode: str = "bearing"      # "bearing" = aim at the target's absolute bearing; "incremental" = legacy P-step controller
    camera_px_per_deg: float = 10.7    # ASSUMPTION: 640 px / ~60 deg horizontal field of view. Measure: diagnostics/head_calibration.py
    loop_delay_s: float = 0.20         # ASSUMPTION (deliberately on the safe/high side): delay from a servo command to its visible effect in the frame. Measure: diagnostics/head_calibration.py
    follow_gain: float = 0.30          # fraction of the remaining angle error commanded per update. Safe default; raise toward 0.5 only AFTER measuring loop_delay_s and camera_px_per_deg
    max_speed_dps: float = 150.0       # speed cap in degrees/second (independent of the update rate)


@dataclass
class StabilityConfig:
    required_s: float = 5.0            # configurable for testing; sent to firmware with CFG
    position_tolerance_px: float = 25.0
    center_tolerance_px: float = 40.0
    dropout_grace_s: float = 0.3
    min_consecutive_detections: int = 5
    require_centered: bool = True


@dataclass
class SerialConfig:
    port: str = ""                     # e.g. COM5
    baud: int = 115200
    heartbeat_interval_s: float = 0.25
    target_rate_hz: float = 15.0
    ack_timeout_s: float = 1.0
    status_timeout_s: float = 1.5
    write_timeout_s: float = 0.5
    connect_settle_s: float = 2.5      # many ESP32 boards reset when the port opens


@dataclass
class SafetyConfig:
    require_status_fresh: bool = True
    status_max_age_s: float = 1.0      # ESP32 telemetry must be this recent
    min_fps_for_activation: float = 5.0  # camera must deliver at least this measured FPS
    rearm_loss_s: float = 1.5          # target must be absent this long before the host latch clears


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def validate(self) -> None:
        errs = []
        c, d, t, s, sr, sf = (self.camera, self.detector, self.tracking,
                              self.stability, self.serial, self.safety)

        if c.index < 0:
            errs.append("camera.index must be >= 0")
        if c.backend not in BACKEND_NAMES:
            errs.append(f"camera.backend must be one of {BACKEND_NAMES}")
        if c.fourcc not in FOURCC_NAMES:
            errs.append(f"camera.fourcc must be one of {FOURCC_NAMES}")
        if (c.width, c.height) != (640, 480):
            errs.append("camera size must be 640x480 (firmware FRAME_W/FRAME_H are 640x480; change both together)")
        if c.target_fps <= 0 or c.fps_window < 2 or c.max_consecutive_failures < 1 or c.open_timeout_s <= 0:
            errs.append("camera fps/failure/timeout values must be positive (fps_window >= 2)")

        for name in ("blur_ksize", "open_ksize", "close_ksize"):
            v = getattr(d, name)
            if v < 1 or v % 2 == 0:
                errs.append(f"detector.{name} must be an odd integer >= 1")
        for name in ("lower_red_1", "upper_red_1", "lower_red_2", "upper_red_2"):
            h, sat, val = getattr(d, name)
            if not (0 <= h <= 180 and 0 <= sat <= 255 and 0 <= val <= 255):
                errs.append(f"detector.{name} out of HSV range (H 0-180, S/V 0-255)")
        if d.min_area_px <= 0 or not (0 < d.max_area_fraction <= 1):
            errs.append("detector area limits invalid")
        if not (0 < d.min_aspect <= 1 <= d.max_aspect):
            errs.append("detector aspect limits must satisfy 0 < min_aspect <= 1 <= max_aspect")
        if any(w < 0 for w in d.score_weights) or sum(d.score_weights) <= 0:
            errs.append("detector.score_weights must be non-negative with a positive sum")
        if not (0 <= d.min_score <= 1) or d.size_reference_area_px <= 0:
            errs.append("detector.min_score must be 0..1 and size_reference_area_px > 0")
        if d.mode not in ("round", "red"):
            errs.append("detector.mode must be 'round' or 'red'")
        if not (2 <= d.round_min_radius_px < d.round_max_radius_px):
            errs.append("detector.round_min_radius_px must be >= 2 and below round_max_radius_px")
        if not (0.2 <= d.round_work_scale <= 1.0):
            errs.append("detector.round_work_scale must be 0.2..1.0")
        if not (0.3 <= d.round_hough_param2 <= 1.0) or d.round_edge_delta_e <= 0:
            errs.append("detector.round_hough_param2 must be 0.3..1 and round_edge_delta_e > 0")
        if not (0 < d.round_min_support <= 1 and 0 < d.round_min_coverage <= 1 and 0 < d.round_min_score <= 1):
            errs.append("detector.round_min_support/coverage/score must be in (0, 1]")
        if d.round_max_candidates < 1:
            errs.append("detector.round_max_candidates must be >= 1")

        if t.deadband_px < 0 or t.kp_pan <= 0 or t.kp_tilt <= 0:
            errs.append("tracking deadband must be >= 0 and gains must be > 0")
        if t.pan_direction not in (-1, 1) or t.tilt_direction not in (-1, 1):
            errs.append("tracking pan_direction/tilt_direction must be +1 or -1")
        if t.max_step_deg <= 0 or not (0 < t.error_filter_alpha <= 1) or not (1 <= t.command_rate_hz <= 30):
            errs.append("tracking max_step_deg > 0, 0 < alpha <= 1, 1 <= command_rate_hz <= 30")
        if t.control_mode not in ("bearing", "incremental"):
            errs.append("tracking.control_mode must be 'bearing' or 'incremental'")
        if not (2.0 <= t.camera_px_per_deg <= 40.0) or not (0.0 <= t.loop_delay_s <= 1.0):
            errs.append("tracking.camera_px_per_deg must be 2..40 and loop_delay_s 0..1")
        if not (0.05 <= t.follow_gain <= 1.0) or t.max_speed_dps <= 0:
            errs.append("tracking.follow_gain must be 0.05..1 and max_speed_dps > 0")
        if not (t.pan_min < t.pan_max and t.tilt_min < t.tilt_max):
            errs.append("tracking angle limits must have min < max")
        if not (t.pan_min <= t.pan_home <= t.pan_max and t.tilt_min <= t.tilt_home <= t.tilt_max):
            errs.append("tracking home angles must lie inside the limits")

        if s.required_s < 0.1 or s.position_tolerance_px <= 0 or s.center_tolerance_px <= 0 or s.dropout_grace_s < 0:
            errs.append("stability values invalid (required_s >= 0.1, tolerances > 0, grace >= 0)")
        if s.min_consecutive_detections < 1:
            errs.append("stability.min_consecutive_detections must be >= 1")
        if t.deadband_px >= s.center_tolerance_px:
            errs.append("tracking.deadband_px must be smaller than stability.center_tolerance_px")

        if sr.baud <= 0 or sr.ack_timeout_s <= 0 or sr.write_timeout_s <= 0 or sr.connect_settle_s < 0:
            errs.append("serial baud/timeouts invalid")
        if not (0 < sr.heartbeat_interval_s <= 0.5):
            errs.append("serial.heartbeat_interval_s must be in (0, 0.5] (firmware timeout is 1.5 s)")
        if not (1 <= sr.target_rate_hz <= 30):
            errs.append("serial.target_rate_hz must be 1..30")
        if sr.status_timeout_s <= 0 or sf.status_max_age_s <= 0:
            errs.append("status timeouts must be > 0")
        if sf.min_fps_for_activation < 0 or sf.rearm_loss_s <= 0:
            errs.append("safety.min_fps_for_activation must be >= 0 and rearm_loss_s > 0")

        if errs:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))


def _apply(obj, data, path=""):
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a JSON object")
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ConfigError(f"Unknown config key: {path}{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _apply(cur, value, f"{path}{key}.")
        elif isinstance(cur, tuple):
            if not isinstance(value, (list, tuple)) or len(value) != len(cur):
                raise ConfigError(f"{path}{key}: expected a list of {len(cur)} numbers")
            try:
                setattr(obj, key, tuple(type(c)(v) for c, v in zip(cur, value)))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}{key}: {exc}") from exc
        elif isinstance(cur, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{path}{key}: expected true/false")
            setattr(obj, key, value)
        elif isinstance(cur, (int, float, str)):
            if isinstance(cur, str) and not isinstance(value, str):
                raise ConfigError(f"{path}{key}: expected a string")
            if isinstance(cur, int) and isinstance(value, float) and not value.is_integer():
                raise ConfigError(f"{path}{key}: expected an integer")
            if isinstance(value, (bool, list, dict)):
                raise ConfigError(f"{path}{key}: wrong type")
            try:
                setattr(obj, key, type(cur)(value))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}{key}: {exc}") from exc
        else:
            raise ConfigError(f"{path}{key}: unsupported field type")


def default_config_path() -> Path:
    return Path(__file__).with_name("config.json")


def load_config(path=None) -> AppConfig:
    """Load defaults, overlay the JSON file if it exists, validate, return."""
    cfg = AppConfig()
    p = Path(path) if path else default_config_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read {p}: {exc}") from exc
        _apply(cfg, data)
    elif path:
        raise ConfigError(f"Config file not found: {p}")
    cfg.validate()
    return cfg


def save_config(cfg: AppConfig, path=None) -> Path:
    cfg.validate()
    p = Path(path) if path else default_config_path()
    p.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return p


if __name__ == "__main__":
    try:
        conf = AppConfig()
        conf.validate()
        if len(sys.argv) == 3 and sys.argv[1] == "--write":
            print("Wrote", save_config(conf, sys.argv[2]))
        else:
            print(json.dumps(asdict(conf), indent=2))
            print("\nDefault configuration is valid.")
    except ConfigError as exc:
        print(exc)
        sys.exit(1)
