"""
red_balloon_detector.py - HSV red-balloon detection, heuristic scoring, overlay drawing.

Pipeline: blur -> HSV -> two red ranges -> open/close -> contours ->
          area / aspect / circularity / fill filters -> heuristic score -> best candidate.

The 'score' is a weighted shape-quality heuristic in 0..1. It is NOT a calibrated
probability and must not be reported as one.

Run (live tuning demo, Phase 4):
    python red_balloon_detector.py --index 1 --backend MSMF
    keys: q = quit, p = print current thresholds
Deps:  opencv-python, numpy, config.py, camera_manager.py (demo only)
Hardware/scene-specific: HSV ranges, min area, lighting. Tune with the demo.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from config import DetectorConfig  # keep before cv2 import
import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    found: bool
    bbox: Optional[Tuple[int, int, int, int]] = None   # x, y, w, h
    cx: Optional[float] = None
    cy: Optional[float] = None
    area: float = 0.0
    circularity: float = 0.0
    score: float = 0.0                                  # heuristic, not a probability
    status: str = "NO_TARGET"                           # TARGET / NO_TARGET / REJECTED


@dataclass(frozen=True)
class DetectionResult:
    detection: Detection
    mask: np.ndarray
    candidate_count: int       # contours that passed the minimum-area test
    rejected_count: int        # of those, rejected by shape filters or score


NO_TARGET = Detection(found=False)


class RedBalloonDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._prev: Optional[Tuple[float, float]] = None
        self._miss = 0
        self._open_k = None
        self._close_k = None
        self.rebuild_kernels()

    def rebuild_kernels(self) -> None:
        """Call after changing kernel sizes in cfg."""
        c = self.cfg
        self._open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.open_ksize, c.open_ksize))
        self._close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.close_ksize, c.close_ksize))

    def reset_tracking(self) -> None:
        self._prev = None
        self._miss = 0

    def _build_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        c = self.cfg
        src = cv2.GaussianBlur(frame_bgr, (c.blur_ksize, c.blur_ksize), 0) if c.blur_ksize > 1 else frame_bgr
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array(c.lower_red_1, np.uint8), np.array(c.upper_red_1, np.uint8))
        m2 = cv2.inRange(hsv, np.array(c.lower_red_2, np.uint8), np.array(c.upper_red_2, np.uint8))
        mask = cv2.bitwise_or(m1, m2)
        if c.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k, iterations=c.open_iterations)
        if c.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_k, iterations=c.close_iterations)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("detect() needs a BGR image with 3 channels")
        c = self.cfg
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        mask = self._build_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        wsum = float(sum(c.score_weights))
        wc, wf, wa, ws = (x / wsum for x in c.score_weights)

        best: Optional[Detection] = None
        candidates = 0
        rejected = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < c.min_area_px:
                continue
            candidates += 1
            if area > c.max_area_fraction * frame_area:
                rejected += 1
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                rejected += 1
                continue
            aspect = bw / float(bh)
            if aspect < c.min_aspect or aspect > c.max_aspect:
                rejected += 1
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                rejected += 1
                continue
            circ = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
            if circ < c.min_circularity:
                rejected += 1
                continue
            fill = area / float(bw * bh)
            if fill < c.min_fill_ratio:
                rejected += 1
                continue

            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = x + bw / 2.0, y + bh / 2.0

            fill_term = 1.0 - min(abs(fill - 0.785) / 0.785, 1.0)   # ellipse fills ~78.5% of its bbox
            aspect_term = min(aspect, 1.0 / aspect)
            size_term = min(area / c.size_reference_area_px, 1.0)
            score = wc * circ + wf * fill_term + wa * aspect_term + ws * size_term
            if self._prev is not None:
                if math.hypot(cx - self._prev[0], cy - self._prev[1]) <= c.temporal_radius_px:
                    score += c.temporal_bonus
            score = min(score, 1.0)
            if score < c.min_score:
                rejected += 1
                continue

            cand = Detection(True, (x, y, bw, bh), cx, cy, float(area), float(circ), float(score), "TARGET")
            if best is None or cand.score > best.score:
                best = cand

        if best is not None:
            self._prev = (best.cx, best.cy)
            self._miss = 0
            return DetectionResult(best, mask, candidates, rejected)

        self._miss += 1
        if self._miss > c.temporal_forget_frames:
            self._prev = None
        status = "REJECTED" if candidates > 0 else "NO_TARGET"
        return DetectionResult(Detection(False, status=status), mask, candidates, rejected)


# ------------------------------------------------------------------ overlay
def _put(img, text, org, color=(255, 255, 255), scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, det: Detection, state_text: str, fps: float,
                 deadband_px: float = 0, center_tol_px: float = 0, extra_text: str = ""):
    """Draw on `frame` IN PLACE (pass a copy) and return it."""
    h, w = frame.shape[:2]
    fx, fy = w // 2, h // 2
    cyan, orange = (255, 255, 0), (0, 165, 255)

    cv2.line(frame, (fx - 15, fy), (fx + 15, fy), cyan, 1)
    cv2.line(frame, (fx, fy - 15), (fx, fy + 15), cyan, 1)
    if deadband_px > 0:
        d = int(deadband_px)
        cv2.rectangle(frame, (fx - d, fy - d), (fx + d, fy + d), cyan, 1)
    if center_tol_px > 0:
        t = int(center_tol_px)
        cv2.rectangle(frame, (fx - t, fy - t), (fx + t, fy + t), (255, 128, 0), 1)

    if det.found and det.bbox is not None:
        x, y, bw, bh = det.bbox
        cx, cy = int(round(det.cx)), int(round(det.cy))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.line(frame, (fx, fy), (cx, fy), orange, 2)          # horizontal error
        cv2.line(frame, (cx, fy), (cx, cy), (0, 100, 255), 2)   # vertical error
        _put(frame, f"ex={cx - fx:+d} ey={cy - fy:+d}", (x, max(15, y - 8)), (0, 255, 0))
        _put(frame, f"heuristic score {det.score:.2f}", (x, y + bh + 18), (0, 255, 0), 0.5)

    _put(frame, f"STATE: {state_text}", (10, 20))
    _put(frame, f"FPS: {fps:.1f}", (10, 42))
    _put(frame, f"DETECT: {det.status}", (10, 64), (0, 255, 0) if det.found else (0, 200, 255))
    if extra_text:
        _put(frame, extra_text, (10, h - 12), (255, 255, 255), 0.5)
    return frame


# --------------------------------------------------------------------- demo
def _demo() -> int:
    import argparse
    from config import load_config, ConfigError
    from camera_manager import CameraManager, CameraError

    ap = argparse.ArgumentParser(description="Live red-balloon detection and HSV tuning demo")
    ap.add_argument("--index", type=int)
    ap.add_argument("--backend", choices=["MSMF", "DSHOW", "DEFAULT"])
    ap.add_argument("--fourcc", choices=["MJPG", "YUY2", "DEFAULT"])
    ap.add_argument("--config")
    a = ap.parse_args()

    try:
        cfg = load_config(a.config)
    except ConfigError as exc:
        print(exc)
        return 1
    if a.index is not None:
        cfg.camera.index = a.index
    if a.backend:
        cfg.camera.backend = a.backend
    if a.fourcc:
        cfg.camera.fourcc = a.fourcc

    cam = CameraManager(cfg.camera)
    try:
        cam.open()
    except CameraError as exc:
        print("CAMERA ERROR:", exc)
        return 1

    det = RedBalloonDetector(cfg.detector)
    dc = cfg.detector
    win = "tune"
    cv2.namedWindow(win)
    nop = lambda _v: None
    cv2.createTrackbar("S min", win, dc.lower_red_1[1], 255, nop)
    cv2.createTrackbar("V min", win, dc.lower_red_1[2], 255, nop)
    cv2.createTrackbar("Hue1 max", win, dc.upper_red_1[0], 30, nop)
    cv2.createTrackbar("Hue2 min", win, dc.lower_red_2[0], 180, nop)
    cv2.createTrackbar("Min area", win, int(dc.min_area_px), 5000, nop)

    last_id = 0
    try:
        while True:
            smin = cv2.getTrackbarPos("S min", win)
            vmin = cv2.getTrackbarPos("V min", win)
            h1 = cv2.getTrackbarPos("Hue1 max", win)
            h2 = max(150, cv2.getTrackbarPos("Hue2 min", win))
            dc.lower_red_1 = (0, smin, vmin)
            dc.upper_red_1 = (h1, 255, 255)
            dc.lower_red_2 = (h2, smin, vmin)
            dc.upper_red_2 = (180, 255, 255)
            dc.min_area_px = float(max(1, cv2.getTrackbarPos("Min area", win)))

            pkt = cam.wait_for_frame(last_id, timeout=0.5)
            if pkt is None:
                if not cam.is_running:
                    print("CAMERA ERROR:", cam.status().error or "camera stopped")
                    break
                continue
            last_id = pkt.frame_id
            res = det.detect(pkt.frame)
            view = draw_overlay(pkt.frame, res.detection, "DEMO", cam.fps(),
                                cfg.tracking.deadband_px, cfg.stability.center_tolerance_px,
                                f"candidates={res.candidate_count} rejected={res.rejected_count}")
            cv2.imshow("detection", view)
            cv2.imshow("mask", res.mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print(f"S>={smin} V>={vmin} hue1 0-{h1} hue2 {h2}-180 min_area={dc.min_area_px:.0f}")
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
