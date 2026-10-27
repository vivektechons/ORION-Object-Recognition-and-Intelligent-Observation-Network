"""
round_object_detector.py - colour-agnostic detector for ROUND objects of any colour and any size.

Why this replaces the red-HSV detector: the red detector only answered "is this pixel red?".
Pastel pink, sage green or any non-red balloon was invisible to it, and skin tones leaked in.
This detector never looks at a specific colour. It looks for a closed ring where the colour
changes, and verifies that the ring is a circle.

Pipeline (all steps are colour/size independent):
  1. CANDIDATES: OpenCV HOUGH_GRADIENT_ALT circle search on several "views" of the frame
     (CIELAB L, a, b, chroma, HSV saturation). Pastel objects show up in the chroma views,
     dark/bright ones in L, so at least one view usually sees the rim.
  2. VERIFY + REFINE: for every candidate, sample 72 radial lines across the rim in Lab space,
     find the strongest colour step on each line, then fit a circle to those rim points with a
     robust (Tukey) fit. A hand, a wrist or a knot attached to the object only spoils a few
     rim sectors and is down-weighted, so the centre stays on the object.
  3. SCORE: rim support (fraction of visible rim sectors on the circle), angular coverage,
     fit residual and agreement between views. Size is NOT scored; radius is only searched in
     [round_min_radius_px, round_max_radius_px].
  4. Pick the best circle (small bonus if it is close to last frame's circle).

The score is a heuristic quality number, NOT a probability.

Interface is the same as RedBalloonDetector: detect(frame) -> DetectionResult, reset_tracking().
Run (live demo):   python round_object_detector.py --index 0 --backend MSMF
                   python round_object_detector.py --image photo.jpg
Deps: opencv-python, numpy, config.py, red_balloon_detector.py (dataclasses + overlay)
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import DetectorConfig  # keep before cv2 import
import cv2
import numpy as np

from red_balloon_detector import Detection, DetectionResult, draw_overlay as _draw_base

N_SECTORS = 72
N_BINS = 12


@dataclass(frozen=True)
class RoundDetection(Detection):
    radius: float = 0.0          # fitted radius in pixels (informational; never used as a rule)
    support: float = 0.0         # fraction of visible rim sectors lying on the circle
    coverage: float = 0.0        # fraction of angular bins with rim points
    views: int = 0               # how many image views proposed this circle


def _stretch(x: np.ndarray) -> Optional[np.ndarray]:
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 4.0:            # flat view: nothing to find (and noise would be amplified)
        return None
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _tukey_circle_fit(pts: np.ndarray, c0: Tuple[float, float], r0: float, iters: int = 4):
    """Robust circle fit. Returns (cx, cy, r, rms_of_inliers, inlier_mask)."""
    cx, cy, r = c0[0], c0[1], r0
    w = np.ones(len(pts))
    for _ in range(iters):
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)]) * w[:, None]
        b = (x * x + y * y) * w
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        cx, cy = sol[0], sol[1]
        r2 = sol[2] + cx * cx + cy * cy
        if r2 <= 1:
            break
        r = math.sqrt(r2)
        res = np.hypot(x - cx, y - cy) - r
        scale = max(1.4826 * np.median(np.abs(res)), 0.015 * r, 0.7)
        u = res / (4.685 * scale)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if w.sum() < 8:
            break
    res = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
    inl = np.abs(res) < max(0.06 * r, 2.0)
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.sum() >= 3 else 1e9
    return cx, cy, r, rms, inl


class RoundObjectDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._prev: Optional[Tuple[float, float, float]] = None
        self._miss = 0
        self._theta = np.linspace(0, 2 * np.pi, N_SECTORS, endpoint=False).astype(np.float32)
        self._cos = np.cos(self._theta)
        self._sin = np.sin(self._theta)
        self.last_view: Optional[np.ndarray] = None
        self.last_verified: list = []

    def reset_tracking(self) -> None:
        self._prev = None
        self._miss = 0

    def rebuild_kernels(self) -> None:   # compatibility with RedBalloonDetector
        pass

    # ------------------------------------------------------------ candidates
    def _candidates(self, frame_bgr: np.ndarray):
        c = self.cfg
        s = c.round_work_scale
        small = cv2.resize(frame_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else frame_bgr
        small = cv2.GaussianBlur(small, (5, 5), 0)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        views = {"L": lab[:, :, 0], "a": a, "b": b, "chroma": np.sqrt(a * a + b * b),
                 "S": hsv[:, :, 1].astype(np.float32)}
        min_r = max(2, int(c.round_min_radius_px * s))
        max_r = max(min_r + 2, int(c.round_max_radius_px * s))
        found: List[Tuple[float, float, float, str]] = []
        for name, v in views.items():
            g = _stretch(v)
            if g is None:
                continue
            if name == "chroma":
                self.last_view = cv2.resize(g, (frame_bgr.shape[1], frame_bgr.shape[0]))
            g = cv2.GaussianBlur(g, (7, 7), 1.5)
            circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=max(8, min_r * 2),
                                       param1=300, param2=c.round_hough_param2,
                                       minRadius=min_r, maxRadius=max_r)
            if circles is None:
                continue
            for x, y, r in circles[0][:c.round_max_candidates]:
                found.append((x / s, y / s, r / s, name))
        return found

    # ------------------------------------------------------ verify + refine
    def _verify(self, lab_full: np.ndarray, cx: float, cy: float, r: float):
        """Returns (cx, cy, r, support, coverage, rms, visible, arc_px) or None."""
        h, w = lab_full.shape[:2]
        c = self.cfg
        rho = np.linspace(0.78 * r, 1.22 * r, 23).astype(np.float32)
        delta = max(0.06 * r, 2.5)
        xs = cx + np.outer(self._cos, rho)            # (S, K)
        ys = cy + np.outer(self._sin, rho)
        xi = cx + np.outer(self._cos, rho + delta)
        yi = cy + np.outer(self._sin, rho + delta)
        xo = cx + np.outer(self._cos, rho - delta)
        yo = cy + np.outer(self._sin, rho - delta)

        def samp(X, Y):
            return cv2.remap(lab_full, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(-1000, -1000, -1000))
        out_s, in_s = samp(xi, yi), samp(xo, yo)
        valid = (out_s[:, :, 0] > -500) & (in_s[:, :, 0] > -500)   # both sides inside the image
        de = np.linalg.norm(out_s - in_s, axis=2)
        de[~valid] = 0.0
        best_k = np.argmax(de, axis=1)
        best_de = de[np.arange(N_SECTORS), best_k]
        sector_valid = valid.any(axis=1)
        if sector_valid.sum() < N_SECTORS // 3:
            return None
        strong = sector_valid & (best_de >= c.round_edge_delta_e)
        if strong.sum() < 14:
            return None
        rr = rho[best_k]
        pts = np.column_stack([cx + self._cos * rr, cy + self._sin * rr])[strong]
        fcx, fcy, fr, rms, inl = _tukey_circle_fit(pts, (cx, cy), r)
        if not (c.round_min_radius_px * 0.8 <= fr <= c.round_max_radius_px * 1.2):
            return None
        if not (abs(fcx - cx) < 0.5 * r and abs(fcy - cy) < 0.5 * r):
            return None
        # support: visible sectors whose strongest step lies on the final circle
        ang = np.arctan2(pts[:, 1] - fcy, pts[:, 0] - fcx)
        on_circle = pts[inl]
        n_on = int(inl.sum())
        support = n_on / float(sector_valid.sum())
        bins = np.floor(((ang[inl] % (2 * np.pi)) / (2 * np.pi)) * N_BINS).astype(int)
        coverage = len(set(bins.tolist())) / float(N_BINS) if n_on else 0.0
        visible = float(sector_valid.sum()) / N_SECTORS      # < 1 when the circle is cut by the frame edge
        arc_px = n_on * (2.0 * math.pi * fr / N_SECTORS)     # length of rim verified on the circle
        return fcx, fcy, fr, support, coverage, rms, visible, arc_px

    # ---------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("detect() needs a BGR image with 3 channels")
        c = self.cfg
        h, w = frame_bgr.shape[:2]
        self.last_view = None
        cands = self._candidates(frame_bgr)
        lab_full = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

        verified = []   # (cx, cy, r, support, coverage, rms, visible, arc_px, views)
        for (cx, cy, r, _name) in cands:
            v = self._verify(lab_full, cx, cy, r)
            if v is None:
                continue
            fcx, fcy, fr, sup, cov, rms, vis, arc = v
            if sup < c.round_min_support or cov < c.round_min_coverage:
                continue
            merged = False
            for i, o in enumerate(verified):    # merge duplicates from different views
                if math.hypot(o[0] - fcx, o[1] - fcy) < 0.25 * min(o[2], fr) and abs(o[2] - fr) < 0.2 * fr:
                    best = o if (o[3] * o[4]) >= (sup * cov) else (fcx, fcy, fr, sup, cov, rms, vis, arc, o[8])
                    verified[i] = best[:8] + (o[8] + 1,)
                    merged = True
                    break
            if not merged:
                verified.append((fcx, fcy, fr, sup, cov, rms, vis, arc, 1))

        self.last_verified = verified     # exposed for diagnostics / tuning scripts
        best_det: Optional[RoundDetection] = None
        best_score = -1.0
        for (fcx, fcy, fr, sup, cov, rms, vis, arc, nviews) in verified:
            quality = max(0.0, 1.0 - (rms / fr) / 0.05)
            # amount of verified rim (log scale). This is EVIDENCE, not a size rule: no circle is ever rejected for being
            # small or large; it only decides WHICH circle to follow when several circular things are visible
            # (a long, well-verified rim beats a tiny sharp one such as a camera lens).
            evidence = min(1.0, max(0.0, math.log2(max(arc, 1.0) / 40.0) / 4.0))
            # weights chosen by replaying recorded real frames (see docs/fix_verification.md): agreement between
            # the independent colour views and the amount of verified rim separate the target from stray arcs
            score = (0.15 * sup + 0.10 * cov + 0.10 * vis + 0.10 * quality + 0.25 * min(nviews, 4) / 4.0
                     + 0.30 * evidence)
            score *= 0.5 + 0.5 * vis          # a circle cut by the frame edge is weaker evidence than a whole one
            if self._prev is not None:
                px, py, pr = self._prev
                if math.hypot(fcx - px, fcy - py) < 0.7 * pr and abs(fr - pr) < 0.3 * pr:
                    score += c.round_temporal_bonus
            score = min(score, 1.0)
            if score < c.round_min_score or score <= best_score:
                continue
            best_score = score
            x0, y0 = int(round(fcx - fr)), int(round(fcy - fr))
            best_det = RoundDetection(True, (x0, y0, int(round(2 * fr)), int(round(2 * fr))), float(fcx), float(fcy),
                                      float(math.pi * fr * fr), float(quality), float(score), "TARGET",
                                      float(fr), float(sup), float(cov), int(nviews))

        mask = self.last_view if self.last_view is not None else np.zeros((h, w), np.uint8)
        rejected = max(0, len(cands) - (1 if best_det else 0))
        if best_det is not None:
            self._prev = (best_det.cx, best_det.cy, best_det.radius)
            self._miss = 0
            return DetectionResult(best_det, mask, len(cands), rejected)
        self._miss += 1
        if self._miss > 5:
            self._prev = None
        status = "REJECTED" if cands else "NO_TARGET"
        return DetectionResult(Detection(False, status=status), mask, len(cands), rejected)


def make_detector(cfg: DetectorConfig):
    """Factory used by main.py: 'round' (default) or the legacy 'red' detector."""
    if cfg.mode == "red":
        from red_balloon_detector import RedBalloonDetector
        return RedBalloonDetector(cfg)
    return RoundObjectDetector(cfg)


def draw_overlay(frame, det: Detection, state_text: str, fps: float, deadband_px: float = 0,
                 center_tol_px: float = 0, extra_text: str = ""):
    """Same overlay as before, plus the fitted circle (drawn in place; pass a copy)."""
    _draw_base(frame, det, state_text, fps, deadband_px, center_tol_px, extra_text)
    r = getattr(det, "radius", 0.0)
    if det.found and r > 0:
        cv2.circle(frame, (int(round(det.cx)), int(round(det.cy))), int(round(r)), (0, 255, 255), 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------- demo
def _demo() -> int:
    import argparse
    import time
    from config import ConfigError, load_config
    ap = argparse.ArgumentParser(description="Round-object detector demo")
    ap.add_argument("--index", type=int)
    ap.add_argument("--backend", choices=["MSMF", "DSHOW", "DEFAULT"])
    ap.add_argument("--image", help="run on one image file instead of the camera")
    ap.add_argument("--config")
    a = ap.parse_args()
    try:
        cfg = load_config(a.config)
    except ConfigError as exc:
        print(exc)
        return 1
    det = RoundObjectDetector(cfg.detector)
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print("cannot read", a.image)
            return 1
        s = min(640 / img.shape[1], 480 / img.shape[0])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        res = det.detect(img)
        print(f"{(time.perf_counter() - t0) * 1000:.1f} ms", res.detection)
        cv2.imshow("round object", draw_overlay(img.copy(), res.detection, "IMAGE", 0.0))
        cv2.waitKey(0)
        return 0
    from camera_manager import CameraError, CameraManager
    if a.index is not None:
        cfg.camera.index = a.index
    if a.backend:
        cfg.camera.backend = a.backend
    cam = CameraManager(cfg.camera)
    try:
        cam.open()
    except CameraError as exc:
        print("CAMERA ERROR:", exc)
        return 1
    last = 0
    try:
        while True:
            pkt = cam.wait_for_frame(last, 0.5)
            if pkt is None:
                if not cam.is_running:
                    break
                continue
            last = pkt.frame_id
            t0 = time.perf_counter()
            res = det.detect(pkt.frame)
            ms = (time.perf_counter() - t0) * 1000
            view = draw_overlay(pkt.frame, res.detection, "DEMO", cam.fps(), cfg.tracking.deadband_px,
                                cfg.stability.center_tolerance_px, f"detect {ms:.0f} ms  cand={res.candidate_count}")
            cv2.imshow("round object", view)
            if res.mask is not None:
                cv2.imshow("chroma view", res.mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
