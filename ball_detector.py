"""Ball detector — YOLO + motion-based white-blob fallback + Kalman smoothing.

Strategy:
1. Use the shared YOLODetector's full-frame ball detections (fast path).
2. When YOLO fails, crop to pitch + upscale and re-run YOLO for small balls.
3. When that fails, frame-differencing + white-blob detection for distant ball.
4. Non-green moving blob (catches balls of ANY color).
5. Kalman filter for temporal smoothing + gap-filling (up to 10 frames).
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np


class BallKalmanFilter:
    """Kalman filter for ball tracking with a constant-velocity motion model.

    State:  [px, py, vx, vy]^T    (position + velocity in pixel-space)
    Meas.:  [px, py]^T             (observed ball position)

    One predict-update cycle per frame:
      predict()   — advance state by one frame-step, return (px, py) or None
      update(px, py) — correct state with a detected ball position

    The velocity term is the key improvement over a position-holder:
    during gaps the ball keeps moving in its last direction instead of
    freezing in place, and the Kalman gate rejects detections that would
    require implausible acceleration.
    """

    def __init__(self, process_noise=2.0, measurement_noise=25.0):
        # State transition: [px+vx, py+vy, vx, vy]  (dt=1 frame-step)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # Measurement: we directly observe px, py
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        # Process noise — higher = reacts faster to ball kicks/acceleration
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        # Measurement noise — higher = smoother but slower to react
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        self.x = None   # state vector (None = uninitialized)
        self.P = np.eye(4, dtype=np.float32) * 500.0  # initial uncertainty

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def initialized(self):
        return self.x is not None

    def predict(self):
        """Advance state by one frame-step. Returns (px, py) or None."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0]), float(self.x[1]))

    def update(self, px, py):
        """Correct state with a measured ball position."""
        if self.x is None:
            self.x = np.array([px, py, 0.0, 0.0], dtype=np.float32)
            return

        z = np.array([px, py], dtype=np.float32)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

    def reset(self):
        self.x = None
        self.P = np.eye(4, dtype=np.float32) * 500.0


class BallDetector:
    """Detects football using YOLO + motion-blob fallback + position tracking.

    Shares the main YOLO model (no second full-frame inference). For small-ball
    detection it can optionally run a tiny YOLO on an upscaled pitch crop.
    """

    _stationary_threshold = 20   # pixels — if ball stays in a circle this small
    _stationary_frames = 15      # for this many consecutive frames → reject

    def __init__(self, yolo_detector, trail_length=30,
                 crop_model_path="__auto__",
                 blob_min_radius=2, blob_max_radius=10,
                 upscale_target=1024, max_upscale=2.0,
                 kalman_gate_px=400,
                 zoom_crop_size=500):
        """
        Args:
            yolo_detector: Shared YOLODetector instance (full-frame inference).
            crop_model_path: Path to ball-specific YOLO model.
                             "__auto__" (default) auto-selects the best available
                             model (ball_detector_yolo26m.pt > soccana_yolo11n.pt).
                             None = skip crop YOLO entirely.
            upscale_target: When running YOLO on pitch crop, upscale so longest
                            edge is this many px (better small-ball detection).
        """
        # Auto-select best available crop model
        if crop_model_path == "__auto__" or crop_model_path is None:
            candidates = [
                ("ball_detector_yolo26m.engine", False),    # TensorRT — fastest, clean license
                ("ball_detector_yolo26m.pt", False),
                ("soccana_yolo11n.pt", False),
            ]
            for path, required in candidates:
                if Path(path).exists():
                    crop_model_path = path
                    break
            if crop_model_path == "__auto__":
                crop_model_path = None  # no model found, run without crop YOLO

        self.detector = yolo_detector
        self.trail = deque(maxlen=trail_length)
        self.zoom_crop_size = zoom_crop_size
        self.kf = BallKalmanFilter()
        self._gate_px = kalman_gate_px
        self._missed_count = 0
        self._prev_crop_gray = None
        self._polygon = None
        self._crop_roi = None           # (x1, y1, x2, y2) in original frame
        self._recent_positions = deque(maxlen=10)
        self._boundary_count = 0
        self._trail_history = deque(maxlen=15)  # (cx, cy, source, conf, frame_idx) — for hysteresis

        # Optional crop-based YOLO model for small-ball detection
        self._crop_yolo = None
        self._crop_ball_id = 1  # default: soccana convention (Player=0, Ball=1, Referee=2)
        self.blob_min_r = blob_min_radius
        self.blob_max_r = blob_max_radius
        self.upscale_target = upscale_target
        self.max_upscale = max_upscale
        if crop_model_path:
            try:
                from ultralytics import YOLO
                self._crop_yolo = YOLO(crop_model_path)
                # Auto-detect ball class ID: 0 for single-class ball model,
                # 1 for soccana 3-class (Player=0, Ball=1, Referee=2)
                nc = len(self._crop_yolo.names)
                self._crop_ball_id = 0 if nc == 1 else 1
                print(f"  BallDetector: crop YOLO model loaded ({crop_model_path})"
                      f" — {nc} classes, ball_id={self._crop_ball_id}")
            except Exception as e:
                print(f"  BallDetector: crop YOLO unavailable ({e})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, polygon=None, frame_idx=0, yolo_ball_xy=None):
        """Detect ball position.

        Args:
            frame: BGR frame.
            polygon: Pitch polygon for filtering.
            frame_idx: Current frame number.
            yolo_ball_xy: Optional (cx, cy, conf) from full-frame YOLO.
                          If provided and valid, skip the expensive fallback pipeline.

        Returns: (cx, cy, confidence) or None.
        """
        if polygon is not None:
            self._polygon = polygon

        # Predict ball position for this frame using velocity-based Kalman filter.
        # This one call replaces the old "store last position" approach — with
        # velocity the ball keeps moving during brief gaps instead of freezing.
        kf_pred = self.kf.predict()

        # Step 0 — full-frame YOLO ball detections are the fast path.
        # If the shared detector already found a ball, validate and use it.
        if yolo_ball_xy is not None:
            cx, cy, conf = yolo_ball_xy
            accepted = self._accept_ball(cx, cy, conf, source="yolo",
                                         kf_pred=kf_pred, frame_idx=frame_idx)
            if accepted is not None:
                return accepted
            # YOLO ball rejected by filters — fall through to crop/blob pipeline

        # Step 1 — Prediction-guided zoom crop YOLO (when Kalman is active).
        # Instead of searching the entire pitch for a ~6px ball, crop a tight
        # window around where the ball should be and upscale aggressively.
        # The ball becomes ~15-20px in the upscaled crop — YOLO can actually see it.
        if kf_pred is not None and self._crop_yolo is not None:
            zoom_crop, zoom_roi = self._predictive_zoom_crop(frame, kf_pred)
            if zoom_crop is not None:
                gray_zoom = cv2.cvtColor(zoom_crop, cv2.COLOR_BGR2GRAY)
                result = self._detect_yolo_on_crop(zoom_crop, gray_zoom,
                                                   crop_roi=zoom_roi)
                if result is not None:
                    zcx, zcy, zconf = self._map_to_original(
                        result[0], result[1], result[2], zoom_roi)
                    accepted = self._accept_ball(zcx, zcy, zconf, source="yolo_zoom",
                                                 kf_pred=kf_pred, frame_idx=frame_idx)
                    if accepted is not None:
                        self._prev_crop_gray = gray_zoom
                        return accepted

        # Step 2 — crop to pitch for focused ball search
        if self._polygon is not None:
            crop_frame, roi = self._crop_to_pitch(frame)
        else:
            crop_frame, roi = frame, (0, 0, frame.shape[1], frame.shape[0])
        self._crop_roi = roi

        if crop_frame is None or crop_frame.size == 0:
            if kf_pred is not None:
                self.trail.append((kf_pred[0], kf_pred[1]))
            return kf_pred

        gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

        # Step 3 — YOLO on upscaled pitch crop (better for distant small ball)
        if self._crop_yolo is not None:
            result = self._detect_yolo_on_crop(crop_frame, gray)
            if result is not None:
                cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
                accepted = self._accept_ball(cx, cy, conf, source="yolo_crop",
                                             kf_pred=kf_pred, frame_idx=frame_idx)
                if accepted is not None:
                    self._prev_crop_gray = gray
                    return accepted

        # Step 4 — Kalman-guided blob search (sensitive, near prediction)
        if kf_pred is not None:
            result = self._detect_near_prediction(crop_frame, gray, roi, kf_pred)
            if result is not None:
                cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
                accepted = self._accept_ball(cx, cy, conf, source="blob_near",
                                             kf_pred=kf_pred, frame_idx=frame_idx)
                if accepted is not None:
                    self._prev_crop_gray = gray
                    return accepted

        # Step 5 — wide white-blob search (no Kalman guidance)
        result = self._detect_motion_blob(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = self._accept_ball(cx, cy, conf, source="blob_wide",
                                         kf_pred=kf_pred, frame_idx=frame_idx)
            if accepted is not None:
                self._prev_crop_gray = gray
                return accepted

        # Step 6 — non-green moving blob (balls of ANY color)
        result = self._detect_motion_any_color(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = self._accept_ball(cx, cy, conf, source="blob_anycolor",
                                         kf_pred=kf_pred, frame_idx=frame_idx)
            if accepted is not None:
                self._prev_crop_gray = gray
                return accepted

        self._prev_crop_gray = gray

        # Step 7 — Kalman prediction for gap-filling
        self._missed_count += 1
        if self._missed_count < 15 and kf_pred is not None:
            px, py = kf_pred[0], kf_pred[1]
            if self._polygon is not None:
                pt_test = cv2.pointPolygonTest(
                    self._polygon.astype(np.float32), (float(px), float(py)), True)
                if pt_test < -200:
                    self._missed_count = 15
                    return None
            self.trail.append((px, py))
            return (px, py, 0.0)

        if self._missed_count >= 15:
            self.kf.reset()
        return None

    def reset(self):
        self.kf.reset()
        self._prev_crop_gray = None
        self.trail.clear()
        self._missed_count = 0
        self._boundary_count = 0

    # ------------------------------------------------------------------
    # Ball acceptance gate — shared across all detection strategies
    # ------------------------------------------------------------------

    # Source reliability tiers — higher tier = stricter acceptance
    # Only YOLO (tier 0) can start or restart a ball track.
    # Blob detections (tier 2) are only accepted when actively tracking.
    _SOURCE_TIER = {
        "yolo_zoom": 0,      # YOLO on predictive zoom crop — ball is 15-20px, very reliable
        "yolo": 0,           # YOLO on full frame — most reliable
        "yolo_crop": 1,      # YOLO on upscaled pitch crop
        "blob_near": 1,      # Motion blob near Kalman prediction
        "blob_wide": 2,      # White motion blob anywhere
        "blob_anycolor": 2,  # Any-colour motion blob
    }

    def _accept_ball(self, cx, cy, conf, source="", kf_pred=None, frame_idx=0):
        """Validate a ball candidate through all rejection filters.

        Returns (cx, cy, conf) if accepted, None if rejected.
        """
        # 1 — Inside pitch polygon
        if self._polygon is not None:
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(cx), float(cy)), True)
            if dist < -10:
                return None
            # Boundary filter: near-edge detections need high conf
            if dist < 10 and not (source == "yolo" and conf >= 0.5):
                return None

        # 2 — Kalman gate: reject detections far from predicted position.
        # With velocity estimation, this now accounts for ball movement,
        # so a fast-moving ball won't be rejected simply because it moved.
        if self.kf.initialized and kf_pred is not None:
            dx = cx - kf_pred[0]
            dy = cy - kf_pred[1]
            if dx * dx + dy * dy > self._gate_px * self._gate_px:
                return None

        # 3 — Stationary rejection (forgive high-confidence detections)
        if conf < 0.4 and self._is_stationary(cx, cy):
            return None

        # 5 — Source-tier hysteresis: blob detections can't start/restart tracks.
        #     Lower-tier sources merely sustain an already-active track.
        tier = self._SOURCE_TIER.get(source, 2)
        if tier >= 2:   # blob_wide, blob_anycolor
            # Must be actively tracking (no missed frames) with history
            if not self.kf.initialized or self._missed_count > 0:
                return None
            if len(self._trail_history) < 3:
                return None
        elif tier >= 1:  # yolo_crop, blob_near
            # Crop YOLO and blob_near can restart a lost track up to a point
            if not self.kf.initialized:
                if source != "yolo_crop" and source != "blob_near":
                    return None
            elif self._missed_count > 10:
                return None

        # — Accepted —
        self._missed_count = 0
        self.kf.update(cx, cy)
        # Use Kalman-corrected position for smooth trail (filters out jitter)
        kx = float(self.kf.x[0])
        ky = float(self.kf.x[1])
        self.trail.append((kx, ky))
        self._recent_positions.append((kx, ky))
        self._trail_history.append((kx, ky, source, float(conf), frame_idx))

        # Boundary stuck detection
        if self._polygon is not None:
            bdist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(cx), float(cy)), True)
            if bdist < 20:
                self._boundary_count += 1
                if self._boundary_count > 30:
                    self._missed_count = 15
                    self.kf.reset()
                    self._boundary_count = 0
                    return None
            else:
                self._boundary_count = 0

        return (float(cx), float(cy), float(conf))

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def _check_fullframe_yolo(self):
        """Get ball detections from the shared YOLODetector (already run per frame)."""
        # The shared YOLO already ran; we can't re-query it cheaply.
        # Instead, the caller (process.py) passes ball_xy from the shared YOLO.
        # This hook is here so BallDetector CAN re-check if needed, but in the
        # normal flow BallDetector.detect() is called AFTER the YOLO detections
        # have been checked in process.py. We keep this for the standalone path.
        return None  # handled externally in the two-pass flow

    def _crop_to_pitch(self, frame):
        """Return (crop_region, (x1, y1, x2, y2)) in original frame coords."""
        poly = self._polygon.astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _predictive_zoom_crop(self, frame, kf_pred):
        """Crop a tight window around the Kalman prediction for zoomed YOLO.

        Instead of running YOLO on the entire pitch (where the ball is ~6px),
        this crops a ~500px window where the ball IS and upscales it. The ball
        becomes ~15-20px — vastly more detectable by YOLO.

        Returns (crop_img, (x1, y1, x2, y2)) or (None, None) when the prediction
        is stale or too close to the frame edge.
        """
        if kf_pred is None:
            return None, None

        px, py = kf_pred

        # Quick sanity — prediction should be at least near the pitch
        if self._polygon is not None:
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(px), float(py)), True)
            if dist < -100:  # well outside pitch → prediction stale
                return None, None

        h, w = frame.shape[:2]
        half = self.zoom_crop_size // 2

        x1 = int(px - half)
        y1 = int(py - half)
        x2 = x1 + self.zoom_crop_size
        y2 = y1 + self.zoom_crop_size

        # Shift the crop window to stay within frame bounds
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Need at least 200px to be useful
        if x2 - x1 < 200 or y2 - y1 < 200:
            return None, None

        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _detect_yolo_on_crop(self, crop, gray, crop_roi=None):
        """Run the small YOLO model on an (optionally upscaled) crop.

        Args:
            crop: BGR crop image.
            gray: Grayscale version of crop.
            crop_roi: (x1, y1, x2, y2) in original frame coords.
                      Defaults to self._crop_roi (set by _crop_to_pitch).
        """
        if crop_roi is None:
            crop_roi = self._crop_roi

        h, w = crop.shape[:2]
        long_side = max(h, w)
        scale = min(self.upscale_target / long_side, self.max_upscale)

        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            inference_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            inference_img = crop
            scale = 1.0

        results = self._crop_yolo(inference_img, conf=0.05, verbose=False,
                                  imgsz=1024)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        ball_idx = np.where(cls_ids == self._crop_ball_id)[0]
        if len(ball_idx) == 0:
            return None

        inv = 1.0 / scale
        motion_mask = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_crop_gray)
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        order = np.argsort(confs[ball_idx])[::-1]
        for idx in ball_idx[order]:
            bx1, by1, bx2, by2 = xyxy[idx]
            cx = (bx1 + bx2) / 2.0 * inv
            cy = (by1 + by2) / 2.0 * inv

            # Inside pitch polygon? (map crop coords → original frame coords)
            rx1, ry1, _, _ = crop_roi
            ox, oy = cx + rx1, cy + ry1
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(ox), float(oy)), True)
            if dist < -10:
                continue

            # Motion check — reject stationary false positives
            if motion_mask is not None:
                mc_x, mc_y = int(cx), int(cy)
                if 0 <= mc_x < motion_mask.shape[1] and 0 <= mc_y < motion_mask.shape[0]:
                    ymn = max(0, mc_y - 4)
                    ymx = min(motion_mask.shape[0], mc_y + 5)
                    xmn = max(0, mc_x - 4)
                    xmx = min(motion_mask.shape[1], mc_x + 5)
                    window = motion_mask[ymn:ymx, xmn:xmx]
                    motion_px = cv2.countNonZero(window)
                    area = window.shape[0] * window.shape[1]
                    if motion_px < 0.05 * area and confs[idx] < 0.4:
                        continue

            if self._is_stationary(ox, oy):
                continue

            return (cx, cy, float(confs[idx]))

        return None

    def _detect_motion_blob(self, crop, gray):
        """Frame-differencing + white blob detection over full crop."""
        motion = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_crop_gray)
            _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=2)

        _, white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        if motion is not None:
            moving_white = cv2.bitwise_and(white, motion)
        else:
            moving_white = white

        return self._find_best_blob(moving_white)

    def _detect_motion_any_color(self, crop, gray):
        """Frame-differencing + any non-green moving blob (ball of any color)."""
        motion = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_crop_gray)
            _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=2)

        if motion is None:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask out grass, white lines, and shadows
        grass_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 180))
        white_mask = cv2.inRange(hsv, (0, 0, 190), (180, 35, 255))
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 35))

        non_grass = cv2.bitwise_not(grass_mask)
        non_white = cv2.bitwise_not(white_mask)
        non_dark = cv2.bitwise_not(dark_mask)
        candidate = cv2.bitwise_and(non_grass, non_white)
        candidate = cv2.bitwise_and(candidate, non_dark)
        moving_candidate = cv2.bitwise_and(candidate, motion)

        moving_candidate = cv2.erode(moving_candidate, np.ones((2, 2), np.uint8), iterations=1)
        moving_candidate = cv2.dilate(moving_candidate, np.ones((3, 3), np.uint8), iterations=1)

        return self._find_best_blob_generous(moving_candidate)

    def _detect_near_prediction(self, crop, gray, roi, kf_pred):
        """Search a window around the Kalman prediction with sensitive thresholds."""
        px, py = kf_pred[0], kf_pred[1]
        if self._polygon is not None:
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(px), float(py)), True)
            if dist < -20:
                return None

        x1, y1, _, _ = roi
        pcx, pcy = px - x1, py - y1
        h, w = crop.shape[:2]
        win = 60
        x_min = max(0, int(pcx) - win)
        x_max = min(w, int(pcx) + win)
        y_min = max(0, int(pcy) - win)
        y_max = min(h, int(pcy) + win)

        if x_max - x_min < 10 or y_max - y_min < 10:
            return None

        window_gray = gray[y_min:y_max, x_min:x_max]
        motion = None
        if self._prev_crop_gray is not None:
            prev_win = self._prev_crop_gray[y_min:y_max, x_min:x_max]
            if prev_win.shape == window_gray.shape:
                diff = cv2.absdiff(window_gray, prev_win)
                _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)

        _, white = cv2.threshold(window_gray, 150, 255, cv2.THRESH_BINARY)
        search_mask = cv2.bitwise_and(white, motion) if motion is not None else white

        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2 or area > 400:
                continue
            center, radius = cv2.minEnclosingCircle(cnt)
            if radius < 1.5 or radius > 15:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            dist_from_pred = np.sqrt((center[0] + x_min - pcx) ** 2 +
                                     (center[1] + y_min - pcy) ** 2)
            proximity = max(0, 1.0 - dist_from_pred / win)
            score = circularity * area * (0.5 + 0.5 * proximity)
            if score > best_score:
                best_score = score
                best = (float(center[0] + x_min), float(center[1] + y_min),
                        min(1.0, score / 30.0) * 0.8)
        return best

    def _find_best_blob(self, binary_mask):
        """Find best white blob by circularity × size."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2 or area > 400:
                continue
            center, radius = cv2.minEnclosingCircle(cnt)
            if radius < self.blob_min_r or radius > self.blob_max_r:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best = (float(center[0]), float(center[1]),
                        min(1.0, score / 50.0))
        return best

    def _find_best_blob_generous(self, binary_mask):
        """Find best blob with wider size range, lower circularity threshold."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2 or area > 500:
                continue
            center, radius = cv2.minEnclosingCircle(cnt)
            if radius < 1.5 or radius > 14:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            size_score = min(area / 30.0, 1.0) if area < 80 else max(0, 1.0 - (area - 80) / 400.0)
            score = circularity * size_score * (1 + radius / 10)
            if score > best_score:
                best_score = score
                best = (float(center[0]), float(center[1]),
                        min(0.7, score / 20.0))
        return best

    @staticmethod
    def _map_to_original(cx_crop, cy_crop, conf, roi):
        x1, y1, _, _ = roi
        return (cx_crop + x1, cy_crop + y1, conf)

    # ------------------------------------------------------------------
    # Stationary detection
    # ------------------------------------------------------------------

    def _is_stationary(self, cx, cy):
        if len(self._recent_positions) < self._stationary_frames:
            return False
        recent = list(self._recent_positions)[-self._stationary_frames:]
        close = sum(1 for x, y in recent
                    if abs(x - cx) < self._stationary_threshold
                    and abs(y - cy) < self._stationary_threshold)
        return close == len(recent)

    # ------------------------------------------------------------------
    # Position filtering — replaced by BallKalmanFilter (above)
    # ------------------------------------------------------------------


# ======================================================================
# Post-processing: trajectory validation
# ======================================================================

def validate_ball_trail(ball_trail, min_fragment_length=4, max_gap_frames=8,
                        max_frame_jump_px=300):
    """Remove false-positive trajectory fragments from the ball trail.

    A real football forms consistent trajectories across many frames.
    False positives (white boots, lines, etc.) appear as isolated
    detections or short, jerky fragments.  This function:

      1. Segments the trail at gaps > *max_gap_frames*
      2. Scores each fragment by length x smoothness x confidence
      3. Discards fragments below threshold

    Args:
        ball_trail:  list of (x, y, frame_idx, confidence)
        min_fragment_length:  discard fragments shorter than this (default 6)
        max_gap_frames:  split trail at gaps larger than this (default 5)
        max_frame_jump_px:  max plausible ball movement between frames

    Returns:
        Cleaned list of (x, y, frame_idx, confidence).
    """
    if not ball_trail:
        return []

    sorted_trail = sorted(ball_trail, key=lambda t: t[2])

    # 1 — Split into fragments at large gaps
    fragments = []
    current = [sorted_trail[0]]
    for i in range(1, len(sorted_trail)):
        if sorted_trail[i][2] - sorted_trail[i - 1][2] > max_gap_frames:
            fragments.append(current)
            current = []
        current.append(sorted_trail[i])
    if current:
        fragments.append(current)

    # 2 — Score each fragment and keep only the plausible ones
    validated = []
    for frag in fragments:
        length = len(frag)
        # Ratio of frame-to-frame jumps that are smooth (< max_frame_jump_px)
        smooth_count = 0
        denom = max(length - 1, 1)
        for i in range(1, length):
            dx = frag[i][0] - frag[i - 1][0]
            dy = frag[i][1] - frag[i - 1][1]
            jump = (dx * dx + dy * dy) ** 0.5
            if jump < max_frame_jump_px:
                smooth_count += 1

        smooth_ratio = smooth_count / denom
        avg_conf = sum(t[3] for t in frag) / length

        length_score = min(length / min_fragment_length, 2.0)
        conf_score = min(avg_conf / 0.3, 1.0)   # 0.3 = acceptable base conf
        composite = length_score * smooth_ratio * conf_score

        # Keep: strong composite score, OR long enough with decent smoothness
        if composite >= 1.0 or (length >= min_fragment_length and smooth_ratio >= 0.6):
            validated.extend(frag)

    return validated
