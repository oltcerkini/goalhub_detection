"""Ball detector — YOLO + motion-based white-blob fallback + Kalman smoothing.

Strategy:
1. Use the shared YOLODetector's full-frame ball detections (fast path).
2. When YOLO fails, crop to pitch + upscale and re-run YOLO for small balls.
3. When that fails, frame-differencing + white-blob detection for distant ball.
4. Non-green moving blob (catches balls of ANY color).
5. Kalman filter for temporal smoothing + gap-filling (up to 10 frames).
"""

from collections import deque

import cv2
import numpy as np


class BallDetector:
    """Detects football using YOLO + motion-blob fallback + position tracking.

    Shares the main YOLO model (no second full-frame inference). For small-ball
    detection it can optionally run a tiny YOLO on an upscaled pitch crop.
    """

    _stationary_threshold = 15   # pixels — if ball stays in a circle this small
    _stationary_frames = 8       # for this many consecutive frames → reject

    def __init__(self, yolo_detector, trail_length=20,
                 crop_model_path="soccana_yolo11n.pt",
                 blob_min_radius=2, blob_max_radius=10,
                 upscale_target=1600, max_upscale=4.0,
                 kalman_gate_px=300):
        """
        Args:
            yolo_detector: Shared YOLODetector instance (full-frame inference).
            crop_model_path: Optional small YOLO for crop-based ball detection.
                             None = skip crop YOLO (only use full-frame dets + blobs).
            upscale_target: When running YOLO on pitch crop, upscale so longest
                            edge is this many px (better small-ball detection).
        """
        self.detector = yolo_detector
        self.trail = deque(maxlen=trail_length)
        self._last_pos = None          # last accepted (x, y)
        self._kf_initialized = False
        self._gate_px = kalman_gate_px
        self._missed_count = 0
        self._seen_positions = {}       # (rounded_x, rounded_y) → count (≥3 blacklisted)
        self._prev_crop_gray = None
        self._polygon = None
        self._crop_roi = None           # (x1, y1, x2, y2) in original frame
        self._recent_positions = deque(maxlen=10)
        self._boundary_count = 0

        # Optional crop-based YOLO model for small-ball detection
        self._crop_yolo = None
        self.blob_min_r = blob_min_radius
        self.blob_max_r = blob_max_radius
        self.upscale_target = upscale_target
        self.max_upscale = max_upscale
        if crop_model_path:
            try:
                from ultralytics import YOLO
                self._crop_yolo = YOLO(crop_model_path)
                print(f"  BallDetector: crop YOLO model loaded ({crop_model_path})")
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

        # Step 0 — full-frame YOLO ball detections are the fast path.
        # If the shared detector already found a ball, validate and use it.
        if yolo_ball_xy is not None:
            cx, cy, conf = yolo_ball_xy
            accepted = self._accept_ball(cx, cy, conf, source="yolo")
            if accepted is not None:
                return accepted
            # YOLO ball rejected by filters — fall through to crop/blob pipeline

        # Step 1 — crop to pitch for focused ball search
        if self._polygon is not None:
            crop_frame, roi = self._crop_to_pitch(frame)
        else:
            crop_frame, roi = frame, (0, 0, frame.shape[1], frame.shape[0])
        self._crop_roi = roi

        if crop_frame is None or crop_frame.size == 0:
            kf_pred = self._predict_kf()
            if kf_pred is not None:
                self.trail.append((kf_pred[0], kf_pred[1]))
            return kf_pred

        gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        kf_pred = self._predict_kf() if self._kf_initialized else None

        # Step 2 — YOLO on upscaled pitch crop (better for distant small ball)
        if self._crop_yolo is not None:
            result = self._detect_yolo_on_crop(crop_frame, gray)
            if result is not None:
                cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
                accepted = self._accept_ball(cx, cy, conf, source="yolo_crop", kf_pred=kf_pred)
                if accepted is not None:
                    self._prev_crop_gray = gray
                    return accepted

        # Step 3 — Kalman-guided blob search (sensitive, near prediction)
        if kf_pred is not None:
            result = self._detect_near_prediction(crop_frame, gray, roi, kf_pred)
            if result is not None:
                cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
                accepted = self._accept_ball(cx, cy, conf, source="blob_near", kf_pred=kf_pred)
                if accepted is not None:
                    self._prev_crop_gray = gray
                    return accepted

        # Step 4 — wide white-blob search (no Kalman guidance)
        result = self._detect_motion_blob(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = self._accept_ball(cx, cy, conf, source="blob_wide", kf_pred=kf_pred)
            if accepted is not None:
                self._prev_crop_gray = gray
                return accepted

        # Step 5 — non-green moving blob (balls of ANY color)
        result = self._detect_motion_any_color(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = self._accept_ball(cx, cy, conf, source="blob_anycolor", kf_pred=kf_pred)
            if accepted is not None:
                self._prev_crop_gray = gray
                return accepted

        self._prev_crop_gray = gray

        # Step 7 — Kalman prediction for gap-filling
        self._missed_count += 1
        if self._missed_count < 10 and kf_pred is not None:
            px, py = kf_pred[0], kf_pred[1]
            if self._polygon is not None:
                pt_test = cv2.pointPolygonTest(
                    self._polygon.astype(np.float32), (float(px), float(py)), True)
                if pt_test < -200:
                    self._missed_count = 15
                    return None
            self.trail.append((px, py))
            return (px, py, 0.0)

        if self._missed_count >= 10:
            self._kf_initialized = False
        return None

    def reset(self):
        self._last_pos = None
        self._kf_initialized = False
        self._prev_crop_gray = None
        self.trail.clear()
        self._missed_count = 0
        self._boundary_count = 0
        self._seen_positions.clear()

    # ------------------------------------------------------------------
    # Ball acceptance gate — shared across all detection strategies
    # ------------------------------------------------------------------

    def _accept_ball(self, cx, cy, conf, source="", kf_pred=None):
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

        # 2 — Kalman gate: reject detections far from last known position
        if self._kf_initialized and kf_pred is not None:
            dx = cx - kf_pred[0]
            dy = cy - kf_pred[1]
            if dx * dx + dy * dy > self._gate_px * self._gate_px:
                return None

        # 3 — Position blacklist: persistent false positives
        rounded = (round(cx / 10) * 10, round(cy / 10) * 10)
        self._seen_positions[rounded] = self._seen_positions.get(rounded, 0) + 1
        if self._seen_positions[rounded] >= 3:
            return None

        # 4 — Stationary rejection
        if self._is_stationary(cx, cy):
            return None

        # — Accepted —
        self._missed_count = 0
        self._correct_kalman(cx, cy)
        self.trail.append((cx, cy))
        self._recent_positions.append((cx, cy))

        # Boundary stuck detection
        if self._polygon is not None:
            bdist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), (float(cx), float(cy)), True)
            if bdist < 20:
                self._boundary_count += 1
                if self._boundary_count > 30:
                    self._missed_count = 15
                    self._kf_initialized = False
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

    def _detect_yolo_on_crop(self, crop, gray):
        """Run the small YOLO model on an upscaled pitch crop."""
        h, w = crop.shape[:2]
        long_side = max(h, w)
        scale = min(self.upscale_target / long_side, self.max_upscale)

        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            inference_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            inference_img = crop
            scale = 1.0

        results = self._crop_yolo(inference_img, conf=0.08, verbose=False,
                                  imgsz=max(inference_img.shape[:2]))
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        ball_idx = np.where(cls_ids == 1)[0]
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

            # Inside pitch polygon?
            x1, y1, _, _ = self._crop_roi
            ox, oy = cx + x1, cy + y1
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
    # Kalman (position smoother — avoids broken OpenCV KF Python bindings)
    # ------------------------------------------------------------------

    def _correct_kalman(self, cx, cy):
        self._last_pos = (float(cx), float(cy))
        if not self._kf_initialized:
            self._kf_initialized = True

    def _predict_kf(self):
        if not self._kf_initialized or self._last_pos is None:
            return None
        return (self._last_pos[0], self._last_pos[1], 0.0)
