"""Standalone ball detector — YOLO + motion-based white-blob fallback.

Does NOT touch player detection.

Strategy:
1. Crop to the pitch polygon (remove background noise).
2. Run YOLO (soccana model) on the crop — catches clear ball views.
3. When YOLO fails, use frame-differencing + white-blob detection to
   find small moving white objects (the ball at distance).
4. Kalman filter for temporal smoothing + gap-filling.
"""

from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kalman filter helpers
# ---------------------------------------------------------------------------

def _make_kalman():
    """8-state Kalman: x, y, vx, vy, ax, ay, width, height."""
    kf = cv2.KalmanFilter(8, 2)
    kf.measurementMatrix = np.eye(2, 8, dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0, 0.5, 0, 0, 0],
        [0, 1, 0, 1, 0, 0.5, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    return kf


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class BallDetector:
    """Detects football using YOLO + motion-blob fallback + Kalman filter."""

    _stationary_threshold = 20  # pixels (> many frames of goal-line drift)
    _stationary_frames = 5  # consecutive same-region → reject

    def __init__(self, model_path="soccana_yolo11n.pt",
                 yolo_conf=0.12, trail_length=15,
                 blob_min_radius=2, blob_max_radius=8,
                 upscale_target=1280, max_upscale=3.0):
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.yolo_conf = yolo_conf
        self.blob_min_r = blob_min_radius
        self.blob_max_r = blob_max_radius
        self.upscale_target = upscale_target
        self.max_upscale = max_upscale
        self.trail = deque(maxlen=trail_length)
        self._kalman = _make_kalman()
        self._kf_initialized = False
        self._prev_crop_gray = None
        self._polygon = None
        self._crop_roi = None  # (x1, y1, x2, y2) in original frame
        self._recent_positions = deque(maxlen=10)  # for stationary detection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame, polygon=None):
        """Detect ball. Returns (cx, cy, confidence) or None."""
        if polygon is not None:
            self._polygon = polygon

        # Step 1 — crop to pitch
        if self._polygon is not None:
            crop_frame, roi = self._crop_to_pitch(frame)
        else:
            crop_frame, roi = frame, (0, 0, frame.shape[1], frame.shape[0])
        self._crop_roi = roi

        if crop_frame is None or crop_frame.size == 0:
            kf_pred = self._predict_kf()
            if kf_pred:
                self.trail.append((kf_pred[0], kf_pred[1]))
            return kf_pred

        # Compute gray once for motion analysis across both methods
        gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

        # Get Kalman prediction ONCE for this frame (advance state)
        kf_pred = self._predict_kf() if self._kf_initialized else None

        # Helper: accept a detection if it passes all filters
        def _accept(cx, cy, conf):
            # Reject stationary detections — ball must move in a match
            if self._is_stationary(cx, cy):
                return None
            if self._polygon is not None:
                dist = cv2.pointPolygonTest(
                    self._polygon.astype(np.float32), (float(cx), float(cy)), True)
                if dist < -10:
                    return None
            self._correct_kalman(cx, cy)
            self.trail.append((cx, cy))
            self._recent_positions.append((cx, cy))
            return (float(cx), float(cy), float(conf))

        # Step 2 — try YOLO on the crop
        result = self._detect_yolo(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = _accept(cx, cy, conf)
            if accepted:
                self._prev_crop_gray = gray
                return accepted

        # Step 3 — Kalman-guided blob search (sensitive search near prediction)
        if kf_pred is not None:
            result = self._detect_near_prediction(crop_frame, gray, roi, kf_pred)
            if result is not None:
                cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
                accepted = _accept(cx, cy, conf)
                if accepted:
                    self._prev_crop_gray = gray
                    return accepted

        # Step 4 — wide blob search (fallback, no Kalman guidance)
        result = self._detect_motion_blob(crop_frame, gray)
        if result is not None:
            cx, cy, conf = self._map_to_original(result[0], result[1], result[2], roi)
            accepted = _accept(cx, cy, conf)
            if accepted:
                self._prev_crop_gray = gray
                return accepted

        self._prev_crop_gray = gray
        # Step 5 — use the prediction already made (no second predict)
        if kf_pred is not None:
            self.trail.append((kf_pred[0], kf_pred[1]))
            return kf_pred
        return None

    def reset(self):
        self._kalman = _make_kalman()
        self._kf_initialized = False
        self._prev_crop_gray = None
        self.trail.clear()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _crop_to_pitch(self, frame):
        """Return (crop, (x1, y1, x2, y2) in original coords)."""
        poly = self._polygon.astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _detect_yolo(self, crop, gray):
        """YOLO on crop (upscaled when small, full-res otherwise).
        Returns (cx, cy, confidence) in crop coords or None."""
        h, w = crop.shape[:2]

        # If crop is smaller than target, upscale it (gives YOLO more pixels on the ball)
        long_side = max(h, w)
        scale = min(self.upscale_target / long_side, self.max_upscale)
        if scale > 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            inference_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            did_resize = True
        else:
            inference_img = crop
            scale = 1.0
            did_resize = False

        results = self.yolo(inference_img, conf=self.yolo_conf, verbose=False,
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

        inv = 1.0 / scale if did_resize else 1.0

        # Motion mask to suppress stationary false positives
        motion_mask = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_crop_gray)
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Sort by confidence descending, return first one inside the pitch
        order = np.argsort(confs[ball_idx])[::-1]
        for idx in ball_idx[order]:
            bx1, by1, bx2, by2 = xyxy[idx]
            cx = (bx1 + bx2) / 2.0 * inv
            cy = (by1 + by2) / 2.0 * inv

            # Check if detection is inside the pitch polygon
            x1, y1, _, _ = self._crop_roi
            ox, oy = cx + x1, cy + y1
            pt = (float(ox), float(oy))
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), pt, True)
            if dist < -10:
                continue

            # Reject stationary false positives (line markings, etc.)
            if motion_mask is not None:
                mc_x, mc_y = int(cx), int(cy)
                if 0 <= mc_x < motion_mask.shape[1] and 0 <= mc_y < motion_mask.shape[0]:
                    y_min = max(0, mc_y - 4)
                    y_max = min(motion_mask.shape[0], mc_y + 5)
                    x_min = max(0, mc_x - 4)
                    x_max = min(motion_mask.shape[1], mc_x + 5)
                    window = motion_mask[y_min:y_max, x_min:x_max]
                    motion_pixels = cv2.countNonZero(window)
                    window_area = window.shape[0] * window.shape[1]
                    if motion_pixels < 0.1 * window_area and confs[idx] < 0.5:
                        continue

            # Reject detections stuck at the same position (goal line, etc.)
            # Convert to original coords for comparison with _accept's entries
            ox, oy = cx + self._crop_roi[0], cy + self._crop_roi[1]
            if self._is_stationary(ox, oy):
                continue

            return (cx, cy, float(confs[idx]))

        return None

    def _detect_motion_blob(self, crop, gray):
        """Frame-differencing + white blob detection over full crop."""
        # Motion mask: |current - previous|
        motion = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_crop_gray)
            _, motion = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            # Dilate to connect nearby regions
            motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)

        # White regions in pitch area (more generous: 180 instead of 200)
        _, white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine: moving white pixels
        if motion is not None:
            moving_white = cv2.bitwise_and(white, motion)
        else:
            moving_white = white

        return self._find_best_blob(moving_white)

    def _detect_near_prediction(self, crop, gray, roi, kf_pred):
        """Search a window around the Kalman prediction for the ball.
        Uses more sensitive thresholds (lower white threshold, accepts
        smaller/larger blobs) since we're searching near where the ball should be."""
        px, py = kf_pred[0], kf_pred[1]

        # Clamp prediction to within the pitch polygon
        if self._polygon is not None:
            pt = (float(px), float(py))
            dist = cv2.pointPolygonTest(
                self._polygon.astype(np.float32), pt, True)
            if dist < -20:
                return None  # prediction way outside pitch, don't bother

        # Convert to crop coords
        x1, y1, _, _ = roi
        pcx, pcy = px - x1, py - y1

        # Motion + white in a search window around the prediction
        h, w = crop.shape[:2]
        window_size = 60  # search 60px around predicted position
        x_min = max(0, int(pcx) - window_size)
        x_max = min(w, int(pcx) + window_size)
        y_min = max(0, int(pcy) - window_size)
        y_max = min(h, int(pcy) + window_size)

        if x_max - x_min < 10 or y_max - y_min < 10:
            return None

        window_gray = gray[y_min:y_max, x_min:x_max]

        # Motion in window
        motion = None
        if self._prev_crop_gray is not None:
            prev_window = self._prev_crop_gray[y_min:y_max, x_min:x_max]
            if prev_window.shape == window_gray.shape:
                diff = cv2.absdiff(window_gray, prev_window)
                _, motion = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # lower motion threshold
                motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)

        # More generous white threshold in search window
        _, white = cv2.threshold(window_gray, 160, 255, cv2.THRESH_BINARY)

        if motion is not None:
            search_mask = cv2.bitwise_and(white, motion)
        else:
            search_mask = white

        # Find blobs, but only in the window
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2 or area > 400:  # wider range
                continue

            center, radius = cv2.minEnclosingCircle(cnt)
            cx_c, cy_c = center
            if radius < 1.5 or radius > 15:
                continue

            # Weaker circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Score: proximity to prediction + circularity + size
            dist_from_pred = np.sqrt((cx_c + x_min - pcx) ** 2 + (cy_c + y_min - pcy) ** 2)
            proximity = max(0, 1.0 - dist_from_pred / window_size)
            score = circularity * area * (0.5 + 0.5 * proximity)

            if score > best_score:
                best_score = score
                # Map back to crop coords
                abs_cx = cx_c + x_min
                abs_cy = cy_c + y_min
                conf = min(1.0, score / 30.0) * 0.8  # reduced confidence for guided mode
                best = (float(abs_cx), float(abs_cy), conf)

        return best

    def _find_best_blob(self, binary_mask):
        """Find the best white blob in a binary mask."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3 or area > 250:
                continue

            center, radius = cv2.minEnclosingCircle(cnt)
            cx_c, cy_c = center
            if radius < self.blob_min_r or radius > self.blob_max_r:
                continue

            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:
                continue

            # Score: bigger + more circular = better
            score = circularity * area
            if score > best_score:
                best_score = score
                best = (float(cx_c), float(cy_c),
                        min(1.0, score / 50.0))

        return best

    def _map_to_original(self, cx_crop, cy_crop, conf, roi):
        """Map crop coords to original frame coords."""
        x1, y1, _, _ = roi
        return (cx_crop + x1, cy_crop + y1, conf)

    # ------------------------------------------------------------------
    # Stationary detection filter
    # ------------------------------------------------------------------

    def _is_stationary(self, cx, cy):
        """Check if (cx, cy) is suspiciously similar to recent detections.
        The ball should move; stationary detections are likely false positives
        (goal line, corner flags, etc.)."""
        if len(self._recent_positions) < self._stationary_frames:
            return False
        recent = list(self._recent_positions)[-self._stationary_frames:]
        close_count = sum(
            1 for x, y in recent
            if abs(x - cx) < self._stationary_threshold
            and abs(y - cy) < self._stationary_threshold
        )
        return close_count == len(recent)  # all recent positions are close

    # ------------------------------------------------------------------
    # Kalman
    # ------------------------------------------------------------------

    def _correct_kalman(self, cx, cy):
        """Update Kalman with a detection. Does NOT call predict()."""
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        if not self._kf_initialized:
            self._kalman.statePost[:2] = measurement
            self._kf_initialized = True
        else:
            self._kalman.correct(measurement)

    def _predict_kf(self):
        """Advance Kalman by one step and return predicted position."""
        if not self._kf_initialized:
            return None
        pred = self._kalman.predict()
        cx, cy = float(pred[0][0]), float(pred[1][0])
        return (cx, cy, 0.0)
