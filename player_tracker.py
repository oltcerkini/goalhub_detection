"""Player tracker — ByteTrack with appearance-based re-identification.

Runs ByteTrack for within-frame tracking, then a custom re-ID layer that
matches reappearing players by BOTH position proximity AND jersey colour
similarity (HSV histogram). This prevents ID explosion (71 "players") when
a player is briefly occluded or the detector misses them for a few frames.
"""

import cv2
import numpy as np
import supervision as sv


class PlayerTracker:
    """ByteTrack + appearance ReID wrapper that preserves IDs across gaps."""

    def __init__(self, max_missed=60, proximity_px=200,
                 appearance_threshold=0.45, match_distance_weight=0.4):
        self.byte_track = sv.ByteTrack()
        self._tracks = {}
        self._byte_to_stable = {}
        self._next_id = 1
        self.max_missed = max_missed
        self.proximity_px = proximity_px
        self.appearance_threshold = appearance_threshold
        self.match_distance_weight = match_distance_weight
        self.frame_count = 0
        self._seen_appearances = {}  # track_id -> accumulated appearance

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, detections, frame=None):
        """Run tracking + re-identification.

        Args:
            detections: sv.Detections (xyxy, confidence, class_id)
            frame: optional BGR frame — needed for appearance extraction.

        Returns:
            sv.Detections with stable tracker_id.
        """
        self.frame_count += 1

        if detections is None or len(detections) == 0:
            self._age_tracks()
            return detections

        tracked = self.byte_track.update_with_detections(detections)
        if tracked is None or len(tracked) == 0:
            self._age_tracks()
            return tracked

        # Fallback IDs if ByteTrack didn't assign
        if tracked.tracker_id is None:
            tracked.tracker_id = np.arange(1, len(tracked) + 1, dtype=int)

        byte_ids = list(int(x) for x in tracked.tracker_id)

        # Extract appearance features from frame
        appearances = []
        if frame is not None:
            for i in range(len(tracked)):
                feat = self._extract_appearance(frame, tracked.xyxy[i])
                appearances.append(feat)
        else:
            appearances = [None] * len(tracked)

        stable_ids = self._assign_stable_ids(tracked, byte_ids, appearances)

        # Update active track store
        current_stable = set(stable_ids)
        for sid in list(self._tracks.keys()):
            if sid in current_stable:
                self._tracks[sid]["missed"] = 0
                self._tracks[sid]["last_frame"] = self.frame_count
            else:
                self._tracks[sid]["missed"] += 1
                if self._tracks[sid]["missed"] > self.max_missed:
                    del self._tracks[sid]

        for i, sid in enumerate(stable_ids):
            self._tracks[sid] = {
                "bbox": tracked.xyxy[i].copy(),
                "missed": 0,
                "last_frame": self.frame_count,
                "byte_id": byte_ids[i],
                "appearance": appearances[i],
            }
            # Accumulate appearance for long-term re-id
            if appearances[i] is not None:
                if sid not in self._seen_appearances:
                    self._seen_appearances[sid] = []
                self._seen_appearances[sid].append(appearances[i])
                # Keep last N for memory
                if len(self._seen_appearances[sid]) > 10:
                    self._seen_appearances[sid].pop(0)

        tracked.tracker_id = np.array(stable_ids, dtype=int)
        return tracked

    # ── Appearance feature extraction ────────────────────────────────────────

    @staticmethod
    def _extract_appearance(frame, bbox):
        """Extract HSV histogram from upper-body (jersey) region.

        Simple and robust: crops the upper body, converts to HSV,
        computes a 2D H×S histogram directly. No masking, no reshape
        gymnastics — the jersey colour dominates the histogram even
        with some background mixed in.
        """
        x1, y1, x2, y2 = map(int, bbox)
        if y1 >= y2 or x1 >= x2:
            return None
        h_f, w_f = frame.shape[:2]
        x1 = max(0, x1 - 3)
        y1 = max(0, y1 - 3)
        x2 = min(w_f - 1, x2 + 3)
        y2 = min(h_f - 1, y2 + 3)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 10 or w < 6:
            return None

        # Upper body: 10%–55% of bbox height (jersey area)
        top = int(h * 0.12)
        bot = int(h * 0.52)
        left = int(w * 0.12)
        right = int(w * 0.88)
        if top >= bot or left >= right:
            return None

        region = crop[top:bot, left:right]
        if region.size == 0:
            return None

        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            # Compute 2D histogram directly on the (H, W, 3) image
            hist = cv2.calcHist([hsv], [0, 1], None,
                                [28, 30], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _appearance_similarity(hist_a, hist_b):
        """Correlation-based similarity: 1 = identical, -1 = opposite, 0 = unrelated."""
        if hist_a is None or hist_b is None:
            return 0.0
        return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    # ── Stable ID assignment ─────────────────────────────────────────────────

    def _assign_stable_ids(self, tracked, byte_ids, appearances):
        stable_ids = []
        for i, bid in enumerate(byte_ids):
            if bid in self._byte_to_stable:
                stable_ids.append(self._byte_to_stable[bid])
                continue

            # New ByteTrack ID — try re-identifying a lost track
            matched = self._match_lost(tracked.xyxy[i], appearances[i])
            if matched is not None:
                self._byte_to_stable[bid] = matched
                stable_ids.append(matched)
            else:
                new_id = self._next_id
                self._next_id += 1
                self._byte_to_stable[bid] = new_id
                stable_ids.append(new_id)

        return stable_ids

    @staticmethod
    def _iou(bbox_a, bbox_b):
        """Intersection-over-union between two [x1,y1,x2,y2] boxes."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _match_lost(self, bbox, appearance):
        """Find the best recently-lost track matching this detection.

        Uses IOU (primary during occlusion) + appearance similarity + position proximity.
        Returns stable track ID or None.
        """
        cx = (float(bbox[0]) + float(bbox[2])) / 2
        cy = (float(bbox[1]) + float(bbox[3])) / 2

        best_sid = None
        best_score = -float("inf")

        for sid, info in self._tracks.items():
            if info["missed"] == 0:
                continue
            if info["missed"] > self.max_missed:
                continue

            lb = info["bbox"]
            lcx = (float(lb[0]) + float(lb[2])) / 2
            lcy = (float(lb[1]) + float(lb[3])) / 2
            dist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)

            if dist > self.proximity_px * 2:
                continue  # too far even with appearance

            # IOU between new detection and track's last-known bbox
            iou = self._iou(bbox, lb)

            # Normalised position score (0..1, higher = better)
            pos_score = max(0.0, 1.0 - dist / self.proximity_px)

            # Appearance score
            if appearance is not None:
                app_scores = []
                if info["appearance"] is not None:
                    app_scores.append(
                        self._appearance_similarity(appearance, info["appearance"])
                    )
                if sid in self._seen_appearances:
                    for hist in self._seen_appearances[sid][-3:]:
                        app_scores.append(
                            self._appearance_similarity(appearance, hist)
                        )
                app_score = max(app_scores) if app_scores else 0.0
            else:
                app_score = 0.0

            # Dynamic weight: when IOU is high, trust geometry over appearance.
            # During occlusion (IOU ~0 but very close), boost position weight.
            w = self.match_distance_weight
            if iou > 0.3:
                w = 0.8  # strong overlap → mostly geometric
                combined = w * iou + (1 - w) * app_score
            elif dist < self.proximity_px * 0.25:
                w = 0.7  # very close → prefer position
                combined = w * pos_score + (1 - w) * app_score
            else:
                combined = w * pos_score + (1 - w) * app_score

            # Require minimum appearance match if appearance is available
            if appearance is not None and app_score < self.appearance_threshold:
                if dist > self.proximity_px and iou < 0.1:
                    continue  # need both proximity AND appearance (unless high IOU)

            if combined > best_score and combined > 0.1:
                best_score = combined
                best_sid = sid

        return best_sid

    # ── Track lifecycle ──────────────────────────────────────────────────────

    def _age_tracks(self):
        for sid in list(self._tracks.keys()):
            self._tracks[sid]["missed"] += 1
            if self._tracks[sid]["missed"] > self.max_missed:
                del self._tracks[sid]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def active_tracks(self):
        return {sid for sid, info in self._tracks.items() if info["missed"] == 0}

    @property
    def track_count(self):
        """Return number of unique stable track IDs ever created."""
        return self._next_id - 1

    def reset(self):
        self._tracks.clear()
        self._byte_to_stable.clear()
        self._seen_appearances.clear()
        self._next_id = 1
        self.frame_count = 0
