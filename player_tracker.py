"""Player tracker — ByteTrack with persistence across brief occlusion gaps.

ByteTrack handles tracking within frames, but when a player is occluded
for 2-5 frames and reappears, ByteTrack assigns a new ID. This wrapper
re-identifies reappearing players by proximity to their last known position,
keeping the same track ID across brief gaps.
"""

import numpy as np
import supervision as sv


class PlayerTracker:
    """ByteTrack wrapper that preserves IDs across brief detection gaps."""

    def __init__(self, max_missed=5, proximity_px=80):
        self.byte_track = sv.ByteTrack()
        self._tracks = {}          # stable_id -> {bbox, missed, last_frame, byte_id}
        self._byte_to_stable = {}  # byte_track_id -> stable_id
        self._next_id = 1
        self.max_missed = max_missed
        self.proximity_px = proximity_px
        self.frame_count = 0

    def update(self, detections):
        """Run ByteTrack then re-identify lost players.

        Args:
            detections: sv.Detections with xyxy, confidence, class_id

        Returns:
            sv.Detections with stable tracker_id that persists across gaps.
        """
        self.frame_count += 1

        if detections is None or len(detections) == 0:
            self._age_tracks()
            return detections

        tracked = self.byte_track.update_with_detections(detections)
        if tracked is None:
            self._age_tracks()
            return tracked

        if len(tracked) == 0:
            self._age_tracks()
            return tracked

        # Fallback: if ByteTrack didn't assign IDs, create sequential ones
        if tracked.tracker_id is None:
            tracked.tracker_id = np.arange(1, len(tracked) + 1, dtype=int)

        # ByteTrack's IDs for this frame
        byte_ids = list(int(x) for x in tracked.tracker_id)

        # Build stable IDs: map ByteTrack IDs -> our stable IDs
        stable_ids = self._assign_stable_ids(tracked, byte_ids)

        # Update our track store
        current_stable = set(stable_ids)
        for sid in list(self._tracks.keys()):
            if sid in current_stable:
                self._tracks[sid]["missed"] = 0
                self._tracks[sid]["last_frame"] = self.frame_count
            else:
                self._tracks[sid]["missed"] += 1
                if self._tracks[sid]["missed"] > self.max_missed:
                    del self._tracks[sid]

        # Store current frame bbox for each track
        for i, sid in enumerate(stable_ids):
            self._tracks[sid] = {
                "bbox": tracked.xyxy[i].copy(),
                "missed": 0,
                "last_frame": self.frame_count,
                "byte_id": byte_ids[i],
            }

        # Override tracker_id with our stable IDs
        tracked.tracker_id = np.array(stable_ids, dtype=int)
        return tracked

    def _assign_stable_ids(self, tracked, byte_ids):
        """Map each ByteTrack ID to a stable ID (re-identifying lost tracks)."""
        stable_ids = []
        for i, bid in enumerate(byte_ids):
            if bid in self._byte_to_stable:
                stable_ids.append(self._byte_to_stable[bid])
                continue

            # ByteTrack assigned a new ID — check if it matches a lost track
            matched = self._match_lost(tracked.xyxy[i])
            if matched is not None:
                self._byte_to_stable[bid] = matched
                stable_ids.append(matched)
            else:
                new_id = self._next_id
                self._next_id += 1
                self._byte_to_stable[bid] = new_id
                stable_ids.append(new_id)

        return stable_ids

    def _match_lost(self, bbox):
        """Find a recently-lost track whose last position is near this bbox."""
        cx = (float(bbox[0]) + float(bbox[2])) / 2
        cy = (float(bbox[1]) + float(bbox[3])) / 2

        best_sid = None
        best_dist = float(self.proximity_px)

        for sid, info in self._tracks.items():
            if info["missed"] == 0:
                continue  # already active this frame
            if info["missed"] > self.max_missed:
                continue  # too stale

            lb = info["bbox"]
            lcx = (float(lb[0]) + float(lb[2])) / 2
            lcy = (float(lb[1]) + float(lb[3])) / 2

            dist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_sid = sid

        return best_sid

    def _age_tracks(self):
        """Increment miss counter for all tracks when no detections at all."""
        for sid in list(self._tracks.keys()):
            self._tracks[sid]["missed"] += 1
            if self._tracks[sid]["missed"] > self.max_missed:
                del self._tracks[sid]

    @property
    def active_tracks(self):
        return {sid for sid, info in self._tracks.items() if info["missed"] == 0}

    def reset(self):
        self._tracks.clear()
        self._byte_to_stable.clear()
        self._next_id = 1
        self.frame_count = 0
