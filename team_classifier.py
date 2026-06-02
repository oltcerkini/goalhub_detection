"""Colour-based team classification using K-means on HSV jersey regions.

Strategy:
  1. For each player crop, extract the upper-body region (jersey area)
  2. Remove grass pixels (green), white pixels (lines), dark pixels (shadows)
  3. Cluster remaining pixels per-player to find dominant jersey colour
  4. Globally cluster all player jersey colours into 2 teams
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamClassifier:
    """Clusters detected players into two teams by dominant jersey colour."""

    # HSV ranges for background removal
    _GRASS_H_RANGE = (35, 85)
    _WHITE_V_MIN = 200
    _WHITE_S_MAX = 30
    _DARK_V_MAX = 30

    def __init__(self):
        self.labels = {}         # detection-index -> 0|1
        self.cluster_centers = None  # (2, 3) HSV
        self.my_team = None
        self._team_bgr = [(0, 255, 0), (255, 0, 100)]  # green, hot pink
        # Per-track team cache for consistent cross-frame classification
        self._track_team = {}     # track_id -> team (0 or 1, once confident)
        self._track_votes = {}    # track_id -> {0: count, 1: count}

    def classify(self, image, detections):
        """Assign team labels (0/1) to each detection.

        Detects ALL players first, then divides by colour.

        Args:
            image: BGR numpy array
            detections: sv.Detections (pre-filtered to inside-pitch)

        Returns:
            dict mapping detection-index -> team (0 or 1), or None if < 4 players
        """
        self.labels = {}
        features = []
        valid = []

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            if y1 >= y2 or x1 >= x2:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            color_feat = self._extract_jersey_colour(crop)
            if color_feat is not None:
                features.append(color_feat)
                valid.append(i)

        if len(valid) < 4:
            print(f"  Too few players for team classification ({len(valid)}).")
            return None

        features = np.array(features)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        preds = kmeans.fit_predict(features)

        for idx, label in zip(valid, preds):
            self.labels[idx] = int(label)

        self.cluster_centers = kmeans.cluster_centers_
        self._assign_team_colours()
        n_team0 = sum(1 for l in self.labels.values() if l == 0)
        n_team1 = sum(1 for l in self.labels.values() if l == 1)
        print(f"  Classified {len(self.labels)} players: Team1={n_team0}, Team2={n_team1}")
        return self.labels

    def set_my_team(self, team_idx):
        """Set which team (0 or 1) is 'my team'."""
        self.my_team = team_idx
        if team_idx == 0:
            self._team_bgr = [(0, 255, 0), (255, 0, 100)]
        else:
            self._team_bgr = [(255, 0, 100), (0, 255, 0)]
        print(f"  My team -> Team {team_idx + 1}")

    def classify_frame(self, image, detections):
        """Classify using cached cluster centres (fast, no retraining).

        Must call classify() at least once first to set cluster_centers.
        Falls back to classify() if no cached model exists.
        """
        if self.cluster_centers is None:
            return self.classify(image, detections)

        self.labels = {}
        c0, c1 = self.cluster_centers

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            if y1 >= y2 or x1 >= x2:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            feat = self._extract_jersey_colour(crop)
            if feat is None:
                continue
            dist0 = np.linalg.norm(feat - c0)
            dist1 = np.linalg.norm(feat - c1)
            self.labels[i] = 0 if dist0 < dist1 else 1

        return self.labels

    def get_colour(self, detection_idx):
        """Get the display colour for a detection based on its team."""
        team = self.labels.get(detection_idx, 0)
        return self._team_bgr[team]

    def get_team_name(self, detection_idx):
        """Return 'My Team', 'Opponent', or 'Unknown'."""
        team = self.labels.get(detection_idx)
        if team is None:
            return "Unknown"
        if self.my_team is not None and team == self.my_team:
            return "My Team"
        return f"Team {team + 1}"

    # ------------------------------------------------------------------
    # Per-track team cache (consistent across frames)
    # ------------------------------------------------------------------

    def update_track_vote(self, track_id, detection_idx):
        """Record a vote for this track_id's team based on current frame label."""
        team = self.labels.get(detection_idx)
        if team is None:
            return
        if track_id not in self._track_votes:
            self._track_votes[track_id] = {0: 0, 1: 0}
        self._track_votes[track_id][team] += 1

        # Once we have 3+ votes and 75%+ majority, lock in the team
        votes = self._track_votes[track_id]
        total = votes[0] + votes[1]
        if total >= 3 and track_id not in self._track_team:
            if votes[0] / total >= 0.75:
                self._track_team[track_id] = 0
            elif votes[1] / total >= 0.75:
                self._track_team[track_id] = 1

    def get_track_team_name(self, track_id):
        """Get team name for a track_id from cache, or the per-frame label."""
        # First check if we have a locked-in team for this track
        team = self._track_team.get(track_id)
        if team is not None:
            return self._format_team_name(team)

        # Fall back to majority vote if available
        votes = self._track_votes.get(track_id)
        if votes and (votes[0] + votes[1]) >= 2:
            team = 0 if votes[0] >= votes[1] else 1
            return self._format_team_name(team)

        return "Unknown"

    def _format_team_name(self, team):
        """Format a team number into display name."""
        if self.my_team is not None and team == self.my_team:
            return "My Team"
        return f"Team {team + 1}"

    def reset_track_cache(self):
        """Clear per-track team cache (e.g., for new video)."""
        self._track_team.clear()
        self._track_votes.clear()

    # ------------------------------------------------------------------
    # Improved colour extraction with background pixel removal
    # ------------------------------------------------------------------

    def _extract_jersey_colour(self, player_crop):
        """Extract dominant HSV from the upper-body region with background removal."""
        h, w = player_crop.shape[:2]
        if h < 15 or w < 10:
            return None

        # Upper body: ~10% to ~50% of bbox height (avoids head/grass/shorts)
        top = int(h * 0.10)
        bot = int(h * 0.50)
        left = int(w * 0.15)
        right = int(w * 0.85)
        if top >= bot or left >= right:
            return None

        region = player_crop[top:bot, left:right]
        if region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Remove background pixels: grass, white lines, shadows
        filtered = []
        for h_val, s_val, v_val in pixels:
            if self._GRASS_H_RANGE[0] <= h_val <= self._GRASS_H_RANGE[1] and s_val > 20 and v_val > 20:
                continue  # grass
            if v_val > self._WHITE_V_MIN and s_val < self._WHITE_S_MAX:
                continue  # white line
            if v_val < self._DARK_V_MAX:
                continue  # shadow
            filtered.append([h_val, s_val, v_val])

        if len(filtered) < 20:
            return None

        filtered = np.array(filtered, dtype=np.float32)

        # Cluster remaining pixels, pick the most colorful cluster as jersey
        k = min(3, len(filtered))
        km = KMeans(n_clusters=k, random_state=0, n_init=3, max_iter=10)
        km.fit(filtered)
        counts = np.bincount(km.labels_)

        best = -1
        best_score = -1.0
        for i in range(k):
            sat = km.cluster_centers_[i][1]
            weight = counts[i]
            # Strongly prefer saturated colors (jersey) over dull ones
            score = sat * weight * (1.0 if sat >= 30 else 0.1)
            if score > best_score:
                best_score = score
                best = i

        return km.cluster_centers_[best] if best >= 0 else None

    def _assign_team_colours(self):
        """Set _team_bgr to distinguishable colours based on hue."""
        hues = [c[0] for c in self.cluster_centers]
        if hues[0] < hues[1]:
            self._team_bgr = [(0, 255, 0), (255, 0, 100)]
        else:
            self._team_bgr = [(255, 0, 100), (0, 255, 0)]
