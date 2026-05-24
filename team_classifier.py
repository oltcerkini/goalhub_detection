"""Colour-based team classification using K-means on HSV jersey regions."""

import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamClassifier:
    """Clusters detected players into two teams by dominant jersey colour."""

    def __init__(self):
        self.labels = {}         # detection-index -> 0|1
        self.cluster_centers = None  # (2, 3) HSV
        self.my_team = None
        self._team_bgr = [(255, 0, 0), (0, 0, 255)]  # blue, red

    def classify(self, image, detections):
        """Assign team labels (0/1) to each detection.

        Args:
            image: BGR numpy array
            detections: sv.Detections (pre-filtered to inside-pitch)

        Returns:
            dict mapping detection-index -> team (0 or 1), or None if < 4 players
        """
        self.labels = {}
        features = []    # (index, hsv_vector)
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
        # Assign team BGR colours based on cluster hue
        self._assign_team_colours()
        print(f"  Classified {len(self.labels)} players into 2 teams.")
        return self.labels

    def set_my_team(self, team_idx):
        """Set which team (0 or 1) is 'my team'."""
        self.my_team = team_idx
        if team_idx == 0:
            self._team_bgr = [(0, 255, 0), (100, 100, 100)]  # green vs gray
        else:
            self._team_bgr = [(100, 100, 100), (0, 255, 0)]
        print(f"  My team → {'Team 1' if team_idx == 0 else 'Team 2'}")

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
            # Nearest-centroid assignment
            dist0 = np.linalg.norm(feat - c0)
            dist1 = np.linalg.norm(feat - c1)
            self.labels[i] = 0 if dist0 < dist1 else 1

        return self.labels

    def get_colour(self, detection_idx, bgr=True):
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
    def _extract_jersey_colour(self, player_crop):
        """Extract dominant HSV from the upper-half centre of a player bbox."""
        h, w = player_crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # Upper 50% (jersey area), centre 60% horizontally (avoid background)
        top = int(h * 0.05)
        bot = int(h * 0.55)
        left = int(w * 0.20)
        right = int(w * 0.80)
        if top >= bot or left >= right:
            return None

        region = player_crop[top:bot, left:right]
        if region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # K-means on this crop to find dominant colour (ignoring dark/bg)
        k = min(3, len(pixels))
        km = KMeans(n_clusters=k, random_state=0, n_init=3, max_iter=10)
        km.fit(pixels)
        counts = np.bincount(km.labels_)
        # Pick the cluster with highest saturation (most colourful = jersey)
        best = np.argmax([
            km.cluster_centers_[i][1] * counts[i]
            for i in range(k)
        ])
        return km.cluster_centers_[best]

    def _assign_team_colours(self):
        """Set _team_bgr to distinguishable colours based on hue."""
        hues = [c[0] for c in self.cluster_centers]
        # Team 0 gets the more-red hue, team 1 the more-blue
        if hues[0] < hues[1]:
            self._team_bgr = [(255, 0, 0), (0, 0, 255)]
        else:
            self._team_bgr = [(0, 0, 255), (255, 0, 0)]
