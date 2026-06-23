"""Team classifier — clusters players into exactly 2 teams by jersey colour.

KMeans on weighted HSV samples from player torso.
Always produces exactly 2 teams: 'My Team' and 'Team 2'.
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classifies tracked players into exactly 'My Team' / 'Team 2'."""

    def __init__(self, min_samples=10, sample_every=5,
                 torso_ratio=(0.15, 0.55), sat_threshold=35,
                 sample_gamma=1.4):
        self.min_samples = min_samples
        self.sample_every = sample_every
        self.torso_ratio = torso_ratio
        self.sat_thresh = sat_threshold
        self.sample_gamma = sample_gamma  # >1 darkens midtones, making jersey colours pop
        self._samples = defaultdict(list)
        self._labels = {}
        self._my_team_idx = 0  # cluster index for "My Team" (0 or 1)

    def sample(self, frame, track_id, bbox, frame_idx):
        """Sample HSV from the centre torso — gamma-darkened for better colour separation."""
        if len(self._samples[track_id]) > 0 and frame_idx % self.sample_every != 0:
            return

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        # Torso region (upper body)
        torso_top = y1 + int((y2 - y1) * self.torso_ratio[0])
        torso_bot = y1 + int((y2 - y1) * self.torso_ratio[1])
        if torso_bot <= torso_top:
            return

        torso = frame[torso_top:torso_bot, x1:x2]
        if torso.size == 0:
            return

        import cv2

        # Darken the torso crop so overexposed jersey colours separate better
        if self.sample_gamma != 1.0:
            inv = 1.0 / self.sample_gamma
            table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
            torso = cv2.LUT(torso, table)

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        cy, cx = hsv.shape[0] // 2, hsv.shape[1] // 2
        crop = hsv[cy // 2:cy + cy // 2, cx // 2:cx + cx // 2]
        if crop.size == 0:
            crop = hsv

        pixels = crop.reshape(-1, 3).astype(np.float32)

        # Keep only saturated pixels (coloured jersey, not shorts/skin/background)
        mask = pixels[:, 1] > self.sat_thresh
        coloured = pixels[mask]
        if len(coloured) < 5:
            return
        if len(coloured) > 100:
            idxs = np.random.choice(len(coloured), 100, replace=False)
            coloured = coloured[idxs]

        # Weight: Hue×3, Saturation×1, Value×1 — Hue is the primary colour signal
        weighted = np.empty_like(coloured)
        weighted[:, 0] = coloured[:, 0] * 3.0   # Hue
        weighted[:, 1] = coloured[:, 1] * 1.0   # Saturation
        weighted[:, 2] = coloured[:, 2] * 1.0   # Value

        self._samples[track_id].extend(weighted.tolist())

    def set_my_team(self, team_index):
        """Set which KMeans cluster (0 or 1) corresponds to 'My Team'."""
        self._my_team_idx = team_index

    def cluster(self):
        """KMeans with k=2 — always produces exactly 2 teams."""
        track_feats = {}
        for tid, samples in self._samples.items():
            if len(samples) < self.min_samples:
                continue
            arr = np.array(samples, dtype=np.float32)
            track_feats[tid] = arr.mean(axis=0)

        tids = list(track_feats.keys())
        if len(tids) < 2:
            print(f"  TeamClassifier: not enough tracks ({len(tids)} < 2)")
            return

        data = np.array([track_feats[t] for t in tids], dtype=np.float32)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=5).fit(data)

        for i, tid in enumerate(tids):
            c = int(kmeans.labels_[i])
            self._labels[tid] = "My Team" if c == self._my_team_idx else "Team 2"

        n_my = sum(1 for v in self._labels.values() if v == "My Team")
        n_t2 = sum(1 for v in self._labels.values() if v == "Team 2")
        print(f"  TeamClassifier: {n_my} My Team, {n_t2} Team 2"
              f" ({len(tids)} tracks)")

    def get_team(self, track_id):
        return self._labels.get(track_id, "Unknown")

    @property
    def all_labels(self):
        return dict(self._labels)
