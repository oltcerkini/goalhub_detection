"""Team classifier — clusters players into teams by jersey color.

Pure 2-cluster KMeans on HSV color samples from player bounding boxes.
Labels: 'My Team', 'Team 2', or 'Unknown'.
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classifies tracked players into 'My Team' / 'Team 2'."""

    def __init__(self, min_samples=10, sample_every=5,
                 torso_ratio=(0.15, 0.55), saturation_threshold=30):
        self.min_samples = min_samples
        self.sample_every = sample_every
        self.torso_ratio = torso_ratio
        self.sat_thresh = saturation_threshold
        self._samples = defaultdict(list)
        self._labels = {}

    def sample(self, frame, track_id, bbox, frame_idx):
        if len(self._samples[track_id]) > 0 and frame_idx % self.sample_every != 0:
            return

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        torso_top = y1 + int((y2 - y1) * self.torso_ratio[0])
        torso_bot = y1 + int((y2 - y1) * self.torso_ratio[1])
        if torso_bot <= torso_top:
            return

        torso = frame[torso_top:torso_bot, x1:x2]
        if torso.size == 0:
            return

        import cv2
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        cy, cx = hsv.shape[0] // 2, hsv.shape[1] // 2
        crop = hsv[cy // 2:cy + cy // 2, cx // 2:cx + cx // 2]
        if crop.size == 0:
            crop = hsv

        pixels = crop.reshape(-1, 3).astype(np.float32)
        mask = pixels[:, 1] > self.sat_thresh
        coloured = pixels[mask]
        if len(coloured) < 5:
            return
        if len(coloured) > 50:
            idxs = np.random.choice(len(coloured), 50, replace=False)
            coloured = coloured[idxs]

        self._samples[track_id].extend(coloured.tolist())

    def cluster(self):
        """Pure 2-cluster KMeans. Labels: 'My Team', 'Team 2'."""
        track_colours = {}
        for tid, samples in self._samples.items():
            if len(samples) < self.min_samples:
                continue
            arr = np.array(samples, dtype=np.float32)
            track_colours[tid] = arr.mean(axis=0)

        tids = list(track_colours.keys())
        if len(tids) < 2:
            print(f"  TeamClassifier: not enough tracks ({len(tids)} < 2)")
            return

        data = np.array([track_colours[t] for t in tids], dtype=np.float32)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=5).fit(data)

        for i, tid in enumerate(tids):
            cluster = int(kmeans.labels_[i])
            self._labels[tid] = "My Team" if cluster == 0 else "Team 2"

        n_my = sum(1 for v in self._labels.values() if v == "My Team")
        n_t2 = sum(1 for v in self._labels.values() if v == "Team 2")
        print(f"  TeamClassifier: {n_my} My Team, {n_t2} Team 2"
              f" ({len(tids)} tracks)")

    def get_team(self, track_id):
        return self._labels.get(track_id, "Unknown")

    @property
    def all_labels(self):
        return dict(self._labels)
