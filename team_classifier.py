"""Team classifier — clusters players into exactly 2 teams by jersey colour.

Frame-by-frame KMeans with majority voting across the entire video.
Each frame votes independently; final team = majority decision across all frames.
This makes the classification robust to lighting variations — a few badly-lit
frames get outvoted by the rest.
"""

import numpy as np
from collections import defaultdict, Counter
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
        self.sample_gamma = sample_gamma
        # Per-frame storage: frame_idx -> {track_id -> mean_hsv_vector}
        self._frame_data = {}
        self._labels = {}
        self._my_team_idx = 0

    def sample(self, frame, track_id, bbox, frame_idx):
        """Sample HSV centre-torso, store per-frame mean for this track."""
        if frame_idx not in self._frame_data:
            self._frame_data[frame_idx] = {}

        # Only sample every Nth frame
        if len(self._frame_data) > 0 and frame_idx % self.sample_every != 0:
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

        # Filter to saturated pixels only
        mask = pixels[:, 1] > self.sat_thresh
        coloured = pixels[mask]
        if len(coloured) < 5:
            return
        if len(coloured) > 100:
            idxs = np.random.choice(len(coloured), 100, replace=False)
            coloured = coloured[idxs]

        # Store the MEAN colour for this (frame, track) — one vector per frame per track
        mean_colour = coloured.mean(axis=0)
        self._frame_data[frame_idx][track_id] = mean_colour

    def cluster(self):
        """Frame-by-frame KMeans, then majority vote across the whole video.

        For each frame with enough players, run 2-cluster KMeans and record
        which team each player was assigned to. Then each player's final team
        is the one they were assigned to most often across all frames.
        """
        # Group tracks by frame: frame_id -> [(track_id, hsv_vector), ...]
        frame_votes = defaultdict(list)  # track_id -> [0 or 1 per frame]

        for frame_idx, tracks in self._frame_data.items():
            tids = list(tracks.keys())
            feats = np.array([tracks[t] for t in tids], dtype=np.float32)

            if len(tids) < 2:
                continue

            # Simple weighted: Hue×3 for discrimination
            weighted = feats.copy()
            weighted[:, 0] *= 3.0

            kmeans = KMeans(n_clusters=2, random_state=0, n_init=3).fit(weighted)
            for i, tid in enumerate(tids):
                frame_votes[tid].append(int(kmeans.labels_[i]))

        # Majority vote across all frames for each track
        for tid, votes in frame_votes.items():
            most_common = Counter(votes).most_common(1)[0][0]
            self._labels[tid] = "My Team" if most_common == self._my_team_idx else "Team 2"

        n_my = sum(1 for v in self._labels.values() if v == "My Team")
        n_t2 = sum(1 for v in self._labels.values() if v == "Team 2")
        n_total = len(frame_votes)
        print(f"  TeamClassifier: {n_my} My Team, {n_t2} Team 2"
              f" ({n_total} tracks)")

    def set_my_team(self, team_index):
        """Set which KMeans cluster (0 or 1) corresponds to 'My Team'."""
        self._my_team_idx = team_index

    def get_team(self, track_id):
        return self._labels.get(track_id, "Unknown")

    @property
    def all_labels(self):
        return dict(self._labels)
