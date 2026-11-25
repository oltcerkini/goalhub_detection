"""Temporal ball tracker for low-confidence full-frame detector candidates.

The ball model fires at ~0.03-0.25 confidence and also fires on static ground
clutter. A single-frame top-1 pick therefore jumps between real balls and
line junctions. This tracker links candidates by constant-velocity motion:

  * a track must be matched on several frames before it is emitted,
  * static low-confidence tracks are rejected (total movement too small),
  * gaps are bridged by the prediction, and
  * the best track is chosen by hits + confidence + distance travelled.

It works in image pixels and is resolution/geometry agnostic. The caller feeds
one candidate list per video frame (empty list is fine).
"""
import math


class BallTrailTracker:
    def __init__(self, gate_px=60.0, min_hits=3, max_missed=10,
                 min_move_px=20.0, min_speed=0.3, high_conf=0.5):
        self.gate_px = float(gate_px)
        self.min_hits = int(min_hits)
        self.max_missed = int(max_missed)
        self.min_move_px = float(min_move_px)
        self.min_speed = float(min_speed)
        self.high_conf = float(high_conf)
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, frame_idx, candidates):
        """candidates: [{'x','y','conf',...}, ...]; returns best track dict or None."""
        candidates = candidates or []
        for t in self.tracks:
            dt = max(1, frame_idx - t["last"])
            t["px"] = t["x"] + t["vx"] * dt
            t["py"] = t["y"] + t["vy"] * dt

        pairs = []
        for ti, t in enumerate(self.tracks):
            for ci, c in enumerate(candidates):
                d = math.hypot(t["px"] - c["x"], t["py"] - c["y"])
                if d <= self.gate_px:
                    pairs.append((d, ti, ci))
        pairs.sort(key=lambda p: p[0])

        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            t, c = self.tracks[ti], candidates[ci]
            dt = max(1, frame_idx - t["last"])
            nvx = (c["x"] - t["x"]) / dt
            nvy = (c["y"] - t["y"]) / dt
            if t["hits"] >= 2:
                t["vx"] = 0.6 * t["vx"] + 0.4 * nvx
                t["vy"] = 0.6 * t["vy"] + 0.4 * nvy
            else:
                t["vx"], t["vy"] = nvx, nvy
            t["x"], t["y"] = float(c["x"]), float(c["y"])
            t["conf"] = 0.7 * t["conf"] + 0.3 * float(c["conf"])
            t["hits"] += 1
            t["missed"] = 0
            t["last"] = frame_idx
            t["move"] += d
            used_t.add(ti)
            used_c.add(ci)

        matched_ids = {self.tracks[i]["id"] for i in used_t}
        for ci, c in enumerate(candidates):
            if ci in used_c:
                continue
            self.tracks.append({
                "id": self._next_id, "x": float(c["x"]), "y": float(c["y"]),
                "x0": float(c["x"]), "y0": float(c["y"]),
                "vx": 0.0, "vy": 0.0, "conf": float(c["conf"]),
                "hits": 1, "missed": 0, "last": frame_idx,
                "first": frame_idx, "move": 0.0,
                "px": float(c["x"]), "py": float(c["y"]),
            })
            matched_ids.add(self._next_id)
            self._next_id += 1

        for t in self.tracks:
            if t["id"] not in matched_ids:
                t["missed"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]

        best, best_score = None, -1e9
        for t in self.tracks:
            if t["hits"] < self.min_hits:
                continue
            net = math.hypot(t["x"] - t["x0"], t["y"] - t["y0"])
            span = max(1, t["last"] - t["first"])
            speed = net / span
            if t["conf"] < self.high_conf:
                if net < self.min_move_px or speed < self.min_speed:
                    continue
            score = (2.0 * t["conf"] + min(net / 50.0, 5.0)
                     + 0.1 * t["hits"] - 0.1 * t["missed"])
            if score > best_score:
                best, best_score = t, score
        if best is not None:
            best = dict(best)
            best["net"] = math.hypot(best["x"] - best["x0"], best["y"] - best["y0"])
            best["speed"] = best["net"] / max(1, best["last"] - best["first"])
        return best
