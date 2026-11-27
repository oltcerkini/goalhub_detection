"""Lock-on ball acquisition and dense local tracking.

The far ball is a handful of pixels: a full-frame detector misses it, but an
upscaled patch finds it. A grid scan over the pitch then yields ~15-60
candidates per frame, most of them static look-alikes (posts, scoreboards,
keeper). "Which one is the ball" is a MOTION question, so acquisition ranks
candidates by persistence + displacement across a few scans, then follows the
winner with a cheap per-frame local window. Tracking is ~30 ms/frame; a scan is
~2 s, so scans are only used to acquire and to re-acquire after loss.

This is deliberately separate from BallTrailTracker (full-frame) and does not
change it: use this when the ball is too small for the full-frame pass.
"""

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "calib"))

from ball_zoom import (detect_patches, merge_candidates, windows_around,
                       windows_over_polygon)


def pitch_roi_filter(cal, W, H, margin_x=5.0, margin_z=4.0):
    """Candidate filter: ground projection must be well INSIDE the pitch.

    Used for acquisition only. It rejects touchline/bench/crowd points that the
    pixel polygon lets through (a linesman's foot at z≈34 m won acquisition on
    the goal sequence). Local tracking is unaffected, so the ball near a line is
    still followed once locked.
    """
    from spherical_calib import equirect_to_pitch, DEFAULT_PITCH
    d = dict(DEFAULT_PITCH); d.update(cal.get("pitch") or {})
    limx, limz = d["length"] / 2.0 - margin_x, d["width"] / 2.0 - margin_z
    pose = {"P": np.array(cal["camera_pose"]["position_m"], float),
            "rvec": np.array(cal["camera_pose"]["rotation_vector"], float)}
    span, vspan = float(cal.get("span_deg", 180.0)), float(cal.get("vspan_deg", 180.0))

    def ok(x, y):
        xm, zm = equirect_to_pitch(x, y, pose, W, H, span, vspan)
        xm, zm = float(np.ravel(xm)[0]), float(np.ravel(zm)[0])
        return np.isfinite(xm) and np.isfinite(zm) and abs(xm) <= limx and abs(zm) <= limz
    return ok


def pitch_polygon(cal, W, H, n=48):
    """Pitch outline in content pixels, as the ordered projected border loop.

    Goal-line corners regularly project OUTSIDE the frame (a 150-deg panorama
    puts the far ends past the edge); a convex hull of in-frame points then
    slices the goal areas off and the ball vanishes from the scan. Keep the
    off-frame points and return the ordered loop instead.
    """
    from spherical_calib import pitch_to_equirect, DEFAULT_PITCH
    d = dict(DEFAULT_PITCH); d.update(cal.get("pitch") or {})
    hl, hw = d["length"] / 2.0, d["width"] / 2.0
    pose = {"P": np.array(cal["camera_pose"]["position_m"], float),
            "rvec": np.array(cal["camera_pose"]["rotation_vector"], float)}
    span = float(cal.get("span_deg", 180.0))
    vspan = float(cal.get("vspan_deg", 180.0))
    ring = []
    for x in np.linspace(-hl, hl, n):
        ring.append((x, -hw))
    for z in np.linspace(-hw, hw, n):
        ring.append((hl, z))
    for x in np.linspace(hl, -hl, n):
        ring.append((x, hw))
    for z in np.linspace(hw, -hw, n):
        ring.append((-hl, z))
    ring = np.array(ring)
    ex, ey = pitch_to_equirect(ring[:, 0], ring[:, 1], pose, W, H, span, vspan)
    ex, ey = np.ravel(ex), np.ravel(ey)
    ok = np.isfinite(ex) & np.isfinite(ey)
    keep = np.stack([ex[ok], ey[ok]], 1)
    if len(keep) < 3:
        return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.int32)
    return keep.astype(np.int32)


class BallLockOn:
    """Grid-scan acquisition (motion-ranked) + dense local tracking."""

    def __init__(self, model, W, H, pitch_poly=None, win=300, up=3.0, imgsz=1024,
                 track_gate=110.0, vel_alpha=0.6, max_missed=36,
                 scan_conf=0.12, track_conf=0.08, candidate_filter=None):
        self.model = model
        self.W, self.H = int(W), int(H)
        self.poly = np.asarray(pitch_poly, np.int32) if pitch_poly is not None else None
        self.candidate_filter = candidate_filter      # acquisition-only (x,y)->bool
        self.win = int(win)
        self.up = float(up)
        self.imgsz = int(imgsz)
        self.track_gate = float(track_gate)
        self.vel_alpha = float(vel_alpha)
        self.max_missed = int(max_missed)
        self.scan_conf = float(scan_conf)
        self.track_conf = float(track_conf)
        self.pos = None
        self.vel = np.zeros(2)
        self.tlast = None
        self.missed = 0
        self.state = "idle"
        self.trail = []          # (t, x, y, conf, detected)

    # ---- scanning -------------------------------------------------------- #
    def _scan_windows(self):
        # Scan the WHOLE content: tiling restricted to the pitch polygon drops
        # the boundary windows and with them the ball at the goal line. The
        # polygon is applied to CANDIDATES instead (below), with a margin.
        inset = np.array([[8, 8], [self.W - 8, 8],
                          [self.W - 8, self.H - 8], [8, self.H - 8]], np.int32)
        return windows_over_polygon(inset, self.W, self.H)

    def _on_pitch(self, c, margin_px=60.0):
        if self.poly is None:
            return True
        return cv2.pointPolygonTest(self.poly.astype(np.float32),
                                    (float(c["x"]), float(c["y"])), True) >= -margin_px

    def scan(self, frame):
        wins = self._scan_windows()
        cands = merge_candidates(detect_patches(self.model, frame, wins,
                                                up=self.up, conf=self.scan_conf,
                                                imgsz=self.imgsz), radius=16.0)
        out = [c for c in cands if self._on_pitch(c)]
        if self.candidate_filter is not None:
            out = [c for c in out if self.candidate_filter(c["x"], c["y"])]
        return out

    def local(self, frame, pred):
        wins = windows_around([(pred[0], pred[1])], self.W, self.H, win=self.win)
        c = merge_candidates(detect_patches(self.model, frame, wins,
                                            up=self.up, conf=self.track_conf,
                                            imgsz=self.imgsz), radius=16.0)
        return [z for z in c if np.hypot(z["x"] - pred[0], z["y"] - pred[1]) <= self.track_gate]

    # ---- acquisition ----------------------------------------------------- #
    def _chains(self, scans, times, gate=160.0, min_len=3):
        """Greedy velocity-predicted chains over the scans (all start points)."""
        out = []
        for k0 in range(len(scans)):
            for si in range(len(scans[k0])):
                pts = [scans[k0][si]]
                ks = [k0]
                for k in range(k0 + 1, len(scans)):
                    if len(pts) == 1:
                        dd = np.hypot(scans[k][:, 0] - pts[-1][0],
                                      scans[k][:, 1] - pts[-1][1])
                        j = int(np.argmin(dd))
                    else:
                        v = (pts[-1][:2] - pts[-2][:2]) / (times[ks[-1]] - times[ks[-2]])
                        pred = pts[-1][:2] + v * (times[k] - times[ks[-1]])
                        dd = np.hypot(scans[k][:, 0] - pred[0], scans[k][:, 1] - pred[1])
                        j = int(np.argmin(dd))
                    if dd[j] > gate:
                        break
                    pts.append(scans[k][j])
                    ks.append(k)
                if len(pts) >= min_len:
                    out.append((np.array([p[:2] for p in pts], float),
                                np.array([p[2] for p in pts], float),
                                np.array([times[k] for k in ks], float)))
        return out

    def acquire(self, scan_frames, scan_times, min_speed=8.0, max_speed=500.0,
                straight_min=0.5, min_conf=0.55):
        """Lock onto the chain with high confidence AND plausible, smooth motion.

        A static look-alike is persistent but does not move; an erratic moving
        false positive is not straight. The ball is confident, moving, and
        smooth, so score = mean_conf * (0.6 + 0.4*straightness) among chains
        whose speed is in [min_speed, max_speed]. Among near-ties prefer the
        LONGEST chain (earliest lock). Returns the seed dict or None.
        """
        scans = [np.array([[z["x"], z["y"], z["conf"]] for z in self.scan(f)], float)
                 for f in scan_frames]
        times = np.asarray(scan_times, float)
        cands = []
        for arr, conf, tt in self._chains(scans, times):
            seg = np.diff(arr, axis=0)
            sp = np.linalg.norm(seg, axis=1) / np.maximum(np.diff(tt), 1e-6)
            mean_sp = float(sp.mean())
            if mean_sp < min_speed or mean_sp > max_speed or sp.max() > 2 * max_speed:
                continue
            d = seg / np.maximum(sp[:, None] * np.diff(tt)[:, None], 1e-6)
            straight = float(d.mean(0) @ d.mean(0)) if len(d) else 1.0
            if straight < straight_min or conf.mean() < min_conf:
                continue
            score = float(conf.mean()) * (0.6 + 0.4 * straight)
            cands.append({"score": score, "arr": arr, "conf": conf, "tt": tt,
                          "straight": straight, "speed": mean_sp})
        if not cands:
            return None
        top = max(c["score"] for c in cands)
        near = [c for c in cands if c["score"] >= 0.97 * top]
        best = max(near, key=lambda c: (len(c["arr"]), c["score"]))
        arr, tt = best["arr"], best["tt"]
        v = (arr[-1] - arr[-2]) / max(1e-6, tt[-1] - tt[-2])
        self.pos = arr[-1].copy()
        self.vel = v.astype(float)
        self.tlast = float(tt[-1])
        self.missed = 0
        self.state = "locked"
        self.trail = [(float(t_), float(x), float(y_), float(cf), True)
                      for (x, y_), cf, t_ in zip(arr, best["conf"], tt)]
        seed = {"x": float(arr[-1][0]), "y": float(arr[-1][1]),
                "vx": float(v[0]), "vy": float(v[1]), "conf": float(best["conf"].mean()),
                "n": len(arr), "straight": best["straight"], "speed": best["speed"]}
        return seed

    # ---- per-frame tracking --------------------------------------------- #
    def update(self, t, frame):
        if self.state != "locked":
            return None
        dt = max(1e-3, t - self.tlast) if self.tlast is not None else 1.0 / 24.0
        pred = self.pos + self.vel * min(dt, 0.5)
        cands = self.local(frame, pred)
        if cands:
            c = min(cands, key=lambda z: np.hypot(z["x"] - pred[0], z["y"] - pred[1]))
            newpos = np.array([c["x"], c["y"]], float)
            nv = (newpos - self.pos) / dt
            self.vel = (1 - self.vel_alpha) * self.vel + self.vel_alpha * nv
            self.pos = newpos
            self.missed = 0
            out = (t, float(self.pos[0]), float(self.pos[1]), float(c["conf"]), True)
        else:
            self.pos = pred
            self.vel *= 0.85
            self.missed += 1
            out = (t, float(pred[0]), float(pred[1]), 0.0, False)
            if self.missed > self.max_missed:
                self.state = "lost"
        self.tlast = t
        self.trail.append(out)
        return out
