"""Zoomed ball search.

The ball at the far goal is ~8-14 px in a 3396-wide frame — too small for the
full-frame detector. But a 200-300 px patch upscaled 3x makes it 25-40 px, and
the model finds it reliably (0.35-0.9 confidence within a few px). This module
scans upscaled patches over the pitch and returns candidates in frame pixels.

Candidates are merged across overlapping windows. Temporal tracking (and static
rejection) happens in the caller, because "is it the ball" is a motion question:
the keeper, posts and scoreboards stay put; the ball does not.
"""
import cv2
import numpy as np

# ball size in CONTENT pixels after mapping back (far ball ~8-16; near ball ~30)
BALL_W_MIN, BALL_W_MAX = 5.0, 34.0
BALL_H_MIN, BALL_H_MAX = 5.0, 34.0
BALL_ASPECT = (0.5, 2.0)

# upscaled-patch detections below this confidence are ignored
ZOOM_CONF = 0.08


def windows_over_polygon(poly, W, H, win=240, stride=120, margin=-40):
    """Tile the pitch polygon with square windows (centre inside the polygon)."""
    poly = np.asarray(poly, np.int32)
    x0s = np.arange(max(0, int(poly[:, 0].min())), max(1, int(poly[:, 0].max())), stride)
    y0s = np.arange(max(0, int(poly[:, 1].min())), max(1, int(poly[:, 1].max())), stride)
    out = []
    for y0 in y0s:
        for x0 in x0s:
            if x0 + win > W or y0 + win > H:
                continue
            if cv2.pointPolygonTest(poly, (x0 + win / 2.0, y0 + win / 2.0), True) < margin:
                continue
            out.append((int(x0), int(y0), int(win)))
    return out


def windows_around(points, W, H, win=220):
    """One window per point (e.g. a player's ground contact), clipped to frame."""
    out = []
    for x, y in points:
        x0 = int(np.clip(x - win / 2.0, 0, max(0, W - win)))
        y0 = int(np.clip(y - win / 2.0, 0, max(0, H - win)))
        out.append((x0, y0, int(win)))
    return out


def detect_patches(model, frame, windows, up=3.0, conf=ZOOM_CONF, imgsz=1024):
    """Run the ball detector on upscaled windows; map hits back to frame px."""
    out = []
    for x0, y0, win in windows:
        crop = frame[y0:y0 + win, x0:x0 + win]
        if crop.shape[0] < win or crop.shape[1] < win:
            continue
        big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        r = model.predict(big, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None:
            continue
        for s, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            bx1, by1, bx2, by2 = map(float, b)
            w, h = (bx2 - bx1) / up, (by2 - by1) / up
            if not (BALL_W_MIN <= w <= BALL_W_MAX and BALL_H_MIN <= h <= BALL_H_MAX
                    and BALL_ASPECT[0] <= w / h <= BALL_ASPECT[1]):
                continue
            out.append({"x": x0 + (bx1 + bx2) / 2.0 / up,
                        "y": y0 + (by1 + by2) / 2.0 / up,
                        "w": w, "h": h, "conf": float(s)})
    return out


def merge_candidates(cands, radius=14.0):
    """Deduplicate candidates seen in overlapping windows (keep the highest conf)."""
    kept = []
    for c in sorted(cands, key=lambda z: -z["conf"]):
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"]) > radius for k in kept):
            kept.append(c)
    return kept


class StaticFilter:
    """Suppress tracks that do not move: posts, the keeper, scoreboards.

    Ball tracks wander; static false positives sit within a few px for seconds.
    Kept separate from BallTrailTracker because the earlier in-tracker version
    produced large outliers.
    """

    def __init__(self, window_s=1.0, min_move_px=14.0, min_age=0.5):
        self.window_s = window_s
        self.min_move_px = min_move_px
        self.min_age = min_age
        self.hist = {}          # id -> [(t, x, y)]

    def is_static(self, tid, t, x, y):
        h = self.hist.setdefault(tid, [])
        h.append((t, x, y))
        while h and t - h[0][0] > self.window_s:
            h.pop(0)
        if t - h[0][0] < self.min_age:
            return False
        xs = [p[1] for p in h]
        ys = [p[2] for p in h]
        return (max(xs) - min(xs)) + (max(ys) - min(ys)) < self.min_move_px
