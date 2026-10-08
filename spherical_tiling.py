"""Phase 2: spherical tiling + equirect-space detection merge for VR180 input.

Standalone module — imports only `spherical_projection`; touches no existing
pipeline module. The canonical output space is **equirect pixel coordinates**
(relative to the cropped content region); metric calibration is a later phase.

CONVENTION (matches spherical_projection.equirect_to_perspective):
  Equirect image centre (W/2, H/2) = yaw 0, pitch 0 (horizon, centre of sphere).
  Yaw increases right (+x in equirect). Pitch increases up (-y in equirect).
  span = horizontal yaw scope:
      180 -> hemisphere (VR180). Yaw range -90 deg .. +90 deg.
      360 -> full sphere (2:1 image).
  Vertical scope is always 180 deg (equirect definition).
"""

import numpy as np
import cv2

from spherical_projection import equirect_to_perspective


# --------------------------------------------------------------------------- #
# 2.8 — letterbox / content-region detection
# --------------------------------------------------------------------------- #
def detect_content_region(frame, thresh=8, min_frac=0.2, margin=0):
    """Locate the non-black content region of a letterboxed frame.

    Returns (x, y, w, h). If the frame has no letterbox (or is too dark to
    tell), returns the full frame. `margin` grows the box slightly to avoid
    clipping soft edges.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    rows = g.mean(axis=1)
    cols = g.mean(axis=0)
    ys = np.where(rows > thresh)[0]
    xs = np.where(cols > thresh)[0]
    if len(ys) < H * min_frac or len(xs) < W * min_frac:
        return 0, 0, W, H
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(H, y1 + margin); x1 = min(W, x1 + margin)
    return x0, y0, x1 - x0, y1 - y0


def apply_region(frame, region):
    x, y, w, h = region
    return frame[y:y + h, x:x + w]


# --------------------------------------------------------------------------- #
# 2.1 — span handling
# --------------------------------------------------------------------------- #
def detect_span(w, h, override=None):
    """Yaw span from content aspect: ~2:1 -> 360 (full sphere), ~1:1 -> 180.

    CAVEAT: unreliable for VR180 stored wide. The Barca VR180 source crops to
    2.18:1 yet is a 180 hemisphere, so this returns 360 for it — Phase 2 passes
    --span 180 explicitly. Aspect is only a proxy for the angular span; the real
    fix is sidecar metadata or a user override.
    """
    if override:
        return float(override)
    return 360.0 if (w / h) >= 1.5 else 180.0


# --------------------------------------------------------------------------- #
# 2.2 / 2.3 — tile grid + tiling
# --------------------------------------------------------------------------- #
TILE_FOV_H = 60.0                 # horizontal FOV per tile (deg)
TILE_W, TILE_H = 1920, 1080       # 16:9 output -> ~36 deg vertical FOV
YAW_CENTRES_180 = (-85.0, -45.0, 0.0, 45.0, 85.0)
PITCH_CENTRES_180 = (-15.0, -45.0)


def build_grid(span):
    """Return [(tile_id, yaw_deg, pitch_deg, fov_h_deg), ...].

    Span 180 (VR180): 5 yaw columns x 2 pitch rows = 10 tiles.
      - yaw cols centred at -85/-45/0/+45/+85 with 60 deg FOV -> each covers
        60 deg, adjacent overlap ~15-25%; union covers the full -90..+90.
      - pitch rows at -15 and -45 with ~36 deg vertical FOV -> each covers
        +-18 deg; union covers about +3 .. -63 deg (the pitch band of an
        elevated VR180 camera).
    Span 360: same grid is a starting point but only covers +-85 of 360; a 360
    run should widen YAW_CENTRES_180 (kept generic, not the Phase 2 target).
    """
    rows = PITCH_CENTRES_180
    tiles = []
    tid = 0
    for p in rows:
        for y in YAW_CENTRES_180:
            tiles.append((tid, float(y), float(p), TILE_FOV_H))
            tid += 1
    return tiles


def tile_equirect(equirect_img, grid, span):
    """Produce tile views: list of (tile_img, yaw, pitch, fov, tile_id).
    No detection here — pure projection (2.3)."""
    out = []
    for (tid, yaw, pitch, fov) in grid:
        img = equirect_to_perspective(equirect_img, yaw, pitch, fov,
                                      TILE_W, TILE_H, span)
        out.append((img, yaw, pitch, fov, tid))
    return out


# --------------------------------------------------------------------------- #
# 2.5 — tile detection -> equirect pixels, then merge
# --------------------------------------------------------------------------- #
def _tile_basis(yaw_deg, pitch_deg):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.cross([0.0, 1.0, 0.0], fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return fwd, right, up


def tile_pixel_to_equirect(u, v, yaw_deg, pitch_deg, fov_deg, span, eq_w, eq_h,
                           out_w=TILE_W, out_h=TILE_H):
    """Inverse of the projection: tile-local pixel -> equirect pixel.

    Tile pixel -> ray direction (pinhole, fx=fy) -> spherical (yaw, pitch) ->
    equirect pixel. This is the exact inverse of equirect_to_perspective.
    """
    fx = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    x = (u - out_w / 2.0) / fx
    y = (v - out_h / 2.0) / fx
    fwd, right, up = _tile_basis(yaw_deg, pitch_deg)
    d = fwd + x * right - y * up
    d = d / np.linalg.norm(d)
    ray_pitch = np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
    ray_yaw = np.degrees(np.arctan2(d[0], d[2]))
    X = (ray_yaw / span + 0.5) * eq_w
    Y = (0.5 - ray_pitch / 180.0) * eq_h
    return float(X), float(Y)


def _iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_to_equirect(tile_dets, span, eq_w, eq_h, iou_thr=0.5):
    """Map tile detections to equirect pixels, then greedy NMS at IoU `iou_thr`
    (tie-break by confidence). NO merging in tile space.

    tile_dets: list of dicts with keys
        tile_id, yaw, pitch, fov, bbox=(x1,y1,x2,y2) tile-local, conf, cls.

    Bbox size approximation: the tile-local box is scaled by the *local linear
    (Jacobian)* of the tile->equirect map at the box centre, i.e. a first-order
    linearisation around the detection. This is exact in the limit of a small
    box; error grows with box size and with yaw distance from the tile centre
    (gnomonic stretch ~ sec^2(yaw_offset)). For player-sized boxes at our tile
    FOV the error is a few percent; it is worst for large boxes near a tile edge.
    """
    mapped = []
    for d in tile_dets:
        x1, y1, x2, y2 = d["bbox"]
        uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        X, Y = tile_pixel_to_equirect(uc, vc, d["yaw"], d["pitch"], d["fov"],
                                      span, eq_w, eq_h)
        eps = 4.0
        Xu, Yu = tile_pixel_to_equirect(uc + eps, vc, d["yaw"], d["pitch"], d["fov"], span, eq_w, eq_h)
        Xv, Yv = tile_pixel_to_equirect(uc, vc + eps, d["yaw"], d["pitch"], d["fov"], span, eq_w, eq_h)
        jxx, jxy = abs(Xu - X) / eps, abs(Xv - X) / eps
        jyx, jyy = abs(Yu - Y) / eps, abs(Yv - Y) / eps
        wt, ht = (x2 - x1), (y2 - y1)
        w_eq = jxx * wt + jxy * ht
        h_eq = jyx * wt + jyy * ht
        cx, cy = X, Y
        # yaw offset from the tile centre -> report the approximation magnitude
        d_yaw = abs(((X / eq_w - 0.5) * span) - d["yaw"])
        mapped.append({
            "tile_id": d["tile_id"], "yaw": d["yaw"], "pitch": d["pitch"],
            "conf": float(d["conf"]), "cls": d.get("cls", 0),
            "bbox": [cx - w_eq / 2, cy - h_eq / 2, cx + w_eq / 2, cy + h_eq / 2],
            "center": (cx, cy), "yaw_offset_from_tile": d_yaw,
            "scale": (w_eq / max(wt, 1e-6)),
        })

    order = sorted(range(len(mapped)), key=lambda i: -mapped[i]["conf"])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(mapped[i]["bbox"], mapped[j]["bbox"]) < iou_thr]
    return [mapped[i] for i in keep], mapped
