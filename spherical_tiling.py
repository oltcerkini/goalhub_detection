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

import json
import os

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
TILE_FOV_H = 60.0                 # default horizontal FOV per tile (deg)
TILE_W, TILE_H = 1920, 1080       # 16:9 output -> ~36 deg vertical FOV

# Grid presets for span=180, keyed by tile horizontal FOV (deg).
GRID_180 = {
    60.0: {"yaw": (-85.0, -45.0, 0.0, 45.0, 85.0),          # 5 cols x 2 rows = 10
           "pitch": (-15.0, -45.0)},
    40.0: {"yaw": (-70.0, -50.0, -30.0, -10.0, 10.0, 30.0, 50.0, 70.0),  # 8 x 3 = 24
           "pitch": (-10.0, -30.0, -50.0)},
}
YAW_CENTRES_180 = GRID_180[60.0]["yaw"]
PITCH_CENTRES_180 = GRID_180[60.0]["pitch"]


def build_grid(span, fov_h=TILE_FOV_H):
    """Return [(tile_id, yaw_deg, pitch_deg, fov_h_deg), ...].

    Span 180 (VR180), fov 60: 5 yaw x 2 pitch = 10 tiles. Yaw cols 60 deg each
    cover -90..+90; pitch rows (+-18 deg) cover about +3..-63.
    Span 180, fov 40: 8 yaw x 3 pitch = 24 tiles (1.5x magnification).
    Span 360: a starting point only (needs more columns); not a target here.
    """
    preset = GRID_180.get(float(fov_h))
    if preset is None:
        raise ValueError(f"no grid preset for fov_h={fov_h} (have {list(GRID_180)})")
    tiles, tid = [], 0
    for p in preset["pitch"]:
        for y in preset["yaw"]:
            tiles.append((tid, float(y), float(p), float(fov_h)))
            tid += 1
    return tiles


def hemisphere_mask(yaw_deg, pitch_deg, fov_deg, span, out_w=TILE_W, out_h=TILE_H):
    """Boolean mask of tile pixels whose ray yaw lies within +-span/2.

    For span=180 an edge tile (e.g. yaw +85, 60 deg FOV) reaches past +90; those
    pixels would otherwise be edge-replicated by cv2.remap. Masking them (3c)
    removes the replicated region before detection.
    """
    fx = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    uu, vv = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (uu - out_w / 2.0) / fx
    y = (vv - out_h / 2.0) / fx
    fwd, right, up = _tile_basis(yaw_deg, pitch_deg)
    dx = fwd[0] + x * right[0] - y * up[0]
    dz = fwd[2] + x * right[2] - y * up[2]
    ray_yaw = np.degrees(np.arctan2(dx, dz))
    return np.abs(ray_yaw) <= (span / 2.0 + 1e-6)


def tile_equirect(equirect_img, grid, span, mask_edges=True):
    """Produce tile views: list of (tile_img, yaw, pitch, fov, tile_id).
    No detection here — pure projection (2.3). For span<360 with mask_edges,
    the out-of-hemisphere (edge-replicated) region is blacked out (3c)."""
    out = []
    for (tid, yaw, pitch, fov) in grid:
        img = equirect_to_perspective(equirect_img, yaw, pitch, fov,
                                      TILE_W, TILE_H, span)
        if mask_edges and span < 360:
            m = hemisphere_mask(yaw, pitch, fov, span)
            if not m.all():
                img = np.where(m[:, :, None], img, 0)
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


def _map_bbox_exact(bbox, yaw, pitch, fov, span, eq_w, eq_h):
    """Project the four tile-local bbox corners and take their equirect AABB.
    This replaces the earlier linear approximation (4.1) — no sec^2 error."""
    x1, y1, x2, y2 = bbox
    xs, ys = [], []
    for (u, v) in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        X, Y = tile_pixel_to_equirect(u, v, yaw, pitch, fov, span, eq_w, eq_h)
        xs.append(X); ys.append(Y)
    return [min(xs), min(ys), max(xs), max(ys)]


def _map_bbox_linear(bbox, yaw, pitch, fov, span, eq_w, eq_h):
    """Previous approximation: centre mapped exactly, size scaled by the local
    linear (Jacobian) of the tile->equirect map. Kept only to measure its error."""
    x1, y1, x2, y2 = bbox
    uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    X, Y = tile_pixel_to_equirect(uc, vc, yaw, pitch, fov, span, eq_w, eq_h)
    eps = 4.0
    Xu, Yu = tile_pixel_to_equirect(uc + eps, vc, yaw, pitch, fov, span, eq_w, eq_h)
    Xv, Yv = tile_pixel_to_equirect(uc, vc + eps, yaw, pitch, fov, span, eq_w, eq_h)
    wt, ht = (x2 - x1), (y2 - y1)
    w_eq = abs(Xu - X) / eps * wt + abs(Xv - X) / eps * ht
    h_eq = abs(Yu - Y) / eps * wt + abs(Yv - Y) / eps * ht
    return [X - w_eq / 2, Y - h_eq / 2, X + w_eq / 2, Y + h_eq / 2]


def merge_to_equirect(tile_dets, span, eq_w, eq_h, iou_thr=0.5, bbox_key="bbox"):
    """Map tile detections to equirect pixels, then greedy NMS at IoU `iou_thr`
    (tie-break by confidence). NO merging in tile space.

    Each mapped detection carries three boxes:
      bbox        exact (4 corner projection, 4.1)
      bbox_linear previous linear/Jacobian approximation (for error reporting)
      center      tile-centre mapped exactly
    `bbox_key` selects which box NMS uses (default "bbox").
    """
    mapped = []
    for d in tile_dets:
        x1, y1, x2, y2 = d["bbox"]
        uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cx, cy = tile_pixel_to_equirect(uc, vc, d["yaw"], d["pitch"], d["fov"],
                                        span, eq_w, eq_h)
        exact = _map_bbox_exact(d["bbox"], d["yaw"], d["pitch"], d["fov"], span, eq_w, eq_h)
        lin = _map_bbox_linear(d["bbox"], d["yaw"], d["pitch"], d["fov"], span, eq_w, eq_h)
        mapped.append({
            "tile_id": d["tile_id"], "yaw": d["yaw"], "pitch": d["pitch"],
            "conf": float(d["conf"]), "cls": d.get("cls", 0),
            "bbox": exact, "bbox_linear": lin,
            "center": (cx, cy),
            "yaw_offset_from_tile": abs(((cx / eq_w - 0.5) * span) - d["yaw"]),
        })

    order = sorted(range(len(mapped)), key=lambda i: -mapped[i]["conf"])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(mapped[i][bbox_key], mapped[j][bbox_key]) < iou_thr]
    return [mapped[i] for i in keep], mapped


# --------------------------------------------------------------------------- #
# 3a — pitch ROI masking
# --------------------------------------------------------------------------- #
def grass_roi_polygon(frame, hue=(35, 85), min_area_frac=0.02):
    """Auto-derive a pitch ROI polygon (equirect pixels) from the grass colour.

    Returns an (N,2) int32 polygon (hull of the largest grass region) or None.
    Used as the fallback when a sidecar has no "roi". Detections whose equirect
    centre falls outside are dropped (3a).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = ((hsv[:, :, 0] >= hue[0]) & (hsv[:, :, 0] <= hue[1])
             & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 40)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * frame.shape[0] * frame.shape[1]:
        return None
    return cv2.convexHull(c).reshape(-1, 2).astype(np.int32)


def roi_filter(dets, polygon, margin=0):
    """Split mapped detections into (kept, dropped) by the pitch polygon."""
    if polygon is None:
        return list(dets), []
    poly = np.asarray(polygon, dtype=np.int32)
    kept, dropped = [], []
    for d in dets:
        cx, cy = d["center"]
        inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= -margin
        (kept if inside else dropped).append(d)
    return kept, dropped


# --------------------------------------------------------------------------- #
# 3b — per-video sidecar config
# --------------------------------------------------------------------------- #
def sidecar_path(video_path):
    return os.path.splitext(video_path)[0] + ".equirect.json"


def load_sidecar(video_path):
    """Load <video>.equirect.json if present, else None.

    Format (all optional; missing keys fall back to auto-detection):
        {
          "span": 180,                    # 180 (VR180) or 360
          "crop": [x, y, w, h],           # content region of the letterboxed frame
          "roi": [[x, y], [x, y], ...]    # pitch polygon in equirect pixels
        }
    """
    p = sidecar_path(video_path)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_sidecar(video_path, cfg):
    with open(sidecar_path(video_path), "w") as f:
        json.dump(cfg, f, indent=2)
    return sidecar_path(video_path)
