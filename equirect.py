"""VR180 video normalization: raw frame -> canonical equirect content region.

Handles the real-world format variety found in the club's footage:
  - letterboxed mono VR180 (black bars, per-source size)
  - side-by-side stereo VR180 (two eyes side by side)  -> take one eye
  - top-bottom stereo VR180                            -> take one eye
  - dark scenes, where a naive "dark row = black bar" test misfires

Canonical coordinate space for the whole pipeline = the **cropped content
region** in pixels: equirect (x, y) with the origin at the top-left of the
content, W x H = content region size. Everything downstream (tiling, merge,
tracking, calibration) uses this space.

No detector, no model, no pitch geometry here.
"""

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# letterbox / content region (robust to dark scenes)
# --------------------------------------------------------------------------- #
def _content_span(profile, profile_std, dark=16.0, flat=10.0, min_frac=0.25):
    """First/last index of content along one axis.

    A letterbox bar is BOTH dark (low mean) and FLAT (low std). A dark *scene*
    row is dark but textured, so it is kept. Falls back to the full axis.
    """
    bar = (profile <= dark) & (profile_std <= flat)
    idx = np.where(~bar)[0]
    if len(idx) < len(profile) * min_frac:
        return 0, len(profile)
    return int(idx[0]), int(idx[-1]) + 1


def detect_content_region(frame, margin=0):
    """(x, y, w, h) of the non-letterbox content. Robust on dark footage."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    y0, y1 = _content_span(g.mean(1), g.std(1))
    x0, x1 = _content_span(g.mean(0), g.std(0))
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(H, y1 + margin); x1 = min(W, x1 + margin)
    return x0, y0, x1 - x0, y1 - y0


# --------------------------------------------------------------------------- #
# stereo layout
# --------------------------------------------------------------------------- #
def _corr(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def detect_layout(frame, thr=0.90):
    """'mono' | 'sbs' (side-by-side stereo) | 'tb' (top-bottom stereo).

    Two eyes of the same scene correlate very highly; different parts of a mono
    panorama do not."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    if _corr(g[:, :W // 2], g[:, W // 2:]) >= thr:
        return "sbs"
    if _corr(g[:H // 2], g[H // 2:]) >= thr:
        return "tb"
    return "mono"


def de_stereo(frame, layout):
    """Keep a single eye from a stereo frame."""
    if layout == "sbs":
        return frame[:, :frame.shape[1] // 2]
    if layout == "tb":
        return frame[:frame.shape[0] // 2]
    return frame


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
def normalize_frame(frame, span=None, layout=None, region=None):
    """raw BGR frame -> (canonical equirect content BGR, meta).

    meta = {"layout", "region": [x,y,w,h], "size": [W,H], "span"}
    Pass span to force the yaw span; otherwise VR180 (180) is assumed.
    """
    layout = layout or detect_layout(frame)
    eye = de_stereo(frame, layout)
    region = list(region) if region else list(detect_content_region(eye))
    x, y, w, h = region
    content = eye[y:y + h, x:x + w]
    meta = {"layout": layout, "region": region,
            "size": [content.shape[1], content.shape[0]],
            "span": float(span if span is not None else 180.0)}
    return content, meta


# --------------------------------------------------------------------------- #
# raw <-> canonical coordinate conversion
# --------------------------------------------------------------------------- #
def raw_to_canonical(pts, region):
    """raw video pixels -> cropped-content pixels."""
    x, y, _, _ = region
    return [(float(px) - x, float(py) - y) for px, py in pts]


def canonical_to_raw(pts, region):
    """cropped-content pixels -> raw video pixels."""
    x, y, _, _ = region
    return [(float(px) + x, float(py) + y) for px, py in pts]


def probe(path, t_frac=0.4):
    """Quick per-video format report (used by tests / CLI)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * t_frac))
    ok, img = cap.read(); cap.release()
    if not ok:
        return None
    content, meta = normalize_frame(img)
    return {"path": path, "raw": [img.shape[1], img.shape[0]], "fps": round(fps, 2),
            "frames": n, **meta, "aspect": round(meta["size"][0] / meta["size"][1], 2)}


if __name__ == "__main__":
    import glob
    import os
    for f in sorted(glob.glob("assets/180Videos/*.mkv")):
        p = probe(f)
        if not p:
            print(f"{os.path.basename(f)[:36]:36s} READ FAIL"); continue
        print(f"{os.path.basename(f)[:36]:36s} raw {p['raw'][0]}x{p['raw'][1]} "
              f"layout {p['layout']:4s} content {p['size'][0]}x{p['size'][1]} "
              f"aspect {p['aspect']}")
