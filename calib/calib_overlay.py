"""Render a calibration check: the fitted pitch drawn on the traced frame.

The overlay is the honest answer to "does this calibration fit my image?" —
users see the model lines sitting on the painted ones (or not) before trusting
any distances.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from spherical_calib import pitch_to_equirect, pitch_geometry, DEFAULT_PITCH

COLORS = {
    "touchline_near": (0, 0, 235),      # red
    "touchline_far": (235, 120, 0),     # blue
    "goal_line_left": (0, 200, 0),      # green
    "goal_line_right": (0, 160, 0),     # green
    "halfway": (230, 0, 230),           # magenta
    "penalty_box_left": (0, 190, 220),  # amber
    "penalty_box_right": (0, 190, 220),
    "centre_circle": (0, 140, 255),     # orange
    "spot": (255, 255, 255),
}


def _draw_curve(img, pts_px, color, thick=2, max_gap=80.0):
    pts = np.asarray(pts_px, float)
    if len(pts) < 2:
        return
    ok = np.isfinite(pts).all(1)
    prev = None
    for p, good in zip(pts, ok):
        if not good:
            prev = None
            continue
        q = (int(round(p[0])), int(round(p[1])))
        if prev is not None and np.hypot(p[0] - prev[0], p[1] - prev[1]) < max_gap:
            cv2.line(img, (int(prev[0]), int(prev[1])), q, color, thick, cv2.LINE_AA)
        prev = p


def _samples(spec, n):
    if spec[0] == "segment":
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        return a + np.linspace(0, 1, n)[:, None] * (b - a)
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), float(spec[2])
        ang = np.linspace(0, 2 * np.pi, n)
        return np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], 1)
    pts = np.asarray(spec[1], float)
    per = max(2, n // max(len(pts), 1))
    return np.vstack([pts[i] + np.linspace(0, 1, per)[:, None] * (pts[(i + 1) % len(pts)] - pts[i])
                      for i in range(len(pts))])


def _pose_maps(cal):
    W, H = cal["image_size"]
    pose = {"P": cal["camera_pose"]["position_m"],
            "rvec": cal["camera_pose"]["rotation_vector"]}
    return W, H, pose, float(cal["span_deg"]), float(cal["vspan_deg"])


def geometry_polylines(cal, n=200):
    """Projected model curves in image pixels (for drawing in the trace tool)."""
    W, H, pose, span, vspan = _pose_maps(cal)
    geom = pitch_geometry(cal.get("pitch") or {})
    out = {}
    for lab, spec in geom.items():
        m = _samples(spec, n)
        ex, ey = pitch_to_equirect(m[:, 0], m[:, 1], pose, W, H, span, vspan)
        pts = [[round(float(x), 1), round(float(y), 1)]
               for x, y in zip(np.ravel(ex), np.ravel(ey))
               if np.isfinite(x) and np.isfinite(y)]
        out[lab] = pts
    return out


def render_overlay(cal, frame_path, trace_path=None, out_path=None, scale=1.0):
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    W, H = cal["image_size"]
    if (img.shape[1], img.shape[0]) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    pose = {"P": cal["camera_pose"]["position_m"],
            "rvec": cal["camera_pose"]["rotation_vector"]}
    span, vspan = float(cal["span_deg"]), float(cal["vspan_deg"])
    dims = cal.get("pitch") or {}
    geom = pitch_geometry(dims)
    hl = dims.get("length", DEFAULT_PITCH["length"]) / 2.0

    def project(pts_m):
        ex, ey = pitch_to_equirect(pts_m[:, 0], pts_m[:, 1], pose, W, H, span, vspan)
        return np.stack([np.ravel(ex), np.ravel(ey)], -1)

    for lab, spec in geom.items():
        m = _samples(spec, 320)
        _draw_curve(img, project(m), COLORS.get(lab, (200, 200, 200)),
                    thick=2 if spec[0] != "segment" else 3)

    # centre spot
    q = project(np.array([[0.0, 0.0]]))[0]
    if np.isfinite(q).all():
        cv2.circle(img, (int(q[0]), int(q[1])), 5, COLORS["spot"], -1)

    # traced clicks, if available
    if trace_path and Path(trace_path).exists():
        tr = json.loads(Path(trace_path).read_text())
        for ln in tr.get("lines", []):
            col = COLORS.get(ln.get("label"), (255, 255, 255))
            for p in ln.get("points", []):
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, col, -1)

    q = cal.get("quality", {})
    lines = [f"span {span:.1f}  vspan {vspan:.1f}  corners {q.get('corners_in_frame')}/4",
             f"traced->model median {q.get('med_px', float('nan')):.1f} px   "
             f"valid={q.get('valid')}"]
    if q.get("suspect_labels"):
        lines.append("suspect: " + ", ".join(q["suspect_labels"]))
    y = 34
    for txt in lines:
        cv2.putText(img, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34

    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path
