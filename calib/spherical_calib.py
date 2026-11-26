"""Phase B/C — spherical ground-plane calibration for equirect (180) footage.

Solves the camera pose (position P + orientation R, 6 DoF) in the PITCH frame
from traced pitch curves in equirect pixels, then maps equirect pixels <-> metric
pitch coordinates on the flat ground plane (y = 0).

Convention (matches spherical_projection / spherical_tiling):
  equirect pixel -> direction in the EQUIRECT frame  [+Y up, +Z at yaw 0]
      yaw   = (x_eq / W - 0.5) * span                 (deg)
      pitch = (0.5 - y_eq / H) * 180                  (deg)
      d_e   = (sin yaw cos pitch, sin pitch, cos yaw cos pitch)
  The pose places that frame in the PITCH frame:
      d_pitch = R @ d_e
      ground  = P + t * d_pitch,   t = -P.y / d_pitch.y   (needs d_pitch.y < 0)

Pitch frame: y up (ground at y=0); x along the length (goal lines at x=+-52.5);
z along the width (touchlines at z=+-34); centre spot at the origin.

Residuals are EXACT distances to the known lines / circle / box outline (no
polyline sampling, which otherwise quantises the fit).
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

# ---- standard dimensions (metres); defaults, overridable by the user ------ #
DEFAULT_PITCH = {
    "length": 105.0, "width": 68.0,
    "penalty_depth": 16.5, "penalty_half": 20.16,   # 40.32 m wide box
    "circle_r": 9.15, "goal_half_width": 3.66, "goal_height": 2.44,
}
# legacy aliases (other modules import these)
PITCH_LEN, PITCH_WID = DEFAULT_PITCH["length"], DEFAULT_PITCH["width"]
PENALTY_BOX_DEPTH, PENALTY_BOX_HALF = DEFAULT_PITCH["penalty_depth"], DEFAULT_PITCH["penalty_half"]
CENTRE_CIRCLE_R = DEFAULT_PITCH["circle_r"]
GOAL_HALF_WIDTH, GOAL_HEIGHT = DEFAULT_PITCH["goal_half_width"], DEFAULT_PITCH["goal_height"]


def pitch_geometry(dims=None):
    """label -> spec in (x, z) metres.

    dims: optional dict overriding DEFAULT_PITCH keys (the user supplies the
    real pitch size, e.g. a youth 9v9 pitch).  Specs:
      ("line", a, b)        infinite line through a,b
      ("circle", c, r)      circle centre c radius r
      ("poly", [p0..pn])    closed outline (segments, clamped)
    """
    d = dict(DEFAULT_PITCH)
    d.update(dims or {})
    hl, hw = d["length"] / 2, d["width"] / 2
    pd, ph = d["penalty_depth"], d["penalty_half"]
    return {
        "touchline_near": ("segment", (-hl, hw), (hl, hw)),
        "touchline_far": ("segment", (-hl, -hw), (hl, -hw)),
        "goal_line_left": ("segment", (-hl, -hw), (-hl, hw)),
        "goal_line_right": ("segment", (hl, -hw), (hl, hw)),
        "halfway": ("segment", (0, -hw), (0, hw)),
        "penalty_box_left": ("poly", [(-hl, -ph), (-hl + pd, -ph), (-hl + pd, ph), (-hl, ph)]),
        "penalty_box_right": ("poly", [(hl, -ph), (hl - pd, -ph), (hl - pd, ph), (hl, ph)]),
        "centre_circle": ("circle", (0.0, 0.0), d["circle_r"]),
    }


def _line_dist(g, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return np.abs((g - a) @ n)


def _seg_dist(g, a, b):
    """Clamped point-to-SEGMENT distance. Unlike the infinite line, the endpoints
    matter — without this the traced points can slide along the line and the pose
    is under-constrained (a degenerate fit that does not reproduce the image)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.asarray(g, float)
    single = g.ndim == 1
    gg = g[None, :] if single else g
    ab = b - a
    t = np.clip(((gg - a) @ ab) / (ab @ ab), 0, 1)
    d = np.linalg.norm(gg - (a + t[:, None] * ab), axis=1)
    return float(d[0]) if single else d


def _poly_dist(g, pts):
    pts = np.asarray(pts, float)
    g = np.asarray(g, float)
    single = g.ndim == 1
    gg = g[None, :] if single else g
    best = None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ab = b - a
        t = np.clip(((gg - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(gg - (a + t[:, None] * ab), axis=1)
        best = dd if best is None else np.minimum(best, dd)
    return float(best[0]) if single else best


def label_residual(g, spec):
    """g: (N,2) ground points (x,z). -> (N,) min distance in metres to the line."""
    if spec[0] == "line":
        return _line_dist(g, spec[1], spec[2])
    if spec[0] == "segment":
        return _seg_dist(g, spec[1], spec[2])
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), spec[2]
        return np.abs(np.linalg.norm(np.asarray(g, float) - c, axis=-1) - r)
    return _poly_dist(g, spec[1])


def nearest_on(g, spec):
    """Nearest point on the known line/circle/outline to a single (x,z) point."""
    g = np.asarray(g, float)
    if spec[0] == "line":
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        d = b - a
        return a + np.dot(g - a, d) / np.dot(d, d) * d
    if spec[0] == "segment":
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        d = b - a
        t = np.clip(np.dot(g - a, d) / np.dot(d, d), 0, 1)
        return a + t * d
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), spec[2]
        v = g - c
        n = np.linalg.norm(v)
        return c if n < 1e-9 else c + v / n * r
    pts = np.asarray(spec[1], float)
    best, bd = None, np.inf
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ab = b - a
        t = np.clip(np.dot(g - a, ab) / np.dot(ab, ab), 0, 1)
        q = a + t * ab
        dd = np.linalg.norm(g - q)
        if dd < bd:
            best, bd = q, dd
    return best


# ---- rotations ------------------------------------------------------------ #
def rotmat(rvec):
    rvec = np.asarray(rvec, float)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rvec_from_matrix(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def rot_angle(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---- projections ---------------------------------------------------------- #
def equirect_dir(x_eq, y_eq, W, H, span, vspan=180.0):
    """equirect pixel -> unit direction in the equirect frame.

    span  = yaw scope (deg) mapped across the image WIDTH  (180 for VR180).
    vspan = pitch scope (deg) mapped across the image HEIGHT. NOT always 180:
    the club's clips crop to very different content aspect ratios (0.88..4.55),
    so the vertical span is fitted by the solver rather than assumed.
    """
    x_eq, y_eq = np.asarray(x_eq, float), np.asarray(y_eq, float)
    yaw = np.radians((x_eq / W - 0.5) * span)
    pit = np.radians((0.5 - y_eq / H) * vspan)
    return np.stack([np.sin(yaw) * np.cos(pit), np.sin(pit), np.cos(yaw) * np.cos(pit)], -1)


def equirect_to_pitch(x_eq, y_eq, pose, W, H, span, vspan=180.0):
    """equirect pixel -> metric ground (x_m, z_m). Rays above the horizon are
    pushed far away so the solver gets a large, smooth residual."""
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    d = equirect_dir(x_eq, y_eq, W, H, span, vspan) @ R.T
    dy = np.minimum(d[..., 1], -1e-3)
    t = -P[1] / dy
    g = P[None, :] + t[..., None] * d
    return g[..., 0], g[..., 2]


def pitch_to_equirect(x_m, z_m, pose, W, H, span, vspan=180.0):
    """metric ground (x_m, z_m) -> equirect pixel."""
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    x_m, z_m = np.asarray(x_m, float), np.asarray(z_m, float)
    g = np.stack([x_m, np.zeros_like(x_m), z_m], -1)
    d_p = g - P[None, :]
    d_p = d_p / np.linalg.norm(d_p, axis=-1, keepdims=True)
    d_e = d_p @ R
    yaw = np.degrees(np.arctan2(d_e[..., 0], d_e[..., 2]))
    pit = np.degrees(np.arcsin(np.clip(d_e[..., 1], -1, 1)))
    return (yaw / span + 0.5) * W, (0.5 - pit / vspan) * H


# ---- solver --------------------------------------------------------------- #
def _group(traces):
    g = {}
    for x, y, lab in traces:
        g.setdefault(lab, []).append((x, y))
    return {k: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
            for k, v in g.items()}


def _rms_px(traces, pose, geom, W, H, span, vspan=180.0):
    errs = []
    for x, y, lab in traces:
        if lab not in geom:
            continue
        gx, gz = equirect_to_pitch(x, y, pose, W, H, span, vspan)
        gx, gz = float(np.ravel(gx)[0]), float(np.ravel(gz)[0])
        if not (np.isfinite(gx) and np.isfinite(gz)):
            continue
        q = nearest_on((gx, gz), geom[lab])
        ex, ey = pitch_to_equirect(q[0], q[1], pose, W, H, span, vspan)
        errs.append(np.hypot(float(np.ravel(ex)[0]) - x, float(np.ravel(ey)[0]) - y))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def solve_pose(traces, W, H, span=180.0, geom=None, seed_inits=None, max_nfev=400,
               fit_vspan=False, vspan0=180.0, loss="linear", f_scale=1.0):
    """Fit camera pose (and optionally the vertical span) so traced equirect
    curves land on the known pitch lines.

    fit_vspan=True adds the pitch scope across the image height as a 7th
    parameter, so clips whose content aspect is not ~1:1 do not need the
    caller to know their vertical FOV.

    loss/f_scale: scipy robust loss ("soft_l1"/"huber"). A mistraced line near
    the horizon can be 100x off; a robust loss keeps it from swamping the fit.
    """
    geom = geom or pitch_geometry()
    groups = _group([(float(x), float(y), l) for x, y, l in traces if l in geom])
    n_par = 7 if fit_vspan else 6

    def unpack(p):
        return {"P": p[:3], "rvec": p[3:6]}, (p[6] if fit_vspan else vspan0)

    def resid(params):
        pose, vs = unpack(params)
        out = []
        for lab, (xs, ys) in groups.items():
            gx, gz = equirect_to_pitch(xs, ys, pose, W, H, span, vs)
            g = np.stack([gx, gz], -1)
            out.append(np.clip(label_residual(g, geom[lab]), 0, 80.0))
        return np.concatenate(out) if out else np.zeros(1)

    if seed_inits is None:
        seed_inits = []
        for h in (4.0, 8.0, 15.0):
            for dist in (40.0, 75.0):
                for a in (0, 90, 180, 270):
                    ang = np.radians(a)
                    seed_inits.append([dist * np.cos(ang), h, dist * np.sin(ang), 0, 0, 0])

    lo = np.array([-200, 0.5, -200, -np.pi, -np.pi, -np.pi])
    hi = np.array([200, 120, 200, np.pi, np.pi, np.pi])
    for x0 in ([[40, 8, 0, 0, 0, 0, 180.0], [60, 5, -40, 0, 0, 0, 120.0],
                [-40, 12, 40, 0, 0, 0, 90.0], [0, 10, 0, 0, 0, 0, 60.0]] if fit_vspan else []):
        seed_inits.append(x0)
    if fit_vspan:
        lo = np.append(lo, 20.0); hi = np.append(hi, 180.0)

    best = None
    for x0 in seed_inits:
        x0 = list(x0)
        if fit_vspan and len(x0) == 6:
            x0 = x0 + [vspan0]
        try:
            r = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), method="trf",
                              max_nfev=max_nfev, xtol=1e-12, ftol=1e-12,
                              loss=loss, f_scale=f_scale)
        except Exception:
            continue
        c = float(r.cost)                 # robust cost when loss != linear
        if best is None or c < best[0]:
            best = (c, r.x.copy())
    if best is None:
        raise RuntimeError("solver failed on all inits")
    cost, p = best
    pose, vspan = unpack(p)
    res = resid(p)
    sse = float(np.sum(res ** 2))
    n = sum(len(v[0]) for v in groups.values())
    return {"P": list(p[:3]), "rvec": list(p[3:6]), "R": rotmat(p[3:6]).tolist(),
            "vspan": float(vspan), "cost": cost,
            "rms_m": float(np.sqrt(sse / max(n, 1))), "med_m": float(np.median(res)),
            "rms_px": _rms_px(traces, pose, geom, W, H, span, vspan),
            "n": int(n), "labels": sorted(groups)}


# ---- pixel-space refinement ----------------------------------------------- #
# The metre-space fit above is dominated by the near camera: a far-field trace
# can be 100+ px wrong while barely moving the metric residual. Refine the pose
# (and the spans, which are NOT reliably 180 in cropped VR180) so the projected
# pitch curves land on the traced pixels — that is exactly what the user drew.
# Median px distance as a fraction of image width. 0.45% of W (~15 px at 3400)
# corresponds to roughly 0.7 deg of projection error — about 0.4 m at 30 m.
PX_VALID_FRAC = 0.0045
PX_OUTLIER_MIN = 10.0        # a label worse than this median px is a mistrace candidate
PX_OUTLIER_FACTOR = 3.0      # ... or this many times the overall label median


def px_valid_threshold(W):
    return max(8.0, PX_VALID_FRAC * float(W))


def _dense_curve(spec, n=800):
    """Dense samples of a pitch feature, for projecting it into the image."""
    if spec[0] in ("line", "segment"):
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        return a + np.linspace(0, 1, n)[:, None] * (b - a)
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), float(spec[2])
        ang = np.linspace(0, 2 * np.pi, n)
        return np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], 1)
    pts = np.asarray(spec[1], float)
    if len(pts) > 1:                              # closed outline -> subdivide
        seg = max(1, n // len(pts))
        out = [pts[i] + np.linspace(0, 1, seg)[:, None] * (pts[(i + 1) % len(pts)] - pts[i])
               for i in range(len(pts))]
        return np.vstack(out)
    return pts


def _project_curve(pts_m, pose, W, H, span, vspan):
    ex, ey = pitch_to_equirect(pts_m[:, 0], pts_m[:, 1], pose, W, H, span, vspan)
    ex, ey = np.ravel(ex), np.ravel(ey)
    ok = np.isfinite(ex) & np.isfinite(ey)
    return np.stack([ex[ok], ey[ok]], 1)


def fit_pixel(traces, W, H, geom, pose0, span0, vspan0, max_nfev=120, f_scale=2.0):
    """Refine (P, rvec, span, vspan) minimising traced-pixel -> model px distance.

    traces: [(x_px, y_px, label)] already parity-mapped. points are compared to
    the DENSELY projected model curve of their own label, so no ground-plane
    inversion is needed (rays above the horizon are handled naturally).
    """
    groups = {}
    for x, y, lab in traces:
        groups.setdefault(lab, []).append((x, y))
    groups = {k: np.asarray(v, float) for k, v in groups.items()}
    curves = {lab: _dense_curve(geom[lab]) for lab in groups}

    def per_point(p):
        pose = {"P": p[:3], "rvec": p[3:6]}
        span, vs = float(p[6]), float(p[7])
        out = []
        for lab, pts in groups.items():
            c = _project_curve(curves[lab], pose, W, H, span, vs)
            if len(c) == 0:
                out.append(np.full(len(pts), 500.0)); continue
            out.append(cKDTree(c).query(pts)[0])
        return np.concatenate(out)

    lo = np.array([-200, 0.5, -200, -np.pi, -np.pi, -np.pi, 120.0, 20.0])
    hi = np.array([200, 120, 200, np.pi, np.pi, np.pi, 220.0, 200.0])
    best = None
    for sp, vs in ((span0, vspan0), (180.0, 180.0), (180.0, round(180.0 * H / W, 1)),
                   (150.0, 70.0)):
        x0 = np.clip(np.array([*pose0["P"], *pose0["rvec"], sp, vs], float),
                     lo + 1e-6, hi - 1e-6)
        try:
            r = least_squares(per_point, x0, bounds=(lo, hi), method="trf",
                              max_nfev=max_nfev, xtol=1e-10, ftol=1e-10,
                              loss="soft_l1", f_scale=f_scale)
        except Exception:
            continue
        d = per_point(r.x)
        k90 = max(1, int(0.9 * len(d)))
        trim = float(np.mean(np.sort(d)[:k90]))
        if best is None or trim < best[0]:
            best = (trim, r.x.copy(), d)
    if best is None:
        raise RuntimeError("pixel refinement failed")
    trim, p, d = best
    pose = {"P": p[:3], "rvec": p[3:6], "R": rotmat(p[3:6]).tolist()}
    span, vs = float(p[6]), float(p[7])
    per, idx = {}, 0
    for lab, pts in groups.items():
        dd = d[idx:idx + len(pts)]; idx += len(pts)
        per[lab] = (len(dd), float(np.median(dd)), float(np.sqrt(np.mean(dd ** 2))))
    med = float(np.median(d))
    rms = float(np.sqrt(np.mean(d ** 2)))
    return {"pose": pose, "span": span, "vspan": vs, "med_px": med,
            "trim90_px": trim, "rms_px": rms, "per": per,
            "n": int(len(d))}


# ---- calibration contract (from the pencil-tool trace file) --------------- #
# v2: span/vspan are solved (pixel-space fit), not assumed; quality is pixel-based.
CALIB_VERSION = 2
RMS_VALID_M = 1.0          # metric RMS below this -> calibration_valid

# The traced goal lines are labelled by IMAGE side (calibrate.html: left/right
# edge of the frame), while the model's camera frame puts +x on the image LEFT.
# A rectangular pitch is 180-deg rotationally symmetric, so a mirror-disagreement
# between the two conventions is invisible in the metres residual: the solver
# happily fits the mirrored pose and the pitch renders off-frame. Flipping the
# two goal labels is the one-pair swap that resolves the handedness.
GOAL_FLIP = {"goal_line_left": "goal_line_right", "goal_line_right": "goal_line_left"}


def _corners_in_frame(pose, W, H, span, vspan, dims=None):
    """How many pitch corners project into (or just outside) the frame."""
    d = dict(DEFAULT_PITCH); d.update(dims or {})
    hl, hw = d["length"] / 2.0, d["width"] / 2.0
    n = 0
    for xm, zm in [(-hl, hw), (hl, hw), (hl, -hw), (-hl, -hw)]:
        ex, ey = pitch_to_equirect(xm, zm, pose, W, H, span, vspan)
        ex, ey = float(np.ravel(ex)[0]), float(np.ravel(ey)[0])
        if -0.06 * W <= ex < 1.06 * W and -0.06 * H <= ey < 1.06 * H:
            n += 1
    return n


def calibration_from_traces(src, span=None, verbose=True):
    """src: path to calib_equirect.json, or the parsed dict.

    Input:  {image, size:[W,H], span?, crop?, pitch?{...dimensions...},
             lines:[{label, points:[[x,y],..]}]}
    Output: a versioned calibration contract (see CALIB_VERSION).
    """
    import json
    import os
    data = json.load(open(src)) if isinstance(src, (str, os.PathLike)) else src
    W, H = (data.get("size") or data.get("image_size"))
    span = float(span if span is not None else data.get("span", 180.0))
    dims = data.get("pitch") or {}
    geom = pitch_geometry(dims)

    traces = []
    used = {}
    dropped = 0
    for ln in data.get("lines", []):
        lab = ln.get("label")
        if lab not in geom:
            continue
        for pt in ln.get("points", []):
            x, y = float(pt[0]), float(pt[1])
            if not (0 <= x < W and 0 <= y < H):     # off-frame pixel = not a valid sample
                dropped += 1
                continue
            traces.append((x, y, lab))
            used[lab] = used.get(lab, 0) + 1
    if verbose and dropped:
        print(f"  dropped {dropped} point(s) outside the {int(W)}x{int(H)} frame")
    if not traces:
        raise RuntimeError("no usable traces (labels must match the known pitch features)")

    # vspan is only fitted when explicitly requested. Freeing it is ill-conditioned
    # (span <-> camera pose trade off), so a bad fit could pass validation while
    # corrupting metres. Default: assume 180 and let a wrong assumption fail loudly.
    fit_vs = bool(data.get("fit_vspan", False))
    vspan0 = float(data.get("vspan_deg", 180.0))

    # robust loss: a mistraced near-horizon line (~1 px = metres) can be 100x off
    # and would otherwise swamp the least-squares fit.
    # Try both goal-label parities (see GOAL_FLIP) and keep the pose that actually
    # renders the pitch in frame, then the lower robust cost.
    best = None
    for flip in (False, True):
        tr = [(x, y, GOAL_FLIP.get(l, l) if flip else l) for x, y, l in traces]
        try:
            s = solve_pose(tr, W, H, span, geom=geom, fit_vspan=fit_vs,
                           vspan0=vspan0, loss="soft_l1", f_scale=1.0)
        except Exception:
            continue
        cif = _corners_in_frame({"P": np.array(s["P"]), "rvec": np.array(s["rvec"])},
                                W, H, span, s["vspan"], dims)
        # a degenerate pose can have a deceptively low cost; in-frame corners first
        key = (cif, -float(s["cost"]))
        if best is None or key > best[0]:
            best = (key, s, flip, cif)
    if best is None:
        raise RuntimeError("pose solve failed for both goal-label parities")
    _, sol, _parity_flip, in_frame = best
    pose = {"P": np.array(sol["P"]), "rvec": np.array(sol["rvec"])}
    vspan_used = float(sol["vspan"])
    label_parity = "goal_lines_flipped" if _parity_flip else "as_traced"
    if _parity_flip:
        # keep every downstream residual consistent with the chosen convention
        traces = [(x, y, GOAL_FLIP.get(l, l)) for x, y, l in traces]
        if verbose:
            print("  goal_line_left/right flipped to match the model's camera frame")

    # ---- pixel-space refinement (accurate fit; see fit_pixel) --------------
    # Reject labels the model cannot reconcile (e.g. a touchline traced above
    # the horizon), then refit. This runs on the traces in pixel space, so what
    # it optimises is exactly what the user sees in the overlay.
    dropped = []
    px = None
    for _ in range(3):
        try:
            px = fit_pixel(traces, W, H, geom, pose, span, vspan_used)
        except Exception as e:                              # noqa: BLE001
            if verbose:
                print(f"  pixel refinement unavailable ({e}); using metric fit")
            px = None
            break
        meds_px = {l: v[1] for l, v in px["per"].items()}
        if len(meds_px) <= 3:
            break
        overall_px = float(np.median(list(meds_px.values())))
        worst = max(meds_px, key=meds_px.get)
        n_goal = sum(1 for l in meds_px if l.startswith("goal_line_"))
        last_goal = worst.startswith("goal_line_") and n_goal <= 1
        if meds_px[worst] > max(PX_OUTLIER_MIN, PX_OUTLIER_FACTOR * overall_px) and not last_goal:
            dropped.append(worst)
            traces = [(x, y, l) for x, y, l in traces if l != worst]
            pose = px["pose"]
            span, vspan_used = px["span"], px["vspan"]
            if verbose:
                print(f"  dropped mistraced label {worst} "
                      f"(median {meds_px[worst]:.1f} px vs {overall_px:.1f} px)")
            continue
        break
    if px is not None:
        pose = px["pose"]
        span, vspan_used = px["span"], px["vspan"]
        med_px = px["med_px"]
        trim90_px = px["trim90_px"]
        rms_px = px["rms_px"]
    else:
        med_px = trim90_px = float("nan")
        rms_px = _rms_px(traces, pose, geom, W, H, span, vspan_used)
    in_frame = _corners_in_frame(pose, W, H, span, vspan_used, dims)

    # metric residual under the final pose (reported; not the validity gate)
    per = {}
    allres = []
    for x, y, l in traces:
        gx, gz = equirect_to_pitch(x, y, pose, W, H, span, vspan_used)
        g = np.array([float(np.ravel(gx)[0]), float(np.ravel(gz)[0])])
        r = min(float(np.ravel(label_residual(g, geom[l]))[0]), 80.0)
        per.setdefault(l, []).append(r)
        allres.append(r)
    allres = np.array(allres)
    med_m = float(np.median(allres))
    rms_m = float(np.sqrt(np.mean(allres ** 2)))
    used = {l: len(v) for l, v in per.items()}

    # a label whose pixels still do not sit on its curve is suspect even if the
    # automatic drop did not trigger (e.g. it was the best of a bad bunch)
    suspect = sorted(set(dropped))
    if px is not None:
        suspect += sorted(l for l, v in px["per"].items()
                          if v[1] > max(PX_OUTLIER_MIN, PX_OUTLIER_FACTOR * med_px))
    suspect = sorted(set(suspect))

    # GEOMETRIC sanity gate: a correct pose renders the pitch inside the frame,
    # and the traced pixels must actually land on the projected curves.
    px_thresh = px_valid_threshold(W)
    valid = bool(px is not None and med_px <= px_thresh
                 and len(allres) >= 12 and in_frame >= 2)
    span_at_bound = bool(span <= 120.01 or span >= 219.99
                         or vspan_used <= 20.01 or vspan_used >= 199.99)
    out = {
        "version": CALIB_VERSION,
        "projection": "equirect",
        "span_deg": span,
        "vspan_deg": vspan_used,
        "image_size": [int(W), int(H)],
        "content_region": data.get("crop"),
        "source_image": data.get("image"),
        "pitch": {**DEFAULT_PITCH, **dims},
        "camera_pose": {"position_m": [float(v) for v in pose["P"]],
                        "rotation_vector": [float(v) for v in pose["rvec"]],
                        "R": rotmat(pose["rvec"]).tolist()},
        "traced_features": {"labels": sorted(used), "points_per_label": used,
                            "label_parity": label_parity, "dropped_labels": dropped},
        "quality": {"med_px": med_px, "trim90_px": trim90_px, "rms_px": rms_px,
                    "med_m": med_m, "rms_m": rms_m,
                    "n_points": int(len(allres)), "valid": valid,
                    "valid_threshold_px": px_thresh,
                    "valid_threshold_m": RMS_VALID_M,
                    "label_med_px": {l: v[1] for l, v in (px or {"per": {}})["per"].items()},
                    "suspect_labels": suspect, "corners_in_frame": in_frame,
                    "span_at_bound": span_at_bound},
    }
    if verbose:
        print(f"calibration: {len(allres)} pts from {sorted(used)}"
              + (f"  (dropped {dropped})" if dropped else ""))
        print(f"  pose P={np.round(pose['P'], 2)}  span={span:.1f} vspan={vspan_used:.1f} deg  "
              f"median_px={med_px:.2f} / med_m={med_m:.3f} / corners={in_frame}  "
              f"-> calibration_valid={valid}")
    return out


# ---- Phase C public API --------------------------------------------------- #
def make_mappers(pose, W, H, span=180.0, vspan=180.0):
    """(equirect_to_pitch, pitch_to_equirect) bound to a pose.

    vspan MUST match the calibration's vspan_deg: the two maps have to agree on
    the vertical span or every metre is scaled by the wrong factor."""
    def e2p(x_eq, y_eq):
        return equirect_to_pitch(x_eq, y_eq, pose, W, H, span, vspan)
    def p2e(x_m, z_m):
        return pitch_to_equirect(x_m, z_m, pose, W, H, span, vspan)
    return e2p, p2e


def equirect_to_ground(x_eq, y_eq, pose, W, H, span, vspan=180.0):
    """equirect pixel -> (x_m, z_m, valid) on the ground plane (y = 0).

    valid is False when the ray does not meet the ground (at or above the
    horizon). equirect_to_pitch clamps such rays to a far but finite point, so
    callers that need a real ground coordinate must use this and check `valid`.
    """
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    d = equirect_dir(x_eq, y_eq, W, H, span, vspan) @ R.T
    dy = d[..., 1]
    valid = dy < -1e-6
    t = -P[1] / np.where(valid, dy, -1.0)
    g = P + t[..., None] * d
    return g[..., 0], g[..., 2], valid


def pitch_contains(x_eq, y_eq, pose, W, H, span, dims=None, vspan=180.0,
                   margin_m=0.0):
    """Is this equirect pixel's ground projection inside the pitch rectangle?

    Above-horizon / off-ray pixels are False, so this doubles as a metric
    validity gate. `margin_m` grows the rectangle (players standing just off the
    touchline still count as on-pitch).
    """
    d = dict(DEFAULT_PITCH)
    d.update(dims or {})
    hl = d["length"] / 2.0 + margin_m
    hw = d["width"] / 2.0 + margin_m
    x_m, z_m, ok = equirect_to_ground(x_eq, y_eq, pose, W, H, span, vspan)
    return ok & (np.abs(x_m) <= hl) & (np.abs(z_m) <= hw)


# ---- Phase B synthetic self-test ------------------------------------------ #
def synthetic_test(W=3401, H=1586, span=180.0, n_per_line=40, verbose=True,
                   noise_px=0.0, seed=0, vspan_true=180.0, fit_vspan=True):
    import time
    rng = np.random.default_rng(seed)
    P_true = np.array([26.0, 9.5, -58.0])
    R_true = rotmat(np.radians([18.0, 62.0, 7.0]))
    geom = pitch_geometry()

    traces = []
    for lab, spec in geom.items():
        pts = _sample_spec(spec)
        idx = rng.choice(len(pts), size=min(n_per_line, len(pts)), replace=False)
        xs, ys = pitch_to_equirect(pts[idx, 0], pts[idx, 1],
                                   {"P": P_true, "rvec": rvec_from_matrix(R_true)},
                                   W, H, span, vspan_true)
        for x, y in zip(np.ravel(xs), np.ravel(ys)):
            if np.isfinite(x) and 0 <= x < W and 0 <= y < H:
                if noise_px:
                    x, y = x + rng.normal(0, noise_px), y + rng.normal(0, noise_px)
                traces.append((float(x), float(y), lab))
    if verbose:
        print(f"synthetic: {len(traces)} points, true vspan {vspan_true:.0f} deg")

    t0 = time.time()
    sol = solve_pose(traces, W, H, span, geom=geom, fit_vspan=fit_vspan,
                     vspan0=180.0 if vspan_true == 180 else vspan_true)
    dt = time.time() - t0
    P_rec = np.array(sol["P"]); R_rec = rotmat(sol["rvec"])
    pos_err = float(np.linalg.norm(P_rec - P_true))
    ang_err = rot_angle(R_rec, R_true)
    vs_err = abs(sol["vspan"] - vspan_true)
    ok = pos_err < 0.01 * PITCH_LEN and ang_err < 0.5
    if fit_vspan:
        ok = ok and vs_err < 5.0
    med_px = _median_px(traces, {"P": np.array(sol["P"]), "rvec": np.array(sol["rvec"])},
                        geom, W, H, span, sol["vspan"])
    if verbose:
        print(f"  true P {np.round(P_true,2)}  rec P {np.round(P_rec,2)}")
        print(f"  pos err {pos_err:.3f} m ({pos_err/PITCH_LEN*100:.3f}%)  "
              f"orient err {ang_err:.3f} deg  vspan {sol['vspan']:.1f} "
              f"(err {vs_err:.1f} deg)")
        print(f"  reprojection RMS {sol['rms_px']:.2f} px, median {med_px:.2f} px   "
              f"runtime {dt:.2f} s   {'PASS' if ok else 'FAIL'}")
    return {"pos_err_m": pos_err, "ang_err_deg": ang_err, "vspan": sol["vspan"],
            "vspan_err_deg": vs_err, "rms_px": sol["rms_px"], "median_px": med_px,
            "runtime_s": dt, "pass": ok}


def _median_px(traces, pose, geom, W, H, span, vspan=180.0):
    errs = []
    for x, y, lab in traces:
        if lab not in geom:
            continue
        gx, gz = equirect_to_pitch(x, y, pose, W, H, span, vspan)
        gx, gz = float(np.ravel(gx)[0]), float(np.ravel(gz)[0])
        if not (np.isfinite(gx) and np.isfinite(gz)):
            continue
        q = nearest_on((gx, gz), geom[lab])
        ex, ey = pitch_to_equirect(q[0], q[1], pose, W, H, span, vspan)
        errs.append(np.hypot(float(np.ravel(ex)[0]) - x, float(np.ravel(ey)[0]) - y))
    return float(np.median(errs)) if errs else float("nan")


def _sample_spec(spec):
    if spec[0] in ("line", "segment"):
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        t = np.linspace(0, 1, 80)[:, None]
        return a + t * (b - a)
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), spec[2]
        a = np.linspace(0, 2 * np.pi, 80)
        return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
    return np.asarray(spec[1], float)


# ---- Phase C metric round-trip test --------------------------------------- #
def metric_test(W=3401, H=1586, span=180.0, noise_px=2.0, seed=0, verbose=True):
    """(a) pure round-trip with one pose (verifies the maps are true inverses);
    (b) end-to-end: true pitch point -> equirect via TRUE pose -> back via
    SOLVED pose -> error in metres (measures calibration accuracy)."""
    rng = np.random.default_rng(seed)
    P_true = np.array([26.0, 9.5, -58.0])
    R_true = rotmat(np.radians([18.0, 62.0, 7.0]))
    geom = pitch_geometry()
    traces = []
    for lab, spec in geom.items():
        pts = _sample_spec(spec)
        idx = rng.choice(len(pts), size=min(40, len(pts)), replace=False)
        xs, ys = pitch_to_equirect(pts[idx, 0], pts[idx, 1],
                                   {"P": P_true, "rvec": rvec_from_matrix(R_true)}, W, H, span)
        for x, y in zip(np.ravel(xs), np.ravel(ys)):
            if np.isfinite(x) and 0 <= x < W and 0 <= y < H:
                traces.append((float(x + rng.normal(0, noise_px)),
                               float(y + rng.normal(0, noise_px)), lab))
    sol = solve_pose(traces, W, H, span, geom=geom)
    pose_solved = {"P": np.array(sol["P"]), "rvec": np.array(sol["rvec"])}
    pose_true = {"P": P_true, "rvec": rvec_from_matrix(R_true)}

    pts = {"centre spot": (0, 0), "pen-box corner TR": (36, 20.16),
           "pen-box corner BR": (36, -20.16), "pen-box corner TL": (-36, 20.16),
           "halfway @ touchline_far": (0, -34)}
    if verbose:
        print(f"metric round-trip (noise {noise_px} px, solved pose):")
    worst = 0.0
    for name, (xm, zm) in pts.items():
        # pure round trip, solved pose
        ex, ey = pitch_to_equirect(xm, zm, pose_solved, W, H, span)
        rx, rz = equirect_to_pitch(ex, ey, pose_solved, W, H, span)
        self_err = float(np.hypot(np.ravel(rx)[0] - xm, np.ravel(rz)[0] - zm))
        # end-to-end: true pose projects, solved pose maps back
        ex2, ey2 = pitch_to_equirect(xm, zm, pose_true, W, H, span)
        gx, gz = equirect_to_pitch(ex2, ey2, pose_solved, W, H, span)
        e2e = float(np.hypot(np.ravel(gx)[0] - xm, np.ravel(gz)[0] - zm))
        worst = max(worst, e2e)
        if verbose:
            print(f"  {name:24s} round-trip {self_err:.4f} m   end-to-end {e2e:.3f} m")
    if verbose:
        print(f"  worst end-to-end error {worst:.3f} m  -> "
              f"{'PASS (<1m)' if worst < 1.0 else 'FAIL'}")
    return {"worst_m": worst, "pass": worst < 1.0}


if __name__ == "__main__":
    print("== Phase B clean ==")
    synthetic_test()
    print("\n== Phase B with trace noise (imprecise clicks) ==")
    for sig in (1.0, 2.0, 5.0):
        print(f"-- sigma {sig} px --")
        synthetic_test(noise_px=sig)
    print("\n== vertical-span fitting (non-180 content), 2px noise ==")
    for vs in (180.0, 120.0, 90.0, 60.0):
        print(f"-- true vspan {vs:.0f} deg --")
        synthetic_test(vspan_true=vs, noise_px=2.0, fit_vspan=True)
    print("\n== Phase C metric round-trip ==")
    metric_test(noise_px=2.0)
