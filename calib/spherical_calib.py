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
        "touchline_near": ("line", (-hl, hw), (hl, hw)),
        "touchline_far": ("line", (-hl, -hw), (hl, -hw)),
        "goal_line_left": ("line", (-hl, -hw), (-hl, hw)),
        "goal_line_right": ("line", (hl, -hw), (hl, hw)),
        "halfway": ("line", (0, -hw), (0, hw)),
        "penalty_box_left": ("poly", [(-hl, -ph), (-hl + pd, -ph), (-hl + pd, ph), (-hl, ph)]),
        "penalty_box_right": ("poly", [(hl, -ph), (hl - pd, -ph), (hl - pd, ph), (hl, ph)]),
        "centre_circle": ("circle", (0.0, 0.0), d["circle_r"]),
    }


def _line_dist(g, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return np.abs((g - a) @ n)


def _poly_dist(g, pts):
    pts = np.asarray(pts, float)
    best = None
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ab = b - a
        t = np.clip(((g - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(g - (a + t[:, None] * ab), axis=1)
        best = dd if best is None else np.minimum(best, dd)
    return best


def label_residual(g, spec):
    """g: (N,2) ground points (x,z). -> (N,) min distance in metres to the line."""
    if spec[0] == "line":
        return _line_dist(g, spec[1], spec[2])
    if spec[0] == "circle":
        c, r = np.asarray(spec[1], float), spec[2]
        return np.abs(np.linalg.norm(g - c, axis=1) - r)
    return _poly_dist(g, spec[1])


def nearest_on(g, spec):
    """Nearest point on the known line/circle/outline to a single (x,z) point."""
    g = np.asarray(g, float)
    if spec[0] == "line":
        a, b = np.asarray(spec[1], float), np.asarray(spec[2], float)
        d = b - a
        return a + np.dot(g - a, d) / np.dot(d, d) * d
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
def equirect_dir(x_eq, y_eq, W, H, span):
    x_eq, y_eq = np.asarray(x_eq, float), np.asarray(y_eq, float)
    yaw = np.radians((x_eq / W - 0.5) * span)
    pit = np.radians((0.5 - y_eq / H) * 180.0)
    return np.stack([np.sin(yaw) * np.cos(pit), np.sin(pit), np.cos(yaw) * np.cos(pit)], -1)


def equirect_to_pitch(x_eq, y_eq, pose, W, H, span):
    """equirect pixel -> metric ground (x_m, z_m). Rays above the horizon are
    pushed far away so the solver gets a large, smooth residual."""
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    d = equirect_dir(x_eq, y_eq, W, H, span) @ R.T
    dy = np.minimum(d[..., 1], -1e-3)
    t = -P[1] / dy
    g = P[None, :] + t[..., None] * d
    return g[..., 0], g[..., 2]


def pitch_to_equirect(x_m, z_m, pose, W, H, span):
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
    return (yaw / span + 0.5) * W, (0.5 - pit / 180.0) * H


# ---- solver --------------------------------------------------------------- #
def _group(traces):
    g = {}
    for x, y, lab in traces:
        g.setdefault(lab, []).append((x, y))
    return {k: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
            for k, v in g.items()}


def _rms_px(traces, pose, geom, W, H, span):
    errs = []
    for x, y, lab in traces:
        if lab not in geom:
            continue
        gx, gz = equirect_to_pitch(x, y, pose, W, H, span)
        gx, gz = float(np.ravel(gx)[0]), float(np.ravel(gz)[0])
        if not (np.isfinite(gx) and np.isfinite(gz)):
            continue
        q = nearest_on((gx, gz), geom[lab])
        ex, ey = pitch_to_equirect(q[0], q[1], pose, W, H, span)
        errs.append(np.hypot(float(np.ravel(ex)[0]) - x, float(np.ravel(ey)[0]) - y))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def solve_pose(traces, W, H, span=180.0, geom=None, seed_inits=None, max_nfev=400):
    """Fit (P, rvec) so traced equirect curves land on the known pitch lines."""
    geom = geom or pitch_geometry()
    groups = _group([(float(x), float(y), l) for x, y, l in traces if l in geom])

    def resid(params):
        pose = {"P": params[:3], "rvec": params[3:6]}
        out = []
        for lab, (xs, ys) in groups.items():
            gx, gz = equirect_to_pitch(xs, ys, pose, W, H, span)
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
    best = None
    for x0 in seed_inits:
        try:
            r = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), method="trf",
                              max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
        except Exception:
            continue
        c = float(np.sum(r.fun ** 2))
        if best is None or c < best[0]:
            best = (c, r.x.copy())
    if best is None:
        raise RuntimeError("solver failed on all inits")
    cost, p = best
    pose = {"P": p[:3], "rvec": p[3:6]}
    res = resid(p)
    n = sum(len(v[0]) for v in groups.values())
    return {"P": list(p[:3]), "rvec": list(p[3:6]), "R": rotmat(p[3:6]).tolist(),
            "cost": cost, "rms_m": float(np.sqrt(cost / max(n, 1))),
            "med_m": float(np.median(res)),
            "rms_px": _rms_px(traces, pose, geom, W, H, span),
            "n": int(n), "labels": sorted(groups)}


# ---- calibration contract (from the pencil-tool trace file) --------------- #
CALIB_VERSION = 1
RMS_VALID_M = 1.0          # metric RMS below this -> calibration_valid


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
    for ln in data.get("lines", []):
        lab = ln.get("label")
        if lab not in geom:
            continue
        for pt in ln.get("points", []):
            traces.append((float(pt[0]), float(pt[1]), lab))
            used[lab] = used.get(lab, 0) + 1
    if not traces:
        raise RuntimeError("no usable traces (labels must match the known pitch features)")

    sol = solve_pose(traces, W, H, span, geom=geom)
    # validity on the MEDIAN residual: the RMS is dominated by near-horizon traces
    # where a pixel maps to metres of ground, which says nothing about fit quality.
    valid = bool(sol["med_m"] < RMS_VALID_M and sol["n"] >= 12)
    out = {
        "version": CALIB_VERSION,
        "projection": "equirect",
        "span_deg": span,
        "image_size": [int(W), int(H)],
        "content_region": data.get("crop"),
        "source_image": data.get("image"),
        "pitch": {**DEFAULT_PITCH, **dims},
        "camera_pose": {"position_m": sol["P"], "rotation_vector": sol["rvec"],
                        "R": sol["R"]},
        "traced_features": {"labels": sorted(used), "points_per_label": used},
        "quality": {"med_m": sol["med_m"], "rms_m": sol["rms_m"], "rms_px": sol["rms_px"],
                    "n_points": sol["n"], "valid": valid,
                    "valid_threshold_m": RMS_VALID_M},
    }
    if verbose:
        print(f"calibration: {sol['n']} pts from {sorted(used)}")
        print(f"  pose P={np.round(sol['P'], 2)}  median {sol['med_m']:.3f} m / "
              f"rms {sol['rms_m']:.3f} m / {sol['rms_px']:.2f} px  -> "
              f"calibration_valid={valid}")
    return out


# ---- Phase C public API --------------------------------------------------- #
def make_mappers(pose, W, H, span=180.0):
    """(equirect_to_pitch, pitch_to_equirect) bound to a pose."""
    def e2p(x_eq, y_eq):
        return equirect_to_pitch(x_eq, y_eq, pose, W, H, span)
    def p2e(x_m, z_m):
        return pitch_to_equirect(x_m, z_m, pose, W, H, span)
    return e2p, p2e


# ---- Phase B synthetic self-test ------------------------------------------ #
def synthetic_test(W=3401, H=1586, span=180.0, n_per_line=40, verbose=True,
                   noise_px=0.0, seed=0):
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
                                   W, H, span)
        for x, y in zip(np.ravel(xs), np.ravel(ys)):
            if np.isfinite(x) and 0 <= x < W and 0 <= y < H:
                if noise_px:
                    x, y = x + rng.normal(0, noise_px), y + rng.normal(0, noise_px)
                traces.append((float(x), float(y), lab))
    if verbose:
        print(f"synthetic: {len(traces)} points from {len(set(t[2] for t in traces))} lines")

    t0 = time.time()
    sol = solve_pose(traces, W, H, span, geom=geom)
    dt = time.time() - t0
    P_rec = np.array(sol["P"]); R_rec = rotmat(sol["rvec"])
    pos_err = float(np.linalg.norm(P_rec - P_true))
    ang_err = rot_angle(R_rec, R_true)
    ok = pos_err < 0.01 * PITCH_LEN and ang_err < 0.5
    med_px = _median_px(traces, {"P": np.array(sol["P"]), "rvec": np.array(sol["rvec"])},
                        geom, W, H, span)
    if verbose:
        print(f"  true P {np.round(P_true,2)}  rec P {np.round(P_rec,2)}")
        print(f"  position error {pos_err:.3f} m ({pos_err/PITCH_LEN*100:.3f}% of length)"
              f"   orientation error {ang_err:.3f} deg")
        print(f"  reprojection: RMS {sol['rms_px']:.2f} px, median {med_px:.2f} px "
              f"(RMS inflated by near-horizon traces)   runtime {dt:.2f} s   "
              f"{'PASS' if ok else 'FAIL'}")
    return {"pos_err_m": pos_err, "ang_err_deg": ang_err, "rms_px": sol["rms_px"],
            "median_px": med_px, "runtime_s": dt, "pass": ok}


def _median_px(traces, pose, geom, W, H, span):
    errs = []
    for x, y, lab in traces:
        if lab not in geom:
            continue
        gx, gz = equirect_to_pitch(x, y, pose, W, H, span)
        gx, gz = float(np.ravel(gx)[0]), float(np.ravel(gz)[0])
        if not (np.isfinite(gx) and np.isfinite(gz)):
            continue
        q = nearest_on((gx, gz), geom[lab])
        ex, ey = pitch_to_equirect(q[0], q[1], pose, W, H, span)
        errs.append(np.hypot(float(np.ravel(ex)[0]) - x, float(np.ravel(ey)[0]) - y))
    return float(np.median(errs)) if errs else float("nan")


def _sample_spec(spec):
    if spec[0] == "line":
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
    print("\n== Phase C metric round-trip ==")
    metric_test(noise_px=2.0)
