"""Phase D — goal detection from calibration geometry + ball detections.

No new model: the goal is a vertical plane at the goal line, and the ball's
per-frame ray is intersected with it. "In the mouth" = the intersection lies
within the posts (|z| <= 3.66 m) and under the bar (0 <= y <= 2.44 m). This uses
the ray directly, so it needs NO ground-plane assumption and gives the ball's
height at the goal plane for free.

A goal = the ball transitions from OUTSIDE the mouth to INSIDE it and stays in
for >= `consecutive` sampled frames.
"""

import numpy as np

from spherical_calib import (GOAL_HALF_WIDTH, GOAL_HEIGHT, PITCH_LEN,
                             equirect_dir, equirect_to_pitch, rotmat)


def pitch3d_to_equirect(pts, pose, W, H, span=180.0):
    """3D pitch points (N,3) metres -> equirect pixels (x_eq, y_eq)."""
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    pts = np.asarray(pts, float).reshape(-1, 3)
    d_p = pts - P[None, :]
    d_p = d_p / np.linalg.norm(d_p, axis=-1, keepdims=True)
    d_e = d_p @ R
    yaw = np.degrees(np.arctan2(d_e[..., 0], d_e[..., 2]))
    pit = np.degrees(np.arcsin(np.clip(d_e[..., 1], -1, 1)))
    return (yaw / span + 0.5) * W, (0.5 - pit / 180.0) * H


def goal_line_x(side, length=PITCH_LEN):
    return (-length / 2) if side == "left" else (length / 2)


def goal_frame_3d(side, length=PITCH_LEN):
    """4 goal-frame corners (post bases + post tops) in metres."""
    x = goal_line_x(side, length)
    hw, h = GOAL_HALF_WIDTH, GOAL_HEIGHT
    return np.array([[x, 0, -hw], [x, 0, hw], [x, h, -hw], [x, h, hw]], float)


def goal_frame_equirect(pose, side, W, H, span=180.0):
    """Project the goal frame into equirect pixels: bases, tops, crossbar."""
    base_l, base_r, top_l, top_r = goal_frame_3d(side)
    b_l = pitch3d_to_equirect(base_l, pose, W, H, span)
    b_r = pitch3d_to_equirect(base_r, pose, W, H, span)
    t_l = pitch3d_to_equirect(top_l, pose, W, H, span)
    t_r = pitch3d_to_equirect(top_r, pose, W, H, span)
    cross = []
    for f in np.linspace(0, 1, 24):
        p = (1 - f) * np.array([goal_line_x(side), GOAL_HEIGHT, -GOAL_HALF_WIDTH]) \
            + f * np.array([goal_line_x(side), GOAL_HEIGHT, GOAL_HALF_WIDTH])
        cross.append(pitch3d_to_equirect(p, pose, W, H, span))
    return {"base_left": b_l, "base_right": b_r, "top_left": t_l, "top_right": t_r,
            "crossbar": np.array(cross)}


def ray_plane_point(x_eq, y_eq, pose, plane_x, W, H, span=180.0):
    """Intersect the pixel's ray with the vertical plane x = plane_x.

    Returns (y_m, z_m) on the plane, or (nan, nan) if the ray is parallel or
    the plane is behind the camera."""
    P = np.asarray(pose["P"], float)
    R = rotmat(pose["rvec"])
    d = equirect_dir(x_eq, y_eq, W, H, span) @ R.T
    dx = d[..., 0]
    ok = np.abs(dx) > 1e-9
    t = np.where(ok, (plane_x - P[0]) / np.where(ok, dx, 1.0), np.nan)
    t = np.where(t > 1e-6, t, np.nan)
    y = P[1] + t * d[..., 1]
    z = P[2] + t * d[..., 2]
    return y, z


def in_mouth(y, z, hw=GOAL_HALF_WIDTH, h=GOAL_HEIGHT):
    return (np.abs(z) <= hw) & (y >= 0.0) & (y <= h)


def detect_goals(trail, pose, side, W, H, span=180.0, consecutive=3):
    """trail: list of (x_eq, y_eq, frame_idx, conf) in equirect pixels (any order).

    Returns a list of events: dicts with the run's start/end frame, peak conf,
    the mean (y,z) at the goal plane, and the ball's equirect position.
    """
    px = goal_line_x(side)
    seq = sorted(trail, key=lambda t: t[2])
    states = []
    for x, y_, f, c in seq:
        # crossing + width from the GROUND position (valid for a grounded ball)
        xm, zm = equirect_to_pitch(x, y_, pose, W, H, span)
        xm, zm = float(np.ravel(xm)[0]), float(np.ravel(zm)[0])
        near = np.isfinite(xm) and abs(xm - px) <= 3.0
        wide = (not np.isfinite(zm)) or abs(zm) > GOAL_HALF_WIDTH
        # height from the ray's intersection with the goal plane (valid at the plane)
        yapp, zapp = ray_plane_point(x, y_, pose, px, W, H, span)
        yapp = float(np.ravel(yapp)[0])
        high = (not np.isfinite(yapp)) or not (0.0 <= yapp <= GOAL_HEIGHT)
        inside = near and not wide and not high
        states.append((f, bool(inside), yapp, zm, float(c), x, y_))

    events, run = [], []
    for st in states:
        if st[1]:
            run.append(st)
        elif run:
            if len(run) >= consecutive:
                events.append(_event(run))
            run = []
    if len(run) >= consecutive:
        events.append(_event(run))
    return events


def _event(run):
    ys = np.array([r[2] for r in run]); zs = np.array([r[3] for r in run])
    cf = np.array([r[4] for r in run])
    return {"frame_start": run[0][0], "frame_end": run[-1][0], "n_frames": len(run),
            "mean_y_m": float(ys.mean()), "mean_z_m": float(zs.mean()),
            "max_conf": float(cf.max()),
            "ball_eq": (run[len(run) // 2][5], run[len(run) // 2][6])}


# ---- synthetic self-test -------------------------------------------------- #
def _trail_from_3d(pts, pose, W, H, span, frames, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = pitch3d_to_equirect(np.asarray(pts, float), pose, W, H, span)
    out = []
    for x, y, f in zip(np.ravel(xs), np.ravel(ys), frames):
        if np.isfinite(x) and 0 <= x < W and 0 <= y < H:
            out.append((float(x + rng.normal(0, noise)), float(y + rng.normal(0, noise)),
                        int(f), 0.3))
    return out


def synthetic_test(W=3401, H=1586, span=180.0, verbose=True):
    from spherical_calib import rotmat as rm
    P = np.array([26.0, 9.5, -58.0])
    pose = {"P": P, "rvec": np.radians([18.0, 62.0, 7.0])}
    side = "right"
    xg = goal_line_x(side)
    f = np.arange(0, 40)

    cases = {}
    # 1) rolling into the goal mouth (z=0, on the ground)
    cases["roll-in (z=0)"] = np.stack([np.linspace(xg - 12, xg + 6, len(f)),
                                       np.zeros(len(f)), np.zeros(len(f))], 1)
    # 2) wide: crosses the line but outside the posts (z=8)
    cases["wide (z=8m)"] = np.stack([np.linspace(xg - 12, xg + 6, len(f)),
                                     np.zeros(len(f)), np.full(len(f), 8.0)], 1)
    # 3) over the bar: airborne at 3.5 m when crossing
    cases["over bar (y=3.5m)"] = np.stack([np.linspace(xg - 12, xg + 6, len(f)),
                                           np.full(len(f), 3.5), np.zeros(len(f))], 1)
    # 4) far away on the pitch, never near the goal
    cases["far away"] = np.stack([np.linspace(-10, 5, len(f)), np.zeros(len(f)),
                                  np.zeros(len(f))], 1)

    if verbose:
        print("Phase D synthetic goal detection (right goal, 3.66m half-width, 2.44m bar):")
    results = {}
    for name, pts in cases.items():
        trail = _trail_from_3d(pts, pose, W, H, span, f)
        ev = detect_goals(trail, pose, side, W, H, span, consecutive=3)
        results[name] = len(ev)
        if verbose:
            detail = ""
            if ev:
                e = ev[0]
                detail = (f"  frames {e['frame_start']}-{e['frame_end']} "
                          f"(n={e['n_frames']}, y={e['mean_y_m']:.2f}m z={e['mean_z_m']:.2f}m)")
            print(f"  {name:18s} -> {len(ev)} goal event(s){detail}")
    want = {"roll-in (z=0)": True, "wide (z=8m)": False,
            "over bar (y=3.5m)": False, "far away": False}
    ok = all((results[k] > 0) == want[k] for k in want)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'}  (expected: roll-in yes, others no)")
    return {"results": results, "pass": ok}


if __name__ == "__main__":
    synthetic_test()
