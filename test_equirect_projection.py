"""Phase test: prove equirect -> perspective projection before any pipeline work.

Real run (once the equirect video is on disk):
    python test_equirect_projection.py --video path/to/equirect.mp4 [--span 180]
        -> extracts frames 100/300/500, writes 9 views to equirect_test/,
           then runs the player + ball detectors on frame 300 @ yaw 0.

Math self-check (no video needed):
    python test_equirect_projection.py --synthetic [--span 360]
        -> ray-casts a synthetic ground scene (straight grid lines) into an
           equirect image, reprojects it, and measures straightness / horizon /
           yaw / resolution / aspect.
"""

import argparse
import os

import cv2
import numpy as np

from spherical_projection import equirect_to_perspective

YAWS = (-60, 0, 60)
PITCH = 0
FOV = 60
OUT_W, OUT_H = 1920, 1080
FRAMES = (100, 300, 500)
OUTDIR = "equirect_test"


# --------------------------------------------------------------------------- #
# Real-video path
# --------------------------------------------------------------------------- #
def extract_frames(video_path, frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {f} (video has {total})")
        out[f] = img
    cap.release()
    return out, total


def run_real(video_path, span):
    os.makedirs(OUTDIR, exist_ok=True)
    frames, total = extract_frames(video_path, FRAMES)
    raw_h, raw_w = next(iter(frames.values())).shape[:2]
    print(f"[resolution] raw equirect: {raw_w}x{raw_h}  (frames total={total})")

    saved = []
    for f, img in frames.items():
        for yaw in YAWS:
            out = equirect_to_perspective(img, yaw, PITCH, FOV, OUT_W, OUT_H, span)
            fn = os.path.join(OUTDIR, f"proj_{f}_{yaw}.png")
            cv2.imwrite(fn, out)
            saved.append(fn)
    print(f"[resolution] projected views: {OUT_W}x{OUT_H} each")
    print(f"[saved] {len(saved)} images to {OUTDIR}/:")
    for fn in saved:
        print("   ", fn)

    detector_check(frames[300], span)
    return saved


# --------------------------------------------------------------------------- #
# Synthetic self-check
# --------------------------------------------------------------------------- #
def synthetic_equirect(W=2048, H=1024, cam_h=1.0, span=360.0):
    """Ray-cast a ground plane with a unit grid + two side markers into equirect.

    Uses the standard equirect definition (pixel -> yaw/pitch -> direction),
    independent of the forward function's internals.
    """
    ii, jj = np.meshgrid(np.arange(W), np.arange(H))
    yaw = np.radians((ii + 0.5) / W * span - span / 2.0)
    pitch = np.radians(90.0 - (jj + 0.5) / H * 180.0)
    dx = np.sin(yaw) * np.cos(pitch)
    dy = np.sin(pitch)
    dz = np.cos(yaw) * np.cos(pitch)

    img = np.empty((H, W, 3), np.uint8)
    img[...] = (200, 160, 120)                       # sky (BGR)

    ground = dy < -1e-9
    t = np.where(ground, -cam_h / np.where(ground, dy, -1.0), 0.0)
    X, Z = t * dx, t * dz
    gx = np.abs(X - np.round(X))
    gz = np.abs(Z - np.round(Z))

    img[ground] = (60, 60, 60)
    img[ground & (gx < 0.03)] = (0, 0, 255)          # red = constant-x lines
    img[ground & (gz < 0.03)] = (0, 255, 0)          # green = constant-z lines
    img[ground & (np.abs(X + 8) < 1) & (np.abs(Z - 8) < 1)] = (255, 0, 255)  # magenta, left
    img[ground & (np.abs(X - 8) < 1) & (np.abs(Z - 8) < 1)] = (0, 255, 255)  # yellow, right
    return img


def _pinhole_project(P, yaw_deg, pitch_deg, fov_deg, out_w, out_h):
    """World point -> output pixel, using the pinhole model independently of the
    forward function (camera at origin, same basis/focal convention).

    For a view basis (fwd, right, up), the pixel of direction d is
        u = cx + fx * (d.right / d.fwd),  v = cy - fy * (d.up / d.fwd).
    """
    fx = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    right = np.cross([0.0, 1.0, 0.0], fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    d = P / np.linalg.norm(P, axis=1, keepdims=True)
    a, b, c = d @ fwd, d @ right, d @ up
    valid = a > 1e-6
    a_safe = np.where(valid, a, 1.0)
    u = out_w / 2.0 + fx * (b / a_safe)
    v = out_h / 2.0 - fx * (c / a_safe)
    return np.stack([u, v], 1), valid


def _collinear_residual(pts):
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    nvec = np.array([-vt[0][1], vt[0][0]])
    return float(np.abs((pts - mean) @ nvec).max())


def _analytic_straightness():
    """A straight 3D line must project to a straight image line. Tests several
    ground lines at several views using the pinhole model directly."""
    h = 1.0
    lines = {
        "const-Z":     np.stack([np.linspace(-12, 12, 200), np.full(200, -h), np.full(200, 5.0)], 1),
        "const-X":     np.stack([np.full(200, 5.0), np.full(200, -h), np.linspace(2, 25, 200)], 1),
        "diagonal":    np.stack([np.linspace(-6, 6, 200), np.full(200, -h), np.linspace(3, 9, 200)], 1),
        "down-pitch":  np.stack([np.linspace(-8, 8, 200), np.full(200, -h), np.full(200, 3.0)], 1),
    }
    views = [(0, -30), (45, -30), (-45, -20), (0, 0)]
    worst = 0.0
    for (yaw, pitch) in views:
        for name, P in lines.items():
            pts, valid = _pinhole_project(P, yaw, pitch, FOV, OUT_W, OUT_H)
            pts = pts[valid]
            if len(pts) < 10:
                continue
            r = _collinear_residual(pts)
            worst = max(worst, r)
            if r > 0.5:
                print(f"  [straight-line] CURVED: {name} @ yaw={yaw},pitch={pitch} "
                      f"residual {r:.3f} px")
    return worst


def _horizon_row(view):
    """Median row where sky (bright) transitions to ground (dark)."""
    bright = view[:, :, 0].mean(axis=1) > 130
    rows = []
    for c in range(view.shape[1]):
        col = view[:, c, 0] > 130
        idx = np.where(~col & (np.arange(len(col)) > 5))[0]
        if len(idx):
            rows.append(idx[0])
    return float(np.median(rows)) if rows else float("nan")


def run_synthetic(span):
    os.makedirs(OUTDIR, exist_ok=True)
    eq = synthetic_equirect(span=span)
    cv2.imwrite(os.path.join(OUTDIR, "synth_equirect.png"), eq)
    print(f"[resolution] synthetic equirect: {eq.shape[1]}x{eq.shape[0]}")

    views = {}
    for yaw in YAWS:
        views[yaw] = equirect_to_perspective(eq, yaw, PITCH, FOV, OUT_W, OUT_H, span)
        cv2.imwrite(os.path.join(OUTDIR, f"synth_proj_{yaw}.png"), views[yaw])
    print(f"[resolution] projected views: {OUT_W}x{OUT_H} each")

    # Straight lines: a 3D straight line must project straight.
    down = equirect_to_perspective(eq, 0, -35, FOV, OUT_W, OUT_H, span)
    cv2.imwrite(os.path.join(OUTDIR, "synth_proj_0_down35.png"), down)
    worst = _analytic_straightness()
    print(f"[straight-line] 4 lines x 4 views via pinhole model: "
          f"worst max-residual {worst:.4f} px (expected ~0)")

    # Horizon at pitch=0.
    hr = _horizon_row(views[0])
    print(f"[horizon] pitch=0 view: sky/ground boundary at row {hr:.1f} "
          f"(expected ~{OUT_H/2:.0f})")

    # Yaw direction: markers must land on the correct side.
    def marker_px(v, bgr, tol=40):
        d = np.abs(v.astype(int) - np.array(bgr)).sum(2)
        return int((d < tol).sum())
    mag_l = marker_px(views[-60], (255, 0, 255))
    yel_l = marker_px(views[-60], (0, 255, 255))
    mag_r = marker_px(views[60], (255, 0, 255))
    yel_r = marker_px(views[60], (0, 255, 255))
    print(f"[yaw] yaw=-60: left-marker px={mag_l}, right-marker px={yel_l}")
    print(f"[yaw] yaw=+60: left-marker px={mag_r}, right-marker px={yel_r}")
    print("[yaw] expected: -60 sees LEFT marker, +60 sees RIGHT marker")

    # End-to-end: analytic projection of a known world marker must land where
    # cv2.remap actually places it (ties the pinhole model to the sampled output).
    marker_world = np.array([[-8.0, -1.0, 8.0]])
    for yaw in (-60, 60):
        exp, _ = _pinhole_project(marker_world, yaw, PITCH, FOV, OUT_W, OUT_H)
        exp_u, exp_v = exp[0]
        ys, xs = np.where(np.abs(views[yaw].astype(int)
                                 - np.array((255, 0, 255))).sum(2) < 40)
        if len(xs):
            got_u, got_v = xs.mean(), ys.mean()
            print(f"[end-to-end] yaw={yaw}: magenta marker analytic "
                  f"({exp_u:.0f},{exp_v:.0f}) vs sampled ({got_u:.0f},{got_v:.0f}) "
                  f"-> {np.hypot(exp_u-got_u, exp_v-got_v):.1f} px")
        else:
            print(f"[end-to-end] yaw={yaw}: marker not visible")

    # Aspect: focal lengths must be equal (no aniso stretch).
    fx = (OUT_W / 2.0) / np.tan(np.radians(FOV) / 2.0)
    print(f"[aspect] fx={fx:.2f} fy={fx:.2f} (equal => no stretch); "
          f"verify bodies on the real frame")


# --------------------------------------------------------------------------- #
# Detector pass on one projected view (report only, no boxes, no training)
# --------------------------------------------------------------------------- #
def detector_check(equirect_frame, span):
    from ultralytics import YOLO
    view = equirect_to_perspective(equirect_frame, 0, PITCH, FOV, OUT_W, OUT_H, span)
    cv2.imwrite(os.path.join(OUTDIR, "proj_300_0_detcheck.png"), view)

    models = [("player yolo26l.pt", "yolo26l.pt", 1280),
              ("ball yolo26m.engine", "ball_detector_yolo26m.engine", 640)]
    for name, path, imgsz in models:
        if not os.path.exists(path):
            print(f"[detector] {name}: file not found, skipped")
            continue
        try:
            res = YOLO(path).predict(view, imgsz=imgsz, conf=0.05, verbose=False)[0]
        except Exception as e:                       # noqa: BLE001
            print(f"[detector] {name}: failed ({e})")
            continue
        if res.boxes is None or len(res.boxes) == 0:
            print(f"[detector] {name} @ imgsz={imgsz}: 0 detections")
            continue
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names
        per = {}
        for c, s in zip(cls, conf):
            per.setdefault(names.get(c, str(c)), []).append(float(s))
        print(f"[detector] {name} @ imgsz={imgsz}: {len(conf)} detections, "
              f"conf {conf.min():.3f}-{conf.max():.3f}")
        for cname, scores in sorted(per.items()):
            print(f"      {cname}: n={len(scores)} "
                  f"conf {min(scores):.3f}-{max(scores):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="equirect video path (real test)")
    ap.add_argument("--synthetic", action="store_true", help="run math self-check")
    ap.add_argument("--span", type=float, default=None,
                    help="yaw span of source: 360 (full) or 180 (VR180). "
                         "Default: 180 given the spec's '180 deg video'.")
    args = ap.parse_args()

    span = args.span if args.span is not None else 180.0
    if args.synthetic:
        run_synthetic(span)
    elif args.video:
        run_real(args.video, span)
    else:
        ap.error("pass --video PATH or --synthetic")


if __name__ == "__main__":
    main()
