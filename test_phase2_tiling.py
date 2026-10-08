"""Phase 2 runner — tiling, per-tile detection, equirect merge, virtual camera.

Runs on the user's VR180 clip. New modules only; no existing pipeline module is
imported or modified.

    python test_phase2_tiling.py [--video PATH] [--time 85] [--span 180]

Reports (2.4 / 2.6 / 2.7): grid coverage, auto-crop result, per-tile detection
counts and confidence split by pitch row, merge results + duplicate rate, and a
synthetic virtual-camera tracking trace.
"""

import argparse
import os

import cv2
import numpy as np

from spherical_tiling import (
    detect_content_region, apply_region, detect_span, build_grid,
    tile_equirect, merge_to_equirect, tile_pixel_to_equirect, _iou,
    TILE_W, TILE_H, PITCH_CENTRES_180, YAW_CENTRES_180,
)
from virtual_camera import compute_camera_pose, equirect_xy_to_spherical

VIDEO = ("equirect_test/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"


def grab(video, t, crop=None):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read t={t}s from {video}")
    return apply_region(img, crop) if crop else img


def detect_tiles(tiles, conf_thr=0.10, imgsz=1280):
    from ultralytics import YOLO
    model = YOLO("yolo26l.pt")
    dets = []
    per_tile = []
    for img, yaw, pitch, fov, tid in tiles:
        r = model.predict(img, imgsz=imgsz, conf=conf_thr, verbose=False)[0]
        n = 0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for c, s, b in zip(cls, conf, xyxy):
                if c != 0:              # person only
                    continue
                n += 1
                dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                             "bbox": tuple(float(v) for v in b), "conf": float(s), "cls": 0})
        per_tile.append((tid, yaw, pitch, n))
    return dets, per_tile


def report_grid(span):
    print(f"[grid] span={span:.0f}  tile FOV_h={60.0:.0f}deg -> FOV_v~"
          f"{2*np.degrees(np.arctan((TILE_H/2)/((TILE_W/2)/np.tan(np.radians(30))))):.1f}deg  "
          f"tiles={len(build_grid(span))}  ({TILE_W}x{TILE_H} each)")
    print(f"[grid] yaw centres : {YAW_CENTRES_180}  (60deg each -> covers -90..+90)")
    print(f"[grid] pitch rows  : {PITCH_CENTRES_180}  (36deg each -> covers +3..-63)")


def run_merge_report(tile_dets, span, eq_w, eq_h):
    merged, mapped = merge_to_equirect(tile_dets, span, eq_w, eq_h, iou_thr=0.5)

    # Cross-tile duplicate pairs (same player caught by two overlapping tiles).
    cross = 0
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            if mapped[i]["tile_id"] != mapped[j]["tile_id"] and \
               _iou(mapped[i]["bbox"], mapped[j]["bbox"]) >= 0.5:
                cross += 1

    raw = len(tile_dets)
    post = len(merged)
    dup_rate = 0.0 if raw == 0 else 1.0 - post / raw
    print(f"\n[merge] raw tile detections : {raw}")
    print(f"[merge] after equirect NMS  : {post}   (duplicate rate {dup_rate*100:.0f}%)")
    print(f"[merge] cross-tile duplicate pairs (IoU>=0.5, different tiles): {cross}")
    if merged:
        sc = [m["scale"] for m in merged]
        yo = [m["yaw_offset_from_tile"] for m in merged]
        print(f"[merge] tile->equirect scale px ratio: {min(sc):.2f}-{max(sc):.2f}; "
              f"yaw offset from tile centre: {min(yo):.0f}-{max(yo):.0f} deg "
              f"(sec^2 stretch -> size error up to ~{(1/np.cos(np.radians(max(yo)))**2-1)*100:.0f}% "
              f"at tile edge)")
    return merged, mapped


def conf_by_pitch_row(tile_dets):
    print("\n[conf] confidence by pitch row:")
    for p in PITCH_CENTRES_180:
        c = [d["conf"] for d in tile_dets if d["pitch"] == p]
        if c:
            print(f"   pitch {p:+.0f}deg: n={len(c):3d}  conf min {min(c):.3f} "
                  f"median {np.median(c):.3f} max {max(c):.3f}")
        else:
            print(f"   pitch {p:+.0f}deg: n=0")


def draw_merged(frame, merged, path):
    vis = frame.copy()
    for m in merged:
        x1, y1, x2, y2 = (int(v) for v in m["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, f'{m["conf"]:.2f}', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(path, vis)
    print(f"[merge] annotated equirect -> {path}")


def virtual_camera_test(eq_w, eq_h, span):
    print("\n[cam] synthetic smooth trajectory (ball sweeps left->right):")
    yaw = pitch = None
    ys, ps, ts = [], [], []
    dt = 1.0 / 30.0
    py = (0.5 - (-30) / 180.0) * eq_h          # ball at ~-30 deg pitch
    for i in range(120):
        x = 0.12 * eq_w + (0.76 * eq_w) * (i / 119.0)
        y = py + 40 * np.sin(i / 15.0)
        if yaw is None:
            yaw, pitch = 0.0, -30.0
        yaw, pitch = compute_camera_pose((x, y), yaw, pitch, dt, span, eq_w, eq_h)
        ys.append(yaw); ps.append(pitch); ts.append(i)
    dys = np.abs(np.diff(ys)); dps = np.abs(np.diff(ps))
    print(f"   yaw {ys[0]:+.1f} -> {ys[-1]:+.1f} deg | max step {dys.max():.2f} deg "
          f"| within +-{span/2:.0f}? {all(-span/2 <= v <= span/2 for v in ys)}")
    print(f"   pitch {ps[0]:+.1f} -> {ps[-1]:+.1f} deg | max step {dps.max():.2f} deg "
          f"| clamped to [-60,-10]? {all(-60 <= v <= -10 for v in ps)}")

    # Abrupt jump: ball teleports left->right; camera must ease, not snap.
    yaw, pitch = -60.0, -30.0
    ys2 = []
    for i in range(40):
        x = (0.05 * eq_w) if i < 20 else (0.95 * eq_w)
        yaw, pitch = compute_camera_pose((x, py), yaw, pitch, dt, span, eq_w, eq_h)
        ys2.append(yaw)
    print(f"   teleport test: settled yaw {ys2[19]:+.1f} then {ys2[-1]:+.1f} deg, "
          f"max step {np.abs(np.diff(ys2)).max():.2f} deg (no snap)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--time", type=float, default=85.0)
    ap.add_argument("--span", type=float, default=180.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # Raw frame -> auto-crop the letterbox (2.8).
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
    ok, raw = cap.read(); cap.release()
    assert ok, "frame read failed"
    region = detect_content_region(raw)
    print(f"[crop] raw {raw.shape[1]}x{raw.shape[0]} -> content region (x,y,w,h)={region}")
    frame = apply_region(raw, region)
    eq_h, eq_w = frame.shape[:2]

    # Span auto-detect vs override (2.1).
    auto = detect_span(eq_w, eq_h)
    print(f"[span] aspect {eq_w/eq_h:.2f}:1 -> auto-detect {auto:.0f}; using --span {args.span:.0f}")

    report_grid(args.span)

    grid = build_grid(args.span)
    tiles = tile_equirect(frame, grid, args.span)
    cv2.imwrite(os.path.join(OUTDIR, "phase2_tiles_montage.png"),
                np.hstack([t[0] for t in tiles[:5]]))
    print(f"[tiles] built {len(tiles)} views; montage (row1) -> phase2_tiles_montage.png")

    # 2.4 detect + 2.4 report.
    dets, per_tile = detect_tiles(tiles)
    print(f"\n[detect] total person detections across {len(tiles)} tiles: {len(dets)}")
    for tid, yaw, pitch, n in per_tile:
        print(f"   tile {tid:2d}  yaw {yaw:+4.0f} pitch {pitch:+3.0f}: {n} detections")
    conf_by_pitch_row(dets)

    # 2.5 / 2.6 merge + verify.
    merged, mapped = run_merge_report(dets, args.span, eq_w, eq_h)
    draw_merged(frame, merged, os.path.join(OUTDIR, "phase2_merged.png"))

    # 2.7 virtual camera.
    virtual_camera_test(eq_w, eq_h, args.span)


if __name__ == "__main__":
    main()
