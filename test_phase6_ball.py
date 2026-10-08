"""Phase 6 — ball detector on equirect tiles (diagnostic only, no changes).

Runs ball_detector_yolo26m.engine (TensorRT, 1024x1024) on the FOV-60 grid for
frames 70/85/100 s, reports counts/confidence/scale, checks far-side tiles, and
(6.6) runs ByteTrack over 30 sampled frames of merged player detections.

    python test_phase6_ball.py
"""

import os

import cv2
import numpy as np
import supervision as sv

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, tile_pixel_to_equirect,
)
from player_tracker import PlayerTracker
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"
BALL_CONF = 0.06
FAR_PITCH = -20.0
TIMES = (70.0, 85.0, 100.0)


def grab(t):
    sc = load_sidecar(V) or {}
    cap = cv2.VideoCapture(V)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, raw = cap.read(); cap.release()
    assert ok, f"frame {t}s"
    return apply_region(raw, sc.get("crop")), sc


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ball = YOLO("ball_detector_yolo26m.engine")
    grid = build_grid(180, 60.0)
    saved = 0

    print("=" * 68)
    print("6.1/6.2/6.3  ball detector on FOV-60 tiles")
    for t in TIMES:
        frame, sc = grab(t)
        H, W = frame.shape[:2]
        tiles = tile_equirect(frame, grid, 180)
        dets = []
        for img, yaw, pitch, fov, tid in tiles:
            r = ball.predict(img, imgsz=1024, conf=BALL_CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = (float(v) for v in b)
                uc, vc = (x1 + x2) / 2, (y1 + y2) / 2
                ex, ey = tile_pixel_to_equirect(uc, vc, yaw, pitch, fov, 180, W, H)
                dets.append({"t": t, "tile_id": tid, "yaw": yaw, "pitch": pitch,
                             "conf": float(s), "w": x2 - x1, "h": y2 - y1,
                             "eq": (ex, ey),
                             "eq_pitch": (0.5 - ey / H) * 180.0})
            if saved < 6:
                cv2.imwrite(f"{OUTDIR}/p6_ball_t{t:.0f}_tile{tid}.png", img)
                saved += 1
        print(f"\n  frame {t:.0f}s  ({len(tiles)} tiles): {len(dets)} ball detections")
        per = {}
        for d in dets:
            per[d["tile_id"]] = per.get(d["tile_id"], 0) + 1
        if per:
            print("    per-tile:", " ".join(f"t{k}={v}" for k, v in sorted(per.items())))
            cf = np.array([d["conf"] for d in dets])
            wh = np.array([max(d["w"], d["h"]) for d in dets])     # tile-space scale
            print(f"    conf  min {cf.min():.3f} median {np.median(cf):.3f} max {cf.max():.3f}")
            print(f"    tile-space ball size (max dim): {wh.min():.1f}-{wh.max():.1f} px "
                  f"(-> model px x0.533 = {wh.min()*0.533:.1f}-{wh.max()*0.533:.1f})")
            far = [d for d in dets if d["eq_pitch"] > FAR_PITCH]
            print(f"    far-side (eq pitch > {FAR_PITCH:.0f}): {len(far)} detections"
                  + (f" confs {sorted(round(d['conf'],2) for d in far)}" if far else ""))
            for d in sorted(dets, key=lambda z: -z["conf"])[:5]:
                print(f"      conf {d['conf']:.2f} tile {d['tile_id']} eq_pitch "
                      f"{d['eq_pitch']:+.0f} size {d['w']:.1f}x{d['h']:.1f}px")
        else:
            print("    none")

    # ---- 6.6 ByteTrack over 30 sampled frames of merged player detections ----
    print("\n" + "=" * 68)
    print("6.6  ByteTrack on merged player detections (30 frames, 70-100s)")
    pl = YOLO("yolo26l.pt")
    tracker = PlayerTracker(max_missed=90, proximity_px=250,
                            appearance_threshold=0.4, match_distance_weight=0.5)
    sc = load_sidecar(V) or {}
    cap = cv2.VideoCapture(V)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    ids, conc = set(), []
    for k in range(30):
        cap.set(cv2.CAP_PROP_POS_MSEC, (70.0 + k) * 1000.0)     # 1 frame/second
        ok, raw = cap.read()
        if not ok:
            break
        fr = apply_region(raw, sc.get("crop"))
        dets = []
        for img, yaw, pitch, fov, tid in tile_equirect(fr, grid, 180):
            r = pl.predict(img, imgsz=1280, conf=0.10, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        mg, _ = merge_to_equirect(dets, 180, fr.shape[1], fr.shape[0], criterion="center")
        conc.append(len(mg))
        if mg:
            d = sv.Detections(xyxy=np.array([x["bbox"] for x in mg], float),
                              confidence=np.array([x["conf"] for x in mg], float),
                              class_id=np.zeros(len(mg), int))
            tr = tracker.update(d, frame=None)
            if tr is not None and tr.tracker_id is not None:
                ids.update(int(i) for i in tr.tracker_id)
    cap.release()
    print(f"  frames {len(conc)}  merged/frame mean {np.mean(conc):.1f} "
          f"(min {min(conc)} max {max(conc)})")
    print(f"  unique track IDs {len(ids)}  vs max-concurrent {max(conc)}  "
          f"-> ratio {len(ids)/max(max(conc),1):.2f}x")


if __name__ == "__main__":
    main()
