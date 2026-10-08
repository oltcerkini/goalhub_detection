"""Phase 5b — far-band zoom, corrected: ROI-filtered, properly placed rows.

The pitch>0 detections are off the ground plane (above the horizon) -> background
false positives; the grass ROI removes them. The FOV20 rows must also cover the
far band with vertical overlap, else mid-band players land on tile edges.
"""

import os
import time

import cv2
import numpy as np

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, roi_filter, grass_roi_polygon,
)
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"
CONF = 0.10


def main():
    sc = load_sidecar(V) or {}
    cap = cv2.VideoCapture(V)
    cap.set(cv2.CAP_PROP_POS_MSEC, 85 * 1000)
    _, raw = cap.read(); cap.release()
    frame = apply_region(raw, sc.get("crop"))
    H, W = frame.shape[:2]
    roi = np.array(sc["roi"], np.int32) if sc.get("roi") else grass_roi_polygon(frame)
    model = YOLO("yolo26l.pt")

    def pitch_of(y):
        return (0.5 - y / H) * 180.0

    def run(grid, mask_edges=True):
        dets = []
        for img, yaw, pitch, fov, tid in tile_equirect(frame, grid, 180, mask_edges):
            r = model.predict(img, imgsz=1280, conf=CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        mg, _ = merge_to_equirect(dets, 180, W, H, criterion="center")
        return dets, mg

    def fargrid(fov, yaw_step, rows):
        ys = np.arange(-90 + yaw_step / 2, 90, yaw_step)
        g, tid = [], 0
        for p in rows:
            for y in ys:
                g.append((tid, float(y), float(p), float(fov))); tid += 1
        return g

    print("FOV60 baseline, ROI-filtered:")
    d60, m60 = run(build_grid(180, 60.0))
    k60, drop60 = roi_filter(m60, roi)
    p_all = [pitch_of(x["center"][1]) for x in m60]
    p_k = [pitch_of(x["center"][1]) for x in k60]
    print(f"  merged {len(m60)}; dropped by ROI {len(drop60)} "
          f"(their pitches {[round(pitch_of(x['center'][1])) for x in drop60]})")
    far60 = [x for x in k60 if -20 < pitch_of(x["center"][1]) < 0]
    print(f"  merged pitch range {min(p_all):.0f}..{max(p_all):.0f}")
    print(f"  ROI-kept far-third (pitch -20..0): {len(far60)} dets, "
          f"confs {sorted(round(x['conf'],2) for x in far60)}")

    print("\nFOV20 far-band passes (ROI-filtered):")
    for tag, rows, step in (
        ("spec  rows(-5,-15) step20", (-5.0, -15.0), 20.0),
        ("overlap rows(-3,-12,-21) step16", (-3.0, -12.0, -21.0), 16.0),
    ):
        g = fargrid(20.0, step, rows)
        t0 = time.time(); dd, mg = run(g); dt = time.time() - t0
        kept, dropped = roi_filter(mg, roi)
        far = [x for x in kept if -25 < pitch_of(x["center"][1]) < 2]
        newv = [x for x in far if not any(np.hypot(x["center"][0]-f["center"][0],
                                                    x["center"][1]-f["center"][1]) < 35
                                           for f in far60)]
        miss = [f for f in far60 if not any(np.hypot(f["center"][0]-x["center"][0],
                                                      f["center"][1]-x["center"][1]) < 35
                                             for x in far)]
        print(f"\n  [{tag}] {len(g)} tiles, {len(dd)} raw, {len(mg)} merged, "
              f"ROI-dropped {len(dropped)}, {dt:.1f}s")
        if far:
            cf = np.array([x["conf"] for x in far])
            print(f"    far dets {len(far)}  conf p10 {np.percentile(cf,10):.3f} "
                  f"p50 {np.percentile(cf,50):.3f} p90 {np.percentile(cf,90):.3f}")
        print(f"    NEW (not matched by FOV60 far set): {len(newv)}")
        print(f"    FOV60 far dets MISSED by this pass: {len(miss)} / {len(far60)}")


if __name__ == "__main__":
    main()
