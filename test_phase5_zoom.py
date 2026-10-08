"""Phase 5 — centre-distance merge (5.1) + targeted far-band FOV-20 zoom (5.2).

New-module usage only; no existing pipeline module touched.

    python test_phase5_zoom.py [--time 85]
"""

import argparse
import os
import time

import cv2
import numpy as np

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, _same_object, _iou,
)
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"
CONF = 0.10
FAR_PITCH = -20.0


def load_frame(t):
    sc = load_sidecar(V) or {}
    cap = cv2.VideoCapture(V)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    _, raw = cap.read(); cap.release()
    return apply_region(raw, sc.get("crop")), sc


def detect(model, tiles):
    dets = []
    for img, yaw, pitch, fov, tid in tiles:
        r = model.predict(img, imgsz=1280, conf=CONF, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy(),
                           r.boxes.xyxy.cpu().numpy()):
            if c == 0:
                dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                             "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
    return dets


def far_grid(fov=20.0, yaw_step=20.0, rows=(-5.0, -15.0)):
    ys = np.arange(-90.0 + yaw_step / 2, 90.0, yaw_step)
    tiles, tid = [], 0
    for p in rows:
        for y in ys:
            tiles.append((tid, float(y), float(p), float(fov)))
            tid += 1
    return tiles


def pitch_of(y, H):
    return (0.5 - y / H) * 180.0


def match_count(a_dets, b_dets, H, tol=35.0):
    """How many of a_dets have NO counterpart in b_dets (within tol px, far band)."""
    missed = 0
    for a in a_dets:
        if pitch_of(a["center"][1], H) <= FAR_PITCH:
            continue
        if not any(np.hypot(a["center"][0] - b["center"][0],
                            a["center"][1] - b["center"][1]) < tol for b in b_dets):
            missed += 1
    return missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=85.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    frame, sc = load_frame(args.time)
    H, W = frame.shape[:2]
    model = YOLO("yolo26l.pt")

    # ---- 5.1 canonical merge: verify the Phase 4 failed pair now merges ----
    print("=" * 68)
    print("5.1  centre-distance merge criterion")
    a = {"center": (2333.6, 985.9), "bbox": [2333.6 - 37.7 / 2, 985.9 - 61.8 / 2,
                                             2333.6 + 37.7 / 2, 985.9 + 61.8 / 2]}
    b = {"center": (2330.6, 982.5), "bbox": [2330.6 - 74.2 / 2, 982.5 - 68.5 / 2,
                                             2330.6 + 74.2 / 2, 982.5 + 68.5 / 2]}
    print(f"  Phase 4 pair: centres 5.0px apart; IoU {_iou(a['bbox'], b['bbox']):.3f}")
    print(f"  centre criterion merges it? "
          f"{_same_object(a, b, 0.5, 0.5, 'center')}  "
          f"(dist 5.0 <= 0.5*min(37.7,74.2,61.8,68.5)=18.85)")

    dets60 = detect(model, tile_equirect(frame, build_grid(180, 60.0), 180))
    for crit in ("iou", "center"):
        mg, _ = merge_to_equirect(dets60, 180, W, H, criterion=crit)
        print(f"  FOV60 raw {len(dets60)} -> merged [{crit}] {len(mg)}")

    mg60, mp60 = merge_to_equirect(dets60, 180, W, H, criterion="center")
    # residual near-duplicates: different kept detections still close together
    near = [(mg60[i], mg60[j]) for i in range(len(mg60)) for j in range(i + 1, len(mg60))
            if np.hypot(mg60[i]["center"][0] - mg60[j]["center"][0],
                        mg60[i]["center"][1] - mg60[j]["center"][1]) < 0.5 * min(
                            mg60[i]["bbox"][2] - mg60[i]["bbox"][0],
                            mg60[j]["bbox"][2] - mg60[j]["bbox"][0])]
    print(f"  residual near-duplicate pairs after centre merge: {len(near)}")
    vis = frame.copy()
    for x in mg60:
        bb = [int(v) for v in x["bbox"]]
        cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(OUTDIR, "phase5_merged_fov60.png"), vis)

    # ---- 5.2 targeted far-band FOV-20 zoom ----
    print("\n" + "=" * 68)
    print("5.2  far-band FOV-20 zoom (band = pitch > -20)")
    far60 = [x for x in mp60 if pitch_of(x["center"][1], H) > FAR_PITCH]
    print(f"  FOV60 far-band detections (pre-merge): {len(far60)}")
    results = {}
    for yaw_step, tag in ((20.0, "spec 9 cols step20"), (16.0, "safe 12 cols step16")):
        g = far_grid(20.0, yaw_step)
        t0 = time.time()
        dets = detect(model, tile_equirect(frame, g, 180))
        dt = time.time() - t0
        mg, mp = merge_to_equirect(dets, 180, W, H, criterion="center")
        conf = np.array([x["conf"] for x in mg])
        results[tag] = (g, dets, mg, mp, dt)
        far = [x for x in mg if pitch_of(x["center"][1], H) > FAR_PITCH]
        new_vs_60 = match_count(mg, mp60, H)                 # found by zoom, missed by FOV60
        missed_by_zoom = match_count(mp60, mg, H)            # FOV60 found, zoom missed
        print(f"\n  [{tag}] {len(g)} tiles, {len(dets)} raw, {len(mg)} merged, {dt:.1f}s "
              f"({dt/len(g):.3f}s/tile)")
        if len(conf):
            print(f"    conf p10 {np.percentile(conf,10):.3f} p50 {np.percentile(conf,50):.3f} "
                  f"p90 {np.percentile(conf,90):.3f}")
        print(f"    recovered vs FOV60 (no FOV60 match within 35px): {new_vs_60}")
        print(f"    FOV60 dets missed by this zoom pass: {missed_by_zoom}")

    # ---- visual: far band with FOV60 (red) + zoom (green) ----
    g, dets, mg, mp, dt = results["safe 12 cols step16"]
    y0, y1 = int((0.5 - 3 / 180) * H), int((0.5 + 20 / 180) * H)
    band = frame[y0:y1]
    panels = []
    for (x0, x1) in ((0, W // 2), (W // 2, W)):
        c = band[:, x0:x1].copy()
        s = 1500.0 / c.shape[1]
        c = cv2.resize(c, (1500, int((y1 - y0) * s)))
        for d in mg60:
            if pitch_of(d["center"][1], H) > FAR_PITCH and x0 <= d["center"][0] < x1:
                bb = d["bbox"]
                cv2.rectangle(c, (int((bb[0] - x0) * s), int((bb[1] - y0) * s)),
                              (int((bb[2] - x0) * s), int((bb[3] - y0) * s)), (0, 0, 255), 3)
        for d in mg:
            if x0 <= d["center"][0] < x1:
                bb = d["bbox"]
                cv2.rectangle(c, (int((bb[0] - x0) * s), int((bb[1] - y0) * s)),
                              (int((bb[2] - x0) * s), int((bb[3] - y0) * s)), (0, 255, 0), 3)
        panels.append(c)
    cv2.imwrite(os.path.join(OUTDIR, "phase5_farband_compare.png"), np.vstack(panels))
    print(f"\n  far-band compare (red=FOV60, green=FOV20) -> {OUTDIR}/phase5_farband_compare.png")


if __name__ == "__main__":
    main()
