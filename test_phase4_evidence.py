"""Phase 4 evidence: what exactly the Phase 3 retraction means.

Distances on a ground plane map monotonically to equirect pitch
(pitch = -atan(cam_h / d)), so pitch is a distance proxy:
  pitch ~ 0    -> far (near horizon)
  pitch << 0   -> near (steeply down)

Prints far-band detection counts for FOV 60 vs 40, the matched-pair
comparison with CORRECT labels, and the failed-merge pair details.
"""

import cv2
import numpy as np

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, _iou,
)
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
FAR_PITCH = -20.0          # pitch above this = far band (near horizon)


def main():
    sc = load_sidecar(V) or {}
    cap = cv2.VideoCapture(V)
    cap.set(cv2.CAP_PROP_POS_MSEC, 85 * 1000)
    _, raw = cap.read(); cap.release()
    f = apply_region(raw, sc.get("crop"))
    H, W = f.shape[:2]
    model = YOLO("yolo26l.pt")

    def run(fov):
        dets = []
        for img, yaw, pitch, fv, tid in tile_equirect(f, build_grid(180, fov), 180):
            r = model.predict(img, imgsz=1280, conf=0.10, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fv,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        mg, mp = merge_to_equirect(dets, 180, W, H, 0.5)
        return dets, mg, mp

    def pitch_of(y):
        return (0.5 - y / H) * 180.0

    results = {}
    for fov in (60.0, 40.0):
        dets, mg, mp = run(fov)
        results[fov] = (dets, mg, mp)
        grid = build_grid(180, fov)
        top = grid[0][2]
        # per-tile raw counts in the top (far) row
        row = [(t[0], t[1], t[2]) for t in grid if t[2] == top]
        counts = {t[0]: sum(1 for d in dets if d["tile_id"] == t[0]) for t in row}
        far_mg = [x for x in mg if pitch_of(x["center"][1]) > FAR_PITCH]
        allp = [pitch_of(x["center"][1]) for x in mg]
        print(f"--- FOV {fov:.0f} --- top row pitch {top:+.0f}deg, {len(row)} tiles")
        print(f"  raw dets total {len(dets)} | merged {len(mg)}")
        print(f"  top-row raw per tile: " +
              " ".join(f"t{tid}(yaw{y:+.0f})={counts[tid]}" for tid, y, _ in row))
        print(f"  merged in FAR band (pitch>{FAR_PITCH:.0f}): {len(far_mg)} "
              f"confs {sorted(round(x['conf'], 2) for x in far_mg)}")
        print(f"  merged pitch range {min(allp):.0f}..{max(allp):.0f}deg")

    # Matched pairs, correct labels (FOV60 pitch as the reference).
    dets60, mg60, mp60 = results[60.0]
    _, mg40, mp40 = results[40.0]
    buckets = {"far  (pitch>-20)": [], "mid  (-35..-20)": [], "near (pitch<-35)": []}
    unmatched = 0
    for a in mp60:
        p = pitch_of(a["center"][1])
        best, bd = None, 140.0
        for b in mp40:
            d = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
            if d < bd:
                best, bd = b, d
        if best is None:
            unmatched += 1
            continue
        key = "far  (pitch>-20)" if p > -20 else "mid  (-35..-20)" if p > -35 else "near (pitch<-35)"
        buckets[key].append((a["conf"], best["conf"]))
    print(f"\n[matched pairs by TRUE distance]  (FOV60 dets unmatched in FOV40: {unmatched})")
    for k, v in buckets.items():
        if not v:
            print(f"  {k}: none")
            continue
        c60 = np.array([x[0] for x in v]); c40 = np.array([x[1] for x in v])
        print(f"  {k}: n={len(v):2d}  conf60 p50 {np.percentile(c60,50):.3f} -> "
              f"conf40 p50 {np.percentile(c40,50):.3f}  delta p50 {np.median(c40-c60):+.3f} "
              f"up {int(((c40-c60)>0.02).sum())}/{len(v)}")

    # Failed merge details (IoU 0.25..0.5, FOV40 exact-bbox merge).
    print("\n[failed-merge candidates]  FOV40, IoU in [0.25,0.5), different detections:")
    shown = 0
    for i in range(len(mg40)):
        for j in range(i + 1, len(mg40)):
            iou = _iou(mg40[i]["bbox"], mg40[j]["bbox"])
            if 0.25 <= iou < 0.5:
                for k in (i, j):
                    b = mg40[k]["bbox"]
                    print(f"    tile {mg40[k]['tile_id']:2d} conf {mg40[k]['conf']:.3f} "
                          f"centre ({mg40[k]['center'][0]:.1f},{mg40[k]['center'][1]:.1f}) "
                          f"eqbox {b[2]-b[0]:.1f}x{b[3]-b[1]:.1f}px")
                print(f"    -> pair IoU {iou:.3f}")
                shown += 1
    if not shown:
        print("    none")

    # Visual: far band crop (upscaled) for counting by eye.
    y0, y1 = int((0.5 - 2 / 180) * H), int((0.5 + 20 / 180) * H)
    band = f[y0:y1]
    vis = cv2.resize(band, (W, (y1 - y0) * 3))
    for x in mg40:
        if pitch_of(x["center"][1]) > FAR_PITCH:
            b = x["bbox"]
            cv2.rectangle(vis, (int(b[0]), int((b[1] - y0) * 3)),
                          (int(b[2]), int((b[3] - y0) * 3)), (0, 0, 255), 3)
    cv2.imwrite("equirect_test/phase4_far_band.png", vis)
    print(f"\n[visual] far band (pitch +2..-20) saved -> equirect_test/phase4_far_band.png "
          f"(rows {y0}-{y1}, {len([x for x in mg40 if pitch_of(x['center'][1])>-20])} boxes)")


if __name__ == "__main__":
    main()
