"""Phase 3 — narrow-FOV diagnostic + infra fixes (3a ROI, 3b sidecar, 3c edges).

New module usage only; no existing pipeline module imported or modified.

    python test_phase3_fov.py [--video PATH] [--time 85]

Reproduces the Phase 2 FOV=60 numbers with full percentiles, then runs the same
pipeline at FOV=40 (24 tiles) to test whether narrowing the FOV raises far-side
confidence. Also demonstrates ROI masking, the sidecar config, and edge masking.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from spherical_tiling import (
    detect_content_region, apply_region, detect_span, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, save_sidecar, sidecar_path,
    grass_roi_polygon, roi_filter, hemisphere_mask, TILE_W, TILE_H,
    GRID_180,
)

VIDEO = ("equirect_test/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"
CONF = 0.10
MATCH_PX = 140.0        # equirect px radius to pair a player across FOVs


def grab(video, t, region=None):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, raw = cap.read(); cap.release()
    assert ok, "frame read failed"
    region = region or detect_content_region(raw)
    return apply_region(raw, region), region


def detect_all(tiles, imgsz=1280):
    from ultralytics import YOLO
    model = YOLO("yolo26l.pt")
    dets, per_tile = [], {}
    for img, yaw, pitch, fov, tid in tiles:
        r = model.predict(img, imgsz=imgsz, conf=CONF, verbose=False)[0]
        n = 0
        if r.boxes is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for c, s, b in zip(cls, conf, xyxy):
                if c != 0:
                    continue
                n += 1
                dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                             "bbox": tuple(float(v) for v in b), "conf": float(s), "cls": 0})
        per_tile[tid] = n
    return dets, per_tile


def run_fov(frame, span, fov):
    grid = build_grid(span, fov)
    tiles = tile_equirect(frame, grid, span)          # edges masked (3c)
    t0 = time.time()
    dets, per_tile = detect_all(tiles)
    dt = time.time() - t0
    merged, mapped = merge_to_equirect(dets, span, frame.shape[1], frame.shape[0], 0.5)
    # per-tile counts
    print(f"\n[FOV {fov:.0f}] {len(grid)} tiles, {len(dets)} raw detections, "
          f"{len(merged)} after merge, {dt:.1f}s ({dt/len(grid):.2f}s/tile)")
    print("   per-tile:", " ".join(f"t{i}={n}" for i, n in sorted(per_tile.items())))
    # per-row percentiles
    for p in GRID_180[float(fov)]["pitch"]:
        c = np.array([d["conf"] for d in mapped if d["pitch"] == p])
        if len(c):
            print(f"   pitch {p:+3.0f}deg: n={len(c):3d}  p10 {np.percentile(c,10):.3f}  "
                  f"p50 {np.percentile(c,50):.3f}  p90 {np.percentile(c,90):.3f}  "
                  f"(min {c.min():.3f} max {c.max():.3f})")
        else:
            print(f"   pitch {p:+3.0f}deg: n=0")
    return {"fov": fov, "dets": dets, "mapped": mapped, "merged": merged,
            "per_tile": per_tile, "dt": dt, "ntiles": len(grid)}


def pair_far_side(r60, r40, eq_h):
    """Match detections across FOVs by equirect proximity; report confidence
    deltas split into far row (pitch<-30) and near row."""
    far, near = [], []
    m40 = r40["mapped"]
    for a in r60["mapped"]:
        best, bd = None, MATCH_PX
        for b in m40:
            d = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
            if d < bd:
                best, bd = b, d
        if best is None:
            continue
        pitch = (0.5 - a["center"][1] / eq_h) * 180.0     # equirect pitch of the player
        d = best["conf"] - a["conf"]
        (far if pitch < -30 else near).append((a["conf"], best["conf"], d))
    print("\n[compare] same player, FOV60 vs FOV40 (matched within "
          f"{MATCH_PX:.0f}px in equirect):")
    for name, arr in (("FAR  (pitch<-30)", far), ("NEAR (pitch>=-30)", near)):
        if not arr:
            print(f"   {name}: no matched pairs")
            continue
        c60 = np.array([x[0] for x in arr]); c40 = np.array([x[1] for x in arr])
        dd = c40 - c60
        print(f"   {name}: n={len(arr):3d}  conf60 p50 {np.percentile(c60,50):.3f} -> "
              f"conf40 p50 {np.percentile(c40,50):.3f}  delta p50 {np.median(dd):+.3f} "
              f"(mean {dd.mean():+.3f}, up {int((dd>0.02).sum())}/{len(arr)})")
    return far, near


def edge_test(frame, span):
    """Compare edge-tile detections with and without hemisphere masking (3c)."""
    grid = build_grid(span, 60.0)
    edge_ids = [t[0] for t in grid if abs(t[1]) >= 85]
    print(f"\n[edge 3c] edge tiles (|yaw|>=85): {edge_ids}")
    unmasked = tile_equirect(frame, grid, span, mask_edges=False)
    masked = tile_equirect(frame, grid, span, mask_edges=True)
    for (img_u, yaw, pitch, fov, tid), (img_m, *_ ) in zip(unmasked, masked):
        if tid not in edge_ids:
            continue
        m = hemisphere_mask(yaw, pitch, fov, span)
        rep_frac = 1.0 - m.mean()
        from ultralytics import YOLO
        model = YOLO("yolo26l.pt")
        def n_persons(im):
            r = model.predict(im, imgsz=1280, conf=CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                return 0
            return int((r.boxes.cls.cpu().numpy().astype(int) == 0).sum())
        nu, nm = n_persons(img_u), n_persons(img_m)
        print(f"   tile {tid} yaw{yaw:+.0f} pitch{pitch:+.0f}: replicated "
              f"{rep_frac*100:4.1f}% of tile | persons unmasked {nu} -> masked {nm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--time", type=float, default=85.0)
    ap.add_argument("--span", type=float, default=180.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # 3b — sidecar config (create if missing, then use it).
    sc = load_sidecar(args.video)
    if sc is None:
        raw = cv2.VideoCapture(args.video)
        raw.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
        _, img0 = raw.read(); raw.release()
        region = detect_content_region(img0)
        frame0 = apply_region(img0, region)
        roi = grass_roi_polygon(frame0)
        roi_s = cv2.approxPolyDP(roi.reshape(-1, 1, 2), 8, True).reshape(-1, 2).tolist() if roi is not None else None
        sc = {"span": args.span, "crop": list(region), "roi": roi_s}
        print(f"[3b] sidecar created: {save_sidecar(args.video, sc)}")
    else:
        print(f"[3b] sidecar loaded: {sidecar_path(args.video)}")
    print(f"[3b] span={sc.get('span')}  crop={sc.get('crop')}  "
          f"roi_pts={len(sc['roi']) if sc.get('roi') else 0}")

    span = float(sc.get("span", args.span))
    frame, region = grab(args.video, args.time, sc.get("crop"))
    auto = detect_span(frame.shape[1], frame.shape[0])
    print(f"[span] content {frame.shape[1]}x{frame.shape[0]} aspect "
          f"{frame.shape[1]/frame.shape[0]:.2f} -> auto {auto:.0f}; sidecar/override {span:.0f}")

    r60 = run_fov(frame, span, 60.0)
    r40 = run_fov(frame, span, 40.0)
    print(f"\n[runtime] FOV60 {r60['dt']:.1f}s/{r60['ntiles']} tiles = "
          f"{r60['dt']/r60['ntiles']:.2f}s per tile; FOV40 {r40['dt']:.1f}s/"
          f"{r40['ntiles']} tiles = {r40['dt']/r40['ntiles']:.2f}s per tile")

    pair_far_side(r60, r40, frame.shape[0])

    # 3a — ROI mask on the FOV40 merged detections.
    roi = sc.get("roi")
    if roi:
        kept, dropped = roi_filter(r40["merged"], np.array(roi, np.int32))
        print(f"\n[3a ROI] FOV40 merged {len(r40['merged'])} -> kept {len(kept)}, "
              f"dropped {len(dropped)} (spectators/off-pitch)")
    else:
        print("\n[3a ROI] no polygon")

    edge_test(frame, span)


if __name__ == "__main__":
    main()
