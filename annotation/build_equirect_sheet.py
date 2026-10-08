#!/usr/bin/env python3
"""Build a ball-annotation sheet from equirect tiles (Phase 7 step 1).

Candidate tiles come from the arm-A ball detector run on the FOV-60 equirect
grid; a detection is a *candidate hint*, NOT a label. Stratified by pitch band
(near/mid/far) and spread over the match. Same layout as the drone sheet:

    annotation_equirect/frames/f{id:06d}.jpg   tile images (1920x1080)
    annotation_equirect/manifest.js            list for annotate.html
    annotation_equirect/annotate.html          copy of the drone annotator

Usage:  python annotation/build_equirect_sheet.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\PC\goalhub_detection")
os.chdir(r"C:\Users\PC\goalhub_detection")
import cv2
import numpy as np
from ultralytics import YOLO

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect, load_sidecar,
)

VIDEO = ("equirect_test/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUT = Path("annotation_equirect")
TARGET = 50
BALL_CONF = 0.06
# A real ball subtends ~13-80 px in a 60deg/1920px tile (0.22 m at 3-30 m).
# Anything much larger is a player's foot / clutter -> reject.
SCALE_MIN, SCALE_MAX = 8.0, 90.0
# sample times across the match (seconds); 24 fps
TIMES = [float(t) for t in range(60, 360, 3)]


def band_of(eq_pitch):
    return "far" if eq_pitch > -20 else ("mid" if eq_pitch > -35 else "near")


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    sc = load_sidecar(VIDEO) or {}
    grid = build_grid(180, 60.0)
    ball = YOLO("ball_detector_yolo26m.engine")

    cap = cv2.VideoCapture(VIDEO)
    cands = []          # (time, tile_id, tile_local_xywh, conf, band, tile_img)
    for t in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, raw = cap.read()
        if not ok:
            continue
        frame = apply_region(raw, sc.get("crop"))
        H, W = frame.shape[:2]
        tiles = tile_equirect(frame, grid, 180)
        for img, yaw, pitch, fov, tid in tiles:
            r = ball.predict(img, imgsz=1024, conf=BALL_CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for b, c in zip(xy, cf):
                x1, y1, x2, y2 = (float(v) for v in b)
                # equirect pitch of the tile centre (band proxy)
                eq_pitch = pitch
                cands.append({
                    "time": t, "tile_id": tid, "bbox": (x1, y1, x2, y2),
                    "conf": float(c), "band": band_of(eq_pitch),
                    "w": x2 - x1, "h": y2 - y1, "img": img,
                })
    cap.release()

    print(f"scanned {len(TIMES)} frames -> {len(cands)} ball candidates")
    plausible = [c for c in cands if SCALE_MIN <= max(c["w"], c["h"]) <= SCALE_MAX]
    print(f"  {len(cands)-len(plausible)} rejected as implausible scale "
          f"(> {SCALE_MAX:.0f}px = feet/clutter)")
    for b in ("far", "mid", "near"):
        cs = [c for c in cands if c["band"] == b]
        pl = [c for c in cs if SCALE_MIN <= max(c["w"], c["h"]) <= SCALE_MAX]
        if cs:
            wh = np.array([max(c["w"], c["h"]) for c in cs])
            whp = np.array([max(c["w"], c["h"]) for c in pl]) if pl else np.array([0])
            print(f"  {b:4s}: {len(cs):3d} candidates ({len(pl)} plausible)  "
                  f"all-scale p50 {np.median(wh):.0f}px  plausible-scale "
                  f"{whp.min():.0f}-{whp.max():.0f}px p50 {np.median(whp):.0f}")

    # select: within each band, keep plausible, top by conf, spread over time
    per_band = max(TARGET // 3, 1)
    selected = []
    for b in ("far", "mid", "near"):
        cs = sorted((c for c in plausible if c["band"] == b),
                    key=lambda z: -z["conf"])[:per_band * 3]
        cs = sorted(cs, key=lambda z: z["time"])
        if not cs:
            continue
        step = max(len(cs) / per_band, 1)
        picked = [cs[int(i * step)] for i in range(min(per_band, len(cs)))]
        selected.extend(picked)
    # top up with the highest-conf leftovers if short
    if len(selected) < TARGET:
        rest = sorted((c for c in cands if c not in selected),
                      key=lambda z: -z["conf"])
        selected.extend(rest[:TARGET - len(selected)])

    for old in (OUT / "frames").glob("*.jpg"):
        old.unlink()
    manifest = []
    for i, c in enumerate(selected):
        name = f"f{i:06d}.jpg"
        cv2.imwrite(str(OUT / "frames" / name), c["img"],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        x1, y1, x2, y2 = c["bbox"]
        manifest.append({
            "frame": i,
            "candidate": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                          round(c["conf"], 3)],
            "third": c["band"],
            "dual": (i % 5 == 0),
            "time_s": c["time"],
            "tile_id": c["tile_id"],
        })
    (OUT / "manifest.js").write_text(
        "const MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")

    # copy the annotator, relabel the clip id
    src = Path("annotation/annotate.html").read_text()
    (OUT / "annotate.html").write_text(
        src.replace("clip:'f5bacf2f'", "clip:'barca_equirect'"))

    print(f"\nwrote {len(manifest)} tiles -> {OUT/'frames'}")
    print(f"  open {OUT/'annotate.html'} in a browser, annotate, Save JSON")
    band_ct = {}
    for m in manifest:
        band_ct[m["third"]] = band_ct.get(m["third"], 0) + 1
    print(f"  band mix: {band_ct}")


if __name__ == "__main__":
    main()
