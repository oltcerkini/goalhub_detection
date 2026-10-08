#!/usr/bin/env python3
"""VR180 end-to-end pipeline (equirect mode).

    python vr180_pipeline.py --video assets/180Videos/<clip>.mkv \
        [--calibration calib/calibration.json] [--start 85] [--frames 120] \
        [--out out/track.json]

Per frame:
    normalize (equirect.normalize_frame)  -> canonical equirect content
    FOV-60 spherical tiling + YOLO + merge -> player detections in equirect px
    full-frame ball detector (low conf)    -> ball in equirect px
    ByteTrack (+ Re-ID) on equirect coords -> stable player ids
    optional calibration                   -> metric pitch (x, z) in metres

Writes a versioned tracking JSON. The flat pipeline is NOT touched.

Coordinate spaces:
    raw video px  --(region)-->  content px (canonical equirect)  --(calib)-->  metres
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "calib"))

from equirect import normalize_frame                       # noqa: E402
from spherical_tiling import (build_grid, tile_equirect, merge_to_equirect,   # noqa: E402
                              grass_roi_polygon, roi_filter)
from player_tracker import PlayerTracker                    # noqa: E402

MODE = "equirect"
TRACK_VERSION = 1
BALL_CONF = 0.05
PLAYER_CONF = 0.10
BALL_MODEL = "ball_detector_yolo26m.engine"
PLAYER_MODEL = "yolo26l.pt"


def load_calibration(path):
    if not path or not Path(path).exists():
        return None
    cal = json.loads(Path(path).read_text())
    if not cal.get("quality", {}).get("valid", False):
        print("[calib] calibration_valid = false -> metric output will be omitted")
    return cal


def make_metric(cal, W, H):
    """Return a function equirect_px -> (x_m, z_m) or None if no valid calib."""
    if not cal:
        return None
    from spherical_calib import make_mappers
    pose = {"P": cal["camera_pose"]["position_m"], "rvec": cal["camera_pose"]["rotation_vector"]}
    span = float(cal.get("span_deg", 180.0)); vspan = float(cal.get("vspan_deg", 180.0))
    e2p, _ = make_mappers(pose, W, H, span)
    valid = cal.get("quality", {}).get("valid", False)

    def f(x_eq, y_eq):
        if not valid:
            return None
        x, z = e2p(x_eq, y_eq)
        x, z = float(np.ravel(x)[0]), float(np.ravel(z)[0])
        if not (np.isfinite(x) and np.isfinite(z)):
            return None
        return [round(x, 2), round(z, 2)]
    return f


def run(video, calibration=None, span=180.0, fov=60.0, start_s=0.0, max_frames=None,
        step=1, ball=True, roi=True, verbose=True):
    cal = load_calibration(calibration)
    player_model = __import__("ultralytics").YOLO(PLAYER_MODEL)
    ball_model = __import__("ultralytics").YOLO(BALL_MODEL) if ball else None
    tracker = PlayerTracker(max_missed=90, proximity_px=250,
                            appearance_threshold=0.4, match_distance_weight=0.5)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    grid = build_grid(span, fov)
    frames, ball_trail = [], []
    W = H = None
    e2p = None
    roi_poly = None
    t0 = time.time()
    n = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        if n % step:
            n += 1; continue
        content, meta = normalize_frame(raw, span=span)
        if W is None:
            H, W = content.shape[:2]
            e2p = make_metric(cal, W, H)
            if roi:
                roi_poly = grass_roi_polygon(content)
        fidx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # players: tile -> detect -> merge (equirect px)
        dets = []
        for img, yaw, pitch, f, tid in tile_equirect(content, grid, span):
            r = player_model.predict(img, imgsz=1280, conf=PLAYER_CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": f,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        merged, _ = merge_to_equirect(dets, span, W, H, criterion="center")
        if roi_poly is not None:
            merged, _ = roi_filter(merged, roi_poly)

        # track
        tracks = []
        if merged:
            d = sv.Detections(xyxy=np.array([m["bbox"] for m in merged], float),
                              confidence=np.array([m["conf"] for m in merged], float),
                              class_id=np.zeros(len(merged), int))
            tr = tracker.update(d, frame=None)
            if tr is not None and tr.tracker_id is not None:
                for i in range(len(tr)):
                    x1, y1, x2, y2 = map(float, tr.xyxy[i])
                    tid = int(tr.tracker_id[i])
                    gx, gy = (x1 + x2) / 2, y2            # GROUND CONTACT (bottom-centre)
                    rec = {"id": tid, "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                           "conf": round(float(tr.confidence[i]), 3),
                           "eq_ground": [round(gx, 1), round(gy, 1)]}
                    if e2p:
                        m = e2p(gx, gy)
                        if m:
                            rec["pitch_m"] = m
                    tracks.append(rec)

        # ball on the FULL frame (validated path), mapped to content px
        ball_rec = None
        if ball_model is not None:
            r = ball_model.predict(raw, imgsz=1024, conf=BALL_CONF, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                j = int(r.boxes.conf.cpu().numpy().argmax())
                if int(r.boxes.cls.cpu().numpy()[j]) == 0:
                    b = r.boxes.xyxy.cpu().numpy()[j]
                    cx, cy = (b[0] + b[2]) / 2 - meta["region"][0], (b[1] + b[3]) / 2 - meta["region"][1]
                    conf = float(r.boxes.conf.cpu().numpy()[j])
                    ball_rec = {"eq": [round(float(cx), 1), round(float(cy), 1)],
                                "conf": round(conf, 3)}
                    if e2p:
                        m = e2p(cx, cy)
                        if m:
                            ball_rec["pitch_m"] = m
                    ball_trail.append([fidx, round(float(cx), 1), round(float(cy), 1), round(conf, 3)])

        frames.append({"frame": fidx, "t": round(fidx / fps, 2),
                       "players": tracks, "ball": ball_rec})
        n += 1
        if max_frames and len(frames) >= max_frames:
            break
        if verbose and len(frames) % 20 == 0:
            print(f"  {len(frames)} frames, {len(tracks)} tracks this frame, "
                  f"{time.time()-t0:.0f}s")
    cap.release()

    out = {
        "version": TRACK_VERSION, "mode": MODE, "video": str(video),
        "image_size": [W, H], "span_deg": span,
        "layout": meta["layout"], "region": meta["region"],
        "calibration_valid": bool(cal and cal.get("quality", {}).get("valid", False)),
        "fps": fps, "frames": frames, "ball_trail": ball_trail,
    }
    if verbose:
        print(f"done: {len(frames)} frames in {time.time()-t0:.0f}s; "
              f"ball detections {len(ball_trail)}; calibration_valid={out['calibration_valid']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--span", type=float, default=180.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--out", default="out/track.json")
    a = ap.parse_args()
    res = run(a.video, a.calibration, a.span, start_s=a.start, max_frames=a.frames,
              step=a.step, ball=not a.no_ball)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
