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
                              grass_roi_polygon, roi_filter,
                              pitch_polygon_from_trace, trace_roi_filter)
from player_tracker import PlayerTracker                    # noqa: E402
from ball_trail import BallTrailTracker                     # noqa: E402
from ball_lockon import (BallLockOn, pitch_polygon as calib_pitch_polygon,  # noqa: E402
                         pitch_roi_filter)
from goal_detect import detect_goal_entries                 # noqa: E402
from spherical_projection import equirect_to_perspective    # noqa: E402
from virtual_camera import compute_camera_pose              # noqa: E402

MODE = "equirect"
TRACK_VERSION = 1
BALL_CAND_CONF = 0.03
PLAYER_CONF = 0.10
# ByteTrack's default activation (0.25) drops far players that the tiled
# detector finds at 0.10-0.25. Activating at 0 -> det_thresh 0.10, so every
# kept detection can start a track; the hit filter below removes one-frame noise.
TRACK_ACTIVATION = 0.0
TRACK_MIN_HITS = 1
MIN_BOX_H = 8.0
BOX_ASPECT = (0.6, 8.0)      # height / width allowed for a standing player
BALL_MODEL = "ball_detector_yolo26m.engine"
PLAYER_MODEL = "yolo26l.pt"


def _plausible_player_box(b):
    w, h = b[2] - b[0], b[3] - b[1]
    if w <= 1.0 or h < MIN_BOX_H:
        return False
    return BOX_ASPECT[0] <= (h / w) <= BOX_ASPECT[1]


ENHANCE = "auto"
ENHANCE_DARK_MEAN = 70.0      # night clip content ~52; overcast day ~92
ENHANCE_DARK_FRAC = 0.45

# Vertical span from content aspect (square-pixel prior). A 180-wide clip whose
# content is twice as tall as wide is NOT 180 tall: assuming it is stretches
# players tall-and-thin in the tile projection and the detector misses most of
# them (measured: 3 -> 29 players/frame on Barca-Challenger at vspan 180 vs 108).
VSPAN_MIN, VSPAN_MAX = 20.0, 180.0
GRASS_ROI_MIN_KEEP = 0.15     # auto-ROI keeping less than this share of detections is broken


def _auto_vspan(eq_w, eq_h):
    return round(float(min(VSPAN_MAX, max(VSPAN_MIN, 180.0 * eq_h / eq_w))), 2)


def _enhance_frame(frame, mode):
    """CLAHE on the luminance channel. 'auto' only fires on dark frames."""
    if mode == "none":
        return frame, "none"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "auto":
        dark = (float(gray.mean()) < ENHANCE_DARK_MEAN
                or float((gray < 32).mean()) > ENHANCE_DARK_FRAC)
        if not dark:
            return frame, "none"
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), "clahe"


def load_calibration(path):
    if not path or not Path(path).exists():
        return None
    cal = json.loads(Path(path).read_text())
    if not cal.get("quality", {}).get("valid", False):
        print("[calib] calibration_valid = false -> metric output will be omitted")
    return cal


def make_metric(cal, W, H, roi_margin_m=3.0):
    """Bind the calibration to the frame size.

    Returns None when there is no calibration. Otherwise a mapping:
        {valid, to_pitch(x,y)->[x_m,z_m]|None, on_pitch(x,y)->bool}
    `to_pitch` returns None for pixels that do not meet the ground (above the
    horizon) or when the calibration is not valid — so metric analytics are
    withheld rather than silently wrong. `on_pitch` is the calibration-derived
    ROI test (None when the calibration is not valid -> caller falls back).
    """
    if not cal:
        return None
    from spherical_calib import equirect_to_ground, pitch_contains
    pose = {"P": cal["camera_pose"]["position_m"], "rvec": cal["camera_pose"]["rotation_vector"]}
    span = float(cal.get("span_deg", 180.0))
    vspan = float(cal.get("vspan_deg", 180.0))
    dims = cal.get("pitch") or {}
    valid = bool(cal.get("quality", {}).get("valid", False))

    def to_pitch(x_eq, y_eq):
        if not valid:
            return None
        x, z, ok = equirect_to_ground(x_eq, y_eq, pose, W, H, span, vspan)
        x, z, ok = float(np.ravel(x)[0]), float(np.ravel(z)[0]), bool(np.ravel(ok)[0])
        if not (ok and np.isfinite(x) and np.isfinite(z)):
            return None
        return [round(x, 2), round(z, 2)]

    def on_pitch(x_eq, y_eq):
        if not valid:
            return None
        m = pitch_contains(x_eq, y_eq, pose, W, H, span, dims, vspan, roi_margin_m)
        return bool(np.ravel(m)[0])

    return {"valid": valid, "to_pitch": to_pitch, "on_pitch": on_pitch}


def _transcode_h264(path, verbose=True):
    """cv2 writes mp4v, which browsers won't play. Remux to h264 when ffmpeg exists."""
    import os
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return path
    src = str(path) + ".src.mp4"
    os.replace(str(path), src)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(path)], check=True, timeout=3600)
    except Exception:                                 # noqa: BLE001
        os.replace(src, str(path))                    # keep the mp4v file if ffmpeg fails
    else:
        try:
            os.remove(src)
        except OSError:
            pass
    if verbose:
        print(f"rendered {path}")
    return path


def run(video, calibration=None, span=None, fov=60.0, start_s=0.0, max_frames=None,
        step=1, ball=True, roi=True, region=None, render=None, follow=None, trace=None,
        roi_margin_frac=0.20, track_activation=TRACK_ACTIVATION,
        track_min_hits=TRACK_MIN_HITS, vspan=None, enhance=ENHANCE, verbose=True,
        goals=False):
    # region: pin the content crop [x,y,w,h]. The auto-detector can drift a few px
    # between frames; when the calibration trace was drawn on one frame, every
    # frame must be cropped identically or the pose is applied to a shifted space.
    cal = load_calibration(calibration)
    if trace is None and calibration:
        # app180 saves <video_id>_trace.json next to <video_id>.json
        cand = Path(calibration).with_name(Path(calibration).stem + "_trace.json")
        if cand.exists():
            trace = cand
    if trace and not isinstance(trace, dict):
        trace = json.loads(Path(trace).read_text())
    # Projection spans: a valid calibration has solved them from the traced
    # pitch (the content is a cropped window, not reliably 180x180). Explicit
    # arguments win; otherwise prefer the calibration, then a 180 span and the
    # content-aspect vspan. Detection is very sensitive to vspan (wrong values
    # stretch players tall-and-thin in the tiles).
    cal_valid = bool(cal and cal.get("quality", {}).get("valid"))
    # Only v2+ calibrations solved the spans from the pitch; v1 stored a vspan
    # of 180, which is wrong for these cropped clips and hurts detection.
    cal_spans = cal_valid and int(cal.get("version", 1)) >= 2
    if span is not None:
        span = float(span)
        span_source = "explicit"
    elif cal_spans:
        span = float(cal.get("span_deg", 180.0))
        span_source = "calibration"
    else:
        span = 180.0
        span_source = "default"
    if vspan is not None:
        vspan = float(vspan)
        vspan_source = "explicit"
    elif cal_spans and cal.get("vspan_deg") is not None:
        vspan = float(cal["vspan_deg"])
        vspan_source = "calibration"
    else:
        vspan = None                 # resolved from the first frame's content aspect
        vspan_source = "auto"
    player_model = __import__("ultralytics").YOLO(PLAYER_MODEL)
    ball_model = __import__("ultralytics").YOLO(BALL_MODEL) if ball else None
    ball_tracker = (BallTrailTracker(gate_px=100.0, min_hits=3, max_missed=10)
                    if ball_model is not None else None)
    tracker = PlayerTracker(max_missed=90, proximity_px=250,
                            appearance_threshold=0.4, match_distance_weight=0.5,
                            track_activation_threshold=track_activation)
    track_hits = {}
    enh_counts = {"none": 0, "clahe": 0}

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    grid = None
    frames, ball_trail = [], []
    # goal detection (optional; needs a valid calibration with traced goal frames)
    lock = None
    lock_trail = []            # [t, x, y, conf] content-pixel detections
    goal_pose = None
    goal_events = []
    scan_buf, scan_times = [], []
    acq_attempts = 0
    scan_pause_until = -1e9
    writer = None
    fwriter = None
    cam_yaw, cam_pitch = 0.0, -30.0      # virtual follow-camera state (deg)
    cam_init = False
    FOLLOW_FOV, FW, FH = 70.0, 1280, 720
    W = H = None
    metric = None
    roi_poly = None
    trace_poly = None
    roi_mode = None
    t0 = time.time()
    n = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        if n % step:
            n += 1; continue
        content, meta = normalize_frame(raw, span=span, region=region)
        proc, enh = _enhance_frame(content, enhance)
        enh_counts[enh] = enh_counts.get(enh, 0) + 1
        if W is None:
            H, W = content.shape[:2]
            if vspan is None:
                vspan = _auto_vspan(W, H)
            grid = build_grid(span, fov, vspan)
            metric = make_metric(cal, W, H)
            if (goals and cal_valid and ball_model is not None
                    and int(cal.get("version", 1)) >= 2):
                try:
                    gp = calib_pitch_polygon(cal, W, H)
                except Exception:
                    gp = None
                goal_pose = {"P": np.array(cal["camera_pose"]["position_m"], float),
                             "rvec": np.array(cal["camera_pose"]["rotation_vector"], float)}
                lock = BallLockOn(ball_model, W, H, pitch_poly=gp,
                                  candidate_filter=pitch_roi_filter(cal, W, H))
            # The traced outline sits on the paint, so prefer it over the fitted
            # pose (ill-conditioned on touchline cameras). Grass is the last resort.
            trace_poly = pitch_polygon_from_trace(trace, W, H) if trace else None
            if roi and trace_poly is None and not (metric and metric["valid"]):
                roi_poly = grass_roi_polygon(proc)
        fidx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # goal ball lock-on (optional): scan to acquire, then dense-follow. Scans
        # are expensive (~2 s each) so only run a few, and only while searching.
        if lock is not None:
            t = fidx / fps
            lf = proc if proc.shape[:2] == (H, W) else cv2.resize(proc, (W, H))
            if lock.state == "locked":
                rec = lock.update(t, lf)
                if rec is not None and rec[4]:
                    lock_trail.append([round(t, 3), round(rec[1], 1),
                                       round(rec[2], 1), round(rec[3], 3)])
            elif acq_attempts < 12 and t >= scan_pause_until:
                # retry after loss as well as at the start; each acquisition is
                # ~10 s of scanning, so keep a global cap on attempts.
                if not scan_times or t - scan_times[-1] >= 0.4:
                    scan_buf.append(lf)
                    scan_times.append(t)
                if len(scan_buf) >= 5:
                    seed = lock.acquire(scan_buf, scan_times)
                    acq_attempts += 1
                    scan_buf, scan_times = [], []
                    if seed is not None:
                        if verbose:
                            print(f"  [goal] lock-on acquired at {t:.1f}s "
                                  f"({seed['x']:.0f},{seed['y']:.0f}) n={seed['n']}")
                        for tt, xx, yy, cc, _ok in lock.trail:
                            lock_trail.append([round(tt, 3), round(xx, 1),
                                               round(yy, 1), round(cc, 3)])
                    else:
                        scan_pause_until = t + 5.0
                        if verbose:
                            print(f"  [goal] acquisition failed at {t:.1f}s; retry later")

        # players: tile -> detect -> merge (equirect px)
        dets = []
        for img, yaw, pitch, f, tid in tile_equirect(proc, grid, span, vspan=vspan):
            r = player_model.predict(img, imgsz=1280, conf=PLAYER_CONF, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": f,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        merged, _ = merge_to_equirect(dets, span, W, H, criterion="center", vspan=vspan)
        merged = [m for m in merged if _plausible_player_box(m["bbox"])]
        roi_mode = None
        if roi and trace_poly is not None:
            # ROI from the user's traced pitch outline (ground-contact point)
            merged, _ = trace_roi_filter(merged, trace_poly,
                                         margin_frac=roi_margin_frac)
            roi_mode = "trace"
        elif roi and metric and metric["valid"]:
            # fallback: calibration-derived ground rectangle
            merged = [m for m in merged
                      if metric["on_pitch"]((m["bbox"][0] + m["bbox"][2]) / 2.0,
                                            m["bbox"][3])]
            roi_mode = "pose"
        elif roi_poly is not None:
            kept, _ = roi_filter(merged, roi_poly)
            if len(merged) >= 8 and len(kept) < max(3, GRASS_ROI_MIN_KEEP * len(merged)):
                # The colour-derived hull missed most of the pitch (night/floodlit
                # frames segment only the lit patches). Dropping real players is
                # worse than keeping a few off-pitch ones.
                roi_mode = "grass-rejected"
            else:
                merged = kept
                roi_mode = "grass"

        # track
        tracks = []
        if merged:
            d = sv.Detections(xyxy=np.array([m["bbox"] for m in merged], float),
                              confidence=np.array([m["conf"] for m in merged], float),
                              class_id=np.zeros(len(merged), int))
            tr = tracker.update(d, frame=None, strict_iou=False)
            if tr is not None and tr.tracker_id is not None:
                for i in range(len(tr)):
                    tid = int(tr.tracker_id[i])
                    track_hits[tid] = track_hits.get(tid, 0) + 1
                    # First frame: emit everything so a 1-frame run still shows
                    # players. After that, a track needs `track_min_hits` frames.
                    if frames and track_hits[tid] < track_min_hits:
                        continue
                    x1, y1, x2, y2 = map(float, tr.xyxy[i])
                    gx, gy = (x1 + x2) / 2, y2            # GROUND CONTACT (bottom-centre)
                    rec = {"id": tid, "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                           "conf": round(float(tr.confidence[i]), 3),
                           "eq_ground": [round(gx, 1), round(gy, 1)]}
                    if metric:
                        m = metric["to_pitch"](gx, gy)
                        if m:
                            rec["pitch_m"] = m
                    tracks.append(rec)

        # ball: low-confidence full-frame candidates -> temporal trail tracker
        ball_rec = None
        if ball_model is not None:
            ball_frame = raw
            if enh == "clahe":
                ball_frame, _ = _enhance_frame(raw, "clahe")
            r = ball_model.predict(ball_frame, imgsz=1024, conf=BALL_CAND_CONF,
                                   verbose=False)[0]
            cands = []
            if r.boxes is not None:
                rx, ry = meta["region"][0], meta["region"][1]
                for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                                   r.boxes.conf.cpu().numpy(),
                                   r.boxes.xyxy.cpu().numpy()):
                    if c != 0:
                        continue
                    x1, y1, x2, y2 = map(float, b)
                    w, h = x2 - x1, y2 - y1
                    if not (3.0 <= w <= 100.0 and 3.0 <= h <= 100.0
                            and 0.4 <= (w / h) <= 2.5):
                        continue
                    cx = (x1 + x2) / 2.0 - rx
                    cy = (y1 + y2) / 2.0 - ry
                    if trace_poly is not None:
                        if cv2.pointPolygonTest(trace_poly, (float(cx), float(cy)),
                                                True) < -20.0:
                            continue
                    elif metric and metric["valid"]:
                        if metric["on_pitch"](cx, cy) is False:
                            continue
                    cands.append({"x": cx, "y": cy, "w": w, "h": h,
                                  "conf": float(s)})
            best = ball_tracker.update(fidx, cands)
            if best is not None:
                bx, by = float(best["x"]), float(best["y"])
                rec = {"eq": [round(bx, 1), round(by, 1)],
                       "conf": round(float(best["conf"]), 3),
                       "id": int(best["id"])}
                if int(best["last"]) != fidx:
                    rec["predicted"] = True
                if metric:
                    m = metric["to_pitch"](bx, by)
                    if m:
                        rec["pitch_m"] = m
                ball_rec = rec
                ball_trail.append([fidx, round(bx, 1), round(by, 1),
                                   round(float(best["conf"]), 3)])

        frames.append({"frame": fidx, "t": round(fidx / fps, 2),
                       "players": tracks, "ball": ball_rec})

        if render is not None or follow is not None:
            # annotated content (drawn on the equirect, then optionally reprojected)
            vis = proc.copy()
            for p in tracks:
                x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 220, 130), 2)
                tag = f'#{p["id"]}'
                if p.get("pitch_m"):
                    tag += f'  {p["pitch_m"][0]:.1f},{p["pitch_m"][1]:.1f} m'
                org = (x1, max(16, y1 - 6))
                cv2.putText(vis, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 130), 1, cv2.LINE_AA)
            if ball_rec:
                cv2.circle(vis, tuple(int(v) for v in ball_rec["eq"]), 12, (0, 215, 255), 2)
            hdr = f'{fidx / fps:6.1f}s   players {len(tracks)}'
            if not (cal and cal.get("quality", {}).get("valid", False)):
                hdr += "   (no calibration — positions unavailable)"
            cv2.putText(vis, hdr, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, hdr, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            if render is not None:
                if writer is None:
                    writer = cv2.VideoWriter(render, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(fps / max(step, 1), 1.0), (W, H))
                writer.write(vis)
            if follow is not None:
                # target the ball if tracked, else the players' ground centroid
                tgt = None
                if ball_rec:
                    tgt = (ball_rec["eq"][0], ball_rec["eq"][1])
                elif tracks:
                    tgt = (float(np.median([p["eq_ground"][0] for p in tracks])),
                           float(np.median([p["eq_ground"][1] for p in tracks])))
                if tgt is not None:
                    # snap onto the first target, then ease for a jitter-free follow
                    cam_yaw, cam_pitch = compute_camera_pose(
                        tgt, cam_yaw, cam_pitch,
                        dt=max(step, 1) / max(fps, 1.0), span=span, eq_w=W, eq_h=H,
                        alpha=(1.0 if not cam_init else 0.1))
                    cam_init = True
                view = equirect_to_perspective(vis, cam_yaw, cam_pitch, FOLLOW_FOV,
                                               FW, FH, h_span_deg=span, v_span_deg=vspan)
                if fwriter is None:
                    fwriter = cv2.VideoWriter(follow, cv2.VideoWriter_fourcc(*"mp4v"),
                                              max(fps / max(step, 1), 1.0), (FW, FH))
                fwriter.write(view)
        n += 1
        if max_frames and len(frames) >= max_frames:
            break
        if verbose and len(frames) % 20 == 0:
            print(f"  {len(frames)} frames, {len(tracks)} tracks this frame, "
                  f"{time.time()-t0:.0f}s")
    cap.release()
    if writer is not None:
        writer.release()
        _transcode_h264(render, verbose)
    if fwriter is not None:
        fwriter.release()
        _transcode_h264(follow, verbose)

    if goals and goal_pose is not None and W is not None:
        srcs = [("lockon", [(x, y, int(round(t * fps)), c) for t, x, y, c in lock_trail]),
                ("fullframe", [(x, y, int(f), c) for f, x, y, c in ball_trail])]
        for side in ("left", "right"):
            side_ev = []
            for name, pts in srcs:
                if not pts:
                    continue
                for e in detect_goal_entries(pts, goal_pose, side, W, H,
                                             span, vspan, fps=fps):
                    e["source"] = name
                    side_ev.append(e)
            side_ev.sort(key=lambda e: e["t_start"])
            kept = []
            for e in side_ev:                  # one event per side per ~5 s
                if kept and abs(e["t_start"] - kept[-1]["t_start"]) < 5.0:
                    if e["source"] == "lockon" and kept[-1]["source"] != "lockon":
                        kept[-1] = e
                    continue
                kept.append(e)
            goal_events.extend(kept)
        if verbose:
            for e in goal_events:
                print(f"  [goal] {e['side']} goal at {e['t_start']:.2f}s "
                      f"({e['source']}, approach {e['approach_px_s']:.0f}px/s)")

    out = {
        "version": TRACK_VERSION, "mode": MODE, "video": str(video),
        "image_size": [W, H], "span_deg": span, "vspan_deg": vspan,
        "span_source": span_source, "vspan_source": vspan_source,
        "enhance": enhance, "enhance_counts": enh_counts,
        "layout": meta["layout"], "region": meta["region"],
        "calibration_valid": bool(cal and cal.get("quality", {}).get("valid", False)),
        "roi_mode": roi_mode, "roi_margin_frac": roi_margin_frac,
        "track_activation": track_activation, "track_min_hits": track_min_hits,
        "tracker_strict_iou": False,
        "fps": fps, "frames": frames, "ball_trail": ball_trail,
        "goals_enabled": bool(goals), "goal_events": goal_events,
        "ball_lockon_trail": lock_trail,
    }
    if verbose:
        print(f"done: {len(frames)} frames in {time.time()-t0:.0f}s; "
              f"ball detections {len(ball_trail)}; calibration_valid={out['calibration_valid']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--span", type=float, default=None,
                    help="horizontal span (default: calibration span, else 180)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--region", type=int, nargs=4, default=None,
                    help="pin the content crop [x y w h] (must match the calibration frame)")
    ap.add_argument("--trace", default=None,
                    help="calibration trace JSON; its pitch outline becomes the ROI")
    ap.add_argument("--roi-margin-frac", type=float, default=0.20,
                    help="ground-contact margin as a fraction of bbox height")
    ap.add_argument("--track-activation", type=float, default=TRACK_ACTIVATION,
                    help="ByteTrack activation threshold (0 = track all kept detections)")
    ap.add_argument("--track-min-hits", type=int, default=TRACK_MIN_HITS,
                    help="frames a track must appear before it is emitted")
    ap.add_argument("--vspan", type=float, default=None,
                    help="vertical span of the content (default: calibration vspan_deg, "
                         "else inferred from the content aspect)")
    ap.add_argument("--enhance", choices=["none", "clahe", "auto"], default=ENHANCE,
                    help="low-light enhancement (auto = CLAHE only on dark frames)")
    ap.add_argument("--render", default=None, help="write an annotated mp4 here")
    ap.add_argument("--render-follow", default=None,
                    help="write a virtual follow-camera mp4 here")
    ap.add_argument("--out", default="out/track.json")
    ap.add_argument("--goals", action="store_true",
                    help="run goal-ball lock-on and emit goal_events (needs a valid "
                         "calibration whose trace includes the goal frames)")
    a = ap.parse_args()
    res = run(a.video, a.calibration, a.span, start_s=a.start, max_frames=a.frames,
              step=a.step, ball=not a.no_ball, region=a.region, render=a.render,
              follow=a.render_follow,
              trace=a.trace, roi_margin_frac=a.roi_margin_frac,
              track_activation=a.track_activation, track_min_hits=a.track_min_hits,
              vspan=a.vspan, enhance=a.enhance, goals=a.goals)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
