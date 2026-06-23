#!/usr/bin/env python3
"""
GoalHub Process — process a video with a saved calibration (no GUI).

Uses YOLO unified detection with ball fallback pipeline.
Two-pass: first pass detects+tracks, second pass renders.

Usage:
    python process.py assets/1.mp4 --calibration calibration.json --threshold 0.25 --skip 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import YOLODetector, PLAYER, BALL, REFEREE
from ball_detector import BallDetector

from player_tracker import PlayerTracker
from stats_computer import PitchMapper, StatsComputer
from post_process import PostProcessor
from team_classifier import TeamClassifier
from heatmap import HeatmapGenerator

CLASS_NAMES = {PLAYER: "Player", BALL: "Ball", REFEREE: "Referee"}
CLASS_COLOURS = {
    PLAYER: (0, 255, 0),       # green
    BALL: (0, 200, 255),       # yellow
    REFEREE: (255, 255, 255),  # white
}


def main():
    ap = argparse.ArgumentParser(description="Process video with saved calibration")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--calibration", default=None,
                    help="Calibration JSON (pitch polygon + goals)")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--model", default=None,
                    help="Path to YOLO .pt model")
    ap.add_argument("--imgsz", type=int, default=3840,
                    help="YOLO inference resolution (longest edge, default: 3840 for 4K). "
                         "Higher = better small-object detection but slower.")
    ap.add_argument("--gamma", type=float, default=0.85,
                    help="Gamma correction (1.0 = no change, <1 brightens shadows, >1 darkens)")
    ap.add_argument("--post-process", action=argparse.BooleanOptionalAction, default=True,
                    help="Post-process tracks: merge fragments, filter noise")
    ap.add_argument("--my-team", type=int, default=None, choices=[0, 1],
                    help="Display labels for one team (0 or 1). Overrides calibration.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory for output video + JSON (default: app_data/output)")
    ap.add_argument("--team-tracks", type=str, default=None,
                    help="Comma-separated track IDs to keep. Run without filter first to see IDs.")
    args = ap.parse_args()

    # Default model: prefer yolo26l.pt (large COCO, best person detection).
    # Fine-tuned soccana models are used only if explicitly specified.
    if args.model is None:
        # yolo26l.pt is the largest COCO model available — best for small players at distance
        args.model = "yolo26l.pt"
        print(f"Using COCO model: {args.model}")

    team_tracks = set()

    # Load calibration
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        polygon = np.array(cal["pitch_polygon"], dtype=np.int32)
        goals = cal.get("goals", [])
        my_team = cal.get("my_team", None)
        if my_team == "All":
            my_team = None
        elif my_team is not None:
            my_team = int(my_team)
        if args.my_team is not None:
            my_team = args.my_team
        team_tracks = set()
        if args.team_tracks:
            team_tracks = set(int(x.strip()) for x in args.team_tracks.split(",") if x.strip())
            print(f"  Track filter: keeping {len(team_tracks)} track(s): {sorted(team_tracks)}")

        print(f"Loaded calibration: {len(polygon)}-point polygon, {len(goals)} goals")
        mapper = PitchMapper(polygon)
    else:
        print("No calibration provided. Run main.py once to create one.")
        sys.exit(1)

    # Open video for header info
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Can't open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")
    cap.release()

    # Output — default to project's app_data/output so the web app can serve results
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "app_data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(args.video).stem}_processed.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    # Modules — unified YOLO detector at high resolution for small-object detection
    detector = YOLODetector(model_path=args.model, conf=args.threshold, imgsz=args.imgsz)
    ball_detector = BallDetector(detector)
    tracker = PlayerTracker(max_missed=60, proximity_px=200,
                             appearance_threshold=0.45,
                             match_distance_weight=0.4)

    # Gamma correction LUT
    if args.gamma != 1.0:
        inv_gamma = 1.0 / args.gamma
        gamma_table = np.array([(i / 255.0) ** inv_gamma * 255
                                for i in range(256)], dtype="uint8")
    else:
        gamma_table = None

    all_players = {}
    ball_trail = []          # [(x, y, frame, conf), …]
    prev_positions = {}      # track_id -> (cx, cy)
    track_distances = {}     # track_id -> total meters
    frame_idx = 0
    processed = 0
    t_start = time.time()
    team_classifier = TeamClassifier()
    if my_team is not None:
        team_classifier.set_my_team(my_team)

    # ── FIRST PASS: detect + track + collect data ─────────────────────
    print("\nFirst pass — detecting & tracking…")
    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        # Gamma correction
        if gamma_table is not None:
            frame = cv2.LUT(frame, gamma_table)

        # Unified detection — all classes from one YOLO call
        full_dets, pitch_dets = detector.detect_and_filter(frame, polygon=polygon)

        # Note: no bbox expansion here — it makes boxes spill outside the pitch visually.
        # YOLO at 3840px gives tight but accurate boxes.

        # Ball detection: try full-frame YOLO first, then fallback pipeline
        ball_xy = None
        if full_dets is not None:
            ball_from_yolo = detector.get_ball(full_dets)
            if ball_from_yolo is not None and len(ball_from_yolo) > 0:
                best = ball_from_yolo.confidence.argmax()
                bx1, by1, bx2, by2 = ball_from_yolo.xyxy[best]
                b_cx = (bx1 + bx2) / 2.0
                b_cy = (by1 + by2) / 2.0
                b_conf = float(ball_from_yolo.confidence[best])
                ball_xy = (b_cx, b_cy, b_conf)

        # Fallback: BallDetector's crop-based YOLO + motion blob + Kalman pipeline
        # Pass the YOLO ball as a hint so it can skip the expensive pipeline if valid
        ball_fallback = ball_detector.detect(frame, polygon=polygon,
                                              frame_idx=frame_idx,
                                              yolo_ball_xy=ball_xy)
        if ball_fallback is not None:
            ball_xy = ball_fallback

        if pitch_dets is not None and len(pitch_dets) > 0:
            # Filter to players only for tracking (exclude referees from tracker)
            players = detector.get_players(pitch_dets)
            if players is not None and len(players) > 0:
                tracked = tracker.update(players, frame=frame)

                for i in range(len(tracked)):
                    tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                    if len(team_tracks) > 0 and tid not in team_tracks:
                        continue
                    if tid <= 0:
                        continue

                    x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                    conf = tracked.confidence[i]
                    cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else PLAYER
                    cls_name = CLASS_NAMES.get(cls_id, "Player")

                    # Sample jersey colour for team classification
                    team_classifier.sample(frame, tid, (x1, y1, x2, y2), frame_idx)

                    player_key = f"{tid}_{frame_idx}"
                    all_players[player_key] = {
                        "track_id": tid,
                        "frame": frame_idx,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class": cls_name,
                    }

                    # Accumulate distance
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if tid in prev_positions:
                        px, py = prev_positions[tid]
                        dx_px = cx - px
                        dy_px = cy - py
                        if dx_px * dx_px + dy_px * dy_px >= 25:
                            dist = mapper.distance_m(px, py, cx, cy)
                            if dist >= 0.3:
                                track_distances[tid] = track_distances.get(tid, 0.0) + dist
                    prev_positions[tid] = (cx, cy)

        # Ball trail
        if ball_xy is not None:
            cx, cy, conf = ball_xy
            ball_trail.append((cx, cy, frame_idx, conf))

        processed += 1
        if frame_idx % max(total // 20, 30) == 0:
            elapsed_t = time.time() - t_start
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%) — {processed/elapsed_t:.1f} fps")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    print(f"First pass done: {processed} frames in {elapsed:.0f}s ({processed/elapsed:.1f} fps)")

    # ── POST-PROCESSING ───────────────────────────────────────────────
    if args.post_process:
        print("\nPost-processing…")
        pp_stats = PostProcessor().process(all_players)
        for k, v in pp_stats.items():
            if v:
                print(f"  {k}: {v}")

    # ── TEAM CLASSIFICATION ────────────────────────────────────────────
    print("\nClassifying teams…")
    team_classifier.cluster()
    team_labels = team_classifier.all_labels
    # Attach team to each detection
    for det in all_players.values():
        det["team"] = team_labels.get(det["track_id"], "Unknown")

    # Build a lookup: frame -> [detection, ...] for the render pass
    frame_players = defaultdict(list)
    for det in all_players.values():
        frame_players[det["frame"]].append(det)

    # Team colours for rendering
    RENDER_TEAM_COLORS = {
        "My Team": (0, 180, 255),     # orange
        "Team 2": (255, 50, 100),      # pinkish-red
        "Referee": (255, 255, 50),     # cyan/light blue — distinct from both teams
        "Unknown": (200, 200, 200),    # grey
    }

    # ── SECOND PASS: render video ─────────────────────────────────────
    print("\nSecond pass — rendering video…")
    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(out_path, codec, fps / args.skip, (w, h))
    frame_idx = 0
    render_processed = 0
    render_distances = {}
    render_prev_pos = {}
    t_render = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        annotated = frame.copy()

        # Draw pitch polygon
        cv2.polylines(annotated, [polygon.reshape(-1, 1, 2).astype(np.int32)],
                      True, (0, 255, 200), 3)

        # Draw goals (4-point rectangle per goal: BL, BR, TL, TR — top bar at indices 2→3)
        if len(goals) >= 8:
            colours = [(0, 0, 255), (0, 200, 0)]
            for g_idx in range(2):
                pts = np.array([(int(g[0]), int(g[1])) for g in goals[g_idx * 4:(g_idx + 1) * 4]], dtype=np.int32)
                # Full rectangle outline
                cv2.polylines(annotated, [pts.reshape(-1, 1, 2)], True, colours[g_idx], 2)
                # Highlight top bar (TL→TR, indices 2→3)
                cv2.line(annotated, tuple(pts[2]), tuple(pts[3]), colours[g_idx], 4)
                label = f"GOAL {g_idx + 1}"
                mx, my = int((pts[2][0] + pts[3][0]) / 2), min(pts[2][1], pts[3][1]) - 12
                cv2.putText(annotated, label, (mx - 30, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw corrected player boxes
        for det in frame_players.get(frame_idx, []):
            tid = det["track_id"]
            if tid <= 0 or (len(team_tracks) > 0 and tid not in team_tracks):
                continue
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("confidence", 0)
            cls_id = det.get("class_id", PLAYER)
            cls_name = det.get("class", "Player")

            # Use team colour for players, class colour for referees
            team = det.get("team", "Unknown")
            if cls_id == PLAYER:
                colour = RENDER_TEAM_COLORS.get(team, (200, 200, 200))
            else:
                colour = CLASS_COLOURS.get(cls_id, (200, 200, 200))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
            if team == "My Team":
                tag = " [M]"
            elif team == "Team 2":
                tag = " [T2]"
            else:
                tag = ""
            label = f"#{tid}{tag} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)

            # Distance display
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if tid in render_prev_pos:
                px, py = render_prev_pos[tid]
                if (cx - px) ** 2 + (cy - py) ** 2 >= 25:
                    dist = mapper.distance_m(px, py, cx, cy)
                    if dist >= 0.3:
                        render_distances[tid] = render_distances.get(tid, 0.0) + dist
            render_prev_pos[tid] = (cx, cy)

        # Draw ball
        frame_ball = [b for b in ball_trail if b[2] == frame_idx]
        if frame_ball:
            cx_b, cy_b, conf_b = frame_ball[0][0], frame_ball[0][1], frame_ball[0][3]
            cv2.circle(annotated, (int(cx_b), int(cy_b)), 10, (0, 255, 255), 2)
            cv2.circle(annotated, (int(cx_b), int(cy_b)), 6, (0, 200, 255), -1)
            cv2.putText(annotated, f"BALL {conf_b:.2f}",
                        (int(cx_b) + 14, int(cy_b) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        trail_upto = [b for b in ball_trail if b[2] <= frame_idx]
        for j, (tx, ty, *_) in enumerate(trail_upto):
            alpha = j / max(len(trail_upto), 1)
            cv2.circle(annotated, (int(tx), int(ty)), 4,
                       (0, int(150 * alpha + 100), 255), -1)

        # Distance leaderboard
        if render_distances:
            sorted_dists = sorted(render_distances.items(), key=lambda x: -x[1])[:10]
            overlay = annotated.copy()
            bx1, bx2 = w - 220, w - 10
            by1, by2 = 10, 30 + len(sorted_dists) * 20
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, "Distance (m)", (bx1 + 8, by1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            for rank, (tid, d) in enumerate(sorted_dists):
                y = by1 + 36 + rank * 18
                colour = (0, 255, 0) if rank < 5 else (180, 180, 180)
                cv2.putText(annotated, f"#{tid}: {d:.1f}m",
                            (bx1 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)

        cv2.putText(annotated, f"Frame {frame_idx}/{total}",
                    (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        writer.write(annotated)
        render_processed += 1
        frame_idx += 1

    writer.release()
    cap.release()
    render_elapsed = time.time() - t_render
    print(f"Second pass done: {render_processed} frames in {render_elapsed:.0f}s ({render_processed/render_elapsed:.1f} fps)")
    print(f"Output: {out_path}")

    # Re-encode to H.264
    opt_path = out_path.replace("_processed.mp4", "_h264.mp4")
    try:
        import subprocess as _sp
        _sp.run(
            ["ffmpeg", "-y", "-i", out_path,
             "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-maxrate", "50M", "-bufsize", "100M",
             "-movflags", "+faststart",
             opt_path],
            capture_output=True, text=True, timeout=3600,
        )
        os.replace(opt_path, out_path)
        print(f"  Re-encoded to H.264 for browser playback")
    except Exception as _e:
        print(f"  [!] H.264 re-encode skipped: {_e}")

    # Compute stats
    print("\nComputing stats…")
    t_stats = time.time()
    computer = StatsComputer(mapper)
    stats = computer.compute(
        ball_trail, list(all_players.values()), goals_px=goals,
        total_frames=total, fps=fps, frame_skip=args.skip,
        team_labels=team_labels,
    )
    t_elapsed = time.time() - t_stats
    n_goals = len(stats["goals"])
    n_passes = len(stats["passes"])
    poss = stats.get("possession", {}).get("percentage_per_team", {})
    poss_str = ", ".join(f"{t}: {p}%" for t, p in poss.items())
    print(f"  Stats done in {t_elapsed:.1f}s: {n_goals} goal(s), {n_passes} pass(es),"
          f" {len(stats['distances']['per_track_meters'])} tracks with distance"
          f"{' | Possession: ' + poss_str if poss_str else ''}")

    # ── HEATMAP ────────────────────────────────────────────────────────
    print("\nGenerating heatmap…")
    try:
        heatmap_gen = HeatmapGenerator()
        heatmap_img = heatmap_gen.generate(
            list(all_players.values()),
            pitch_polygon=polygon.tolist() if args.calibration else None,
        )
        heatmap_path = Path(out_path).with_suffix(".jpg")
        cv2.imwrite(str(heatmap_path), heatmap_img)
        print(f"  Heatmap saved: {heatmap_path}")
    except Exception as _e:
        print(f"  [!] Heatmap generation skipped: {_e}")
        heatmap_path = None

    # Save JSON
    json_path = Path(out_path).with_suffix(".json")
    data = {
        "video": args.video,
        "output": out_path,
        "track_filter_ids": sorted(team_tracks) if team_tracks else [],
        "calibration": {"pitch_polygon": polygon.tolist(), "goals": goals},
        "detections": list(all_players.values()),
        "ball_trail": ball_trail,
        "stats": stats,
        "team_labels": team_labels,
        "heatmap": str(heatmap_path) if heatmap_path else None,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))
    print(f"Data: {json_path} ({len(all_players)} detections, {len(ball_trail)} ball positions)")


if __name__ == "__main__":
    main()
