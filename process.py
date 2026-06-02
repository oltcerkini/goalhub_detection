#!/usr/bin/env python3
"""
GoalHub Process — process a video with a saved calibration (no GUI).

Shows all detected players. Use --team-tracks to filter to specific track IDs.

Usage:
    python process.py assets/1.mp4 --calibration calibration.json --threshold 0.30 --skip 3

First time: calibrate once with main.py. After processing, check the output
video and note the track IDs. Filter with --team-tracks:

    python process.py assets/1.mp4 --calibration calibration.json --team-tracks 2,4,8,11
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import PlayerDetector
from team_classifier import TeamClassifier
from ball_detector import BallDetector

from player_tracker import PlayerTracker
from stats_computer import PitchMapper, StatsComputer


def main():
    ap = argparse.ArgumentParser(description="Process video with saved calibration")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--calibration", default=None,
                    help="Calibration JSON (pitch polygon + goals)")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--resize", type=int, default=2560)
    ap.add_argument("--my-team", type=int, default=None, choices=[0, 1],
                    help="Display labels for one team (0 or 1). Overrides calibration.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory for output video + JSON (default: same dir as input)")
    ap.add_argument("--team-tracks", type=str, default=None,
                    help="Comma-separated track IDs to keep (bypasses colour classifier). "
                         "Run without filter first to see track IDs.")
    args = ap.parse_args()

    team_tracks = set()  # always defined for the loop below

    # Load calibration
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        polygon = np.array(cal["pitch_polygon"], dtype=np.int32)
        goals = cal.get("goals", [])
        my_team = cal.get("my_team", None)
        if my_team == "All":
            my_team = None
            print("  No team selected in calibration — showing all players.")
        elif my_team is not None:
            my_team = int(my_team)
            if args.my_team is None:
                print(f"  Team preference: Team {my_team + 1} (from calibration)")
        # --my-team CLI arg overrides calibration
        if args.my_team is not None:
            my_team = args.my_team
            print(f"  Team filter: Team {my_team + 1} (from --my-team arg)")
        # Parse --team-tracks into a set of track IDs (bypasses colour classifier)
        team_tracks = set()
        if args.team_tracks:
            team_tracks = set(int(x.strip()) for x in args.team_tracks.split(",") if x.strip())
            my_team = None  # colour filter is bypassed, just show these tracks
            print(f"  Track filter: keeping {len(team_tracks)} track(s): {sorted(team_tracks)}")

        print(f"Loaded calibration: {len(polygon)}-point polygon, {len(goals)} goals")
        mapper = PitchMapper(polygon)

        # Compute pitch center (intersection of diagonals ≈ center spot)
        poly_pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        pitch_center = poly_pts.mean(axis=0)  # (cx, cy)
        center_radius = max(poly_pts[:, 0].max() - poly_pts[:, 0].min(),
                           poly_pts[:, 1].max() - poly_pts[:, 1].min()) * 0.08
        print(f"  Pitch center: ({int(pitch_center[0])}, {int(pitch_center[1])}) "
              f"(radius {int(center_radius)}px)")
    else:
        print("No calibration provided. Run main.py once to create one.")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Can't open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f} fps, {total} frames")

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.video).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(args.video).stem}_processed.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, codec, fps / args.skip, (w, h))

    # Modules
    detector = PlayerDetector(model_size=args.model)
    classifier = TeamClassifier()
    center_tuple = (int(pitch_center[0]), int(pitch_center[1]), int(center_radius)) if args.calibration else None
    ball_detector = BallDetector(center_prior=center_tuple)
    tracker = PlayerTracker(max_missed=20, proximity_px=120)

    all_players = {}
    ball_trail = []          # [(x, y, frame, conf), …]
    prev_positions = {}      # track_id -> (cx, cy)
    track_distances = {}     # track_id -> total meters
    frame_idx = 0
    processed = 0
    t_start = time.time()

    print("\nProcessing…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        _, inside = detector.detect_and_filter_by_polygon(
            frame, polygon, threshold=args.threshold,
            resize_long_side=args.resize if args.resize > 0 else None,
        )
        ball_xy = ball_detector.detect(frame, polygon=polygon, frame_idx=frame_idx)

        annotated = frame.copy()
        cv2.polylines(annotated, [polygon.reshape(-1, 1, 2).astype(np.int32)],
                      True, (0, 255, 200), 3)
        # Draw goals as lines between post pairs (indices 0-1 = left goal, 2-3 = right goal)
        if len(goals) >= 4:
            colours = [(0, 0, 255), (0, 200, 0)]  # red for left goal, green for right
            for g_idx in range(2):
                i = g_idx * 2
                x1, y1 = int(goals[i][0]), int(goals[i][1])
                x2, y2 = int(goals[i+1][0]), int(goals[i+1][1])
                cv2.line(annotated, (x1, y1), (x2, y2), colours[g_idx], 4)
                # Post markers
                cv2.circle(annotated, (x1, y1), 8, colours[g_idx], -1)
                cv2.circle(annotated, (x1, y1), 8, (255, 255, 255), 2)
                cv2.circle(annotated, (x2, y2), 8, colours[g_idx], -1)
                cv2.circle(annotated, (x2, y2), 8, (255, 255, 255), 2)
                # "GOAL" label above the line
                label = f"GOAL {g_idx + 1}"
                mx, my = (x1 + x2) // 2, min(y1, y2) - 12
                cv2.putText(annotated, label, (mx - 30, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        elif len(goals) >= 2:
            for gx, gy in goals:
                cv2.circle(annotated, (int(gx), int(gy)), 10, (0, 255, 255), -1)
                cv2.circle(annotated, (int(gx), int(gy)), 10, (255, 255, 255), 2)

        if inside is not None and len(inside) > 0:
            # Track across frames for consistent IDs
            tracked = tracker.update(inside)

            # Run colour classifier (trains once, then uses cached centres)
            classifier.classify_frame(frame, tracked)
            if my_team is not None:
                classifier.set_my_team(my_team)

            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1

                # Skip if filtering by track ID list
                if len(team_tracks) > 0 and tid not in team_tracks:
                    continue

                # Update per-track team cache (majority vote across frames)
                if tid > 0:
                    classifier.update_track_vote(tid, i)

                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                conf = tracked.confidence[i]

                team_name = classifier.get_track_team_name(tid) if tid > 0 else \
                    (classifier.get_team_name(i) if (classifier.labels and i in classifier.labels) else "?")
                if team_name == "My Team":
                    colour = (0, 255, 0)       # bright green
                    thickness = 3
                elif team_name == "Unknown":
                    colour = (0, 255, 255)      # yellow
                    thickness = 3
                else:
                    colour = (255, 0, 100)      # hot pink / magenta (very visible)
                    thickness = 3
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
                label = f"#{tid} {conf:.2f}"
                cv2.putText(annotated, label,
                            (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)

                player_key = f"{tid}_{frame_idx}" if tid > 0 else f"raw_{frame_idx}_{i}"
                all_players[player_key] = {
                    "track_id": tid,
                    "frame": frame_idx,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "team": team_name,
                }

                # Accumulate covered distance (skip jitter < 5px and < 0.3m)
                if tid > 0:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if tid in prev_positions:
                        px, py = prev_positions[tid]
                        dx_px = cx - px
                        dy_px = cy - py
                        if dx_px * dx_px + dy_px * dy_px >= 25:  # >= 5px movement
                            dist = mapper.distance_m(px, py, cx, cy)
                            if dist >= 0.3:
                                track_distances[tid] = track_distances.get(tid, 0.0) + dist
                    prev_positions[tid] = (cx, cy)

        # Ball
        if ball_xy is not None:
            cx, cy, conf = ball_xy
            ball_trail.append((cx, cy, frame_idx, conf))
            # Bigger ball circle with white outline for visibility
            cv2.circle(annotated, (int(cx), int(cy)), 10, (0, 255, 255), 2)  # yellow outline
            cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 200, 255), -1)  # filled orange
            cv2.putText(annotated, f"BALL {conf:.2f}",
                        (int(cx) + 14, int(cy) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            # Trail — thicker, brighter
            for j, (tx, ty) in enumerate(ball_detector.trail):
                alpha = j / len(ball_detector.trail)
                cv2.circle(annotated, (int(tx), int(ty)), 4,
                           (0, int(150 * alpha + 100), 255), -1)

        # Distance leaderboard (top-right)
        if track_distances:
            sorted_dists = sorted(track_distances.items(), key=lambda x: -x[1])[:10]
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
        processed += 1

        if frame_idx % max(total // 20, 30) == 0:
            elapsed = time.time() - t_start
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%) — {processed/elapsed:.1f} fps")

        frame_idx += 1

    writer.release()
    cap.release()
    elapsed = time.time() - t_start
    print(f"\nDone: {processed} frames in {elapsed:.0f}s ({processed/elapsed:.1f} fps)")
    print(f"Output: {out_path}")

    # Compute stats
    print("\nComputing stats…")
    t_stats = time.time()
    computer = StatsComputer(mapper)
    stats = computer.compute(
        ball_trail, list(all_players.values()), goals_px=goals,
        total_frames=total, fps=fps, frame_skip=args.skip,
    )
    t_elapsed = time.time() - t_stats
    n_goals = len(stats["goals"])
    n_passes = len(stats["passes"])
    print(f"  Stats done in {t_elapsed:.1f}s: {n_goals} goal(s), {n_passes} pass(es),"
          f" {len(stats['distances']['per_track_meters'])} tracks with distance")

    # Save JSON
    json_path = Path(out_path).with_suffix(".json")
    data = {
        "video": args.video,
        "output": out_path,
        "calibration_team": my_team,
        "track_filter_ids": sorted(team_tracks) if team_tracks else [],
        "calibration": {"pitch_polygon": polygon.tolist(), "goals": goals},
        "detections": list(all_players.values()),
        "ball_trail": ball_trail,
        "stats": stats,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data: {json_path} ({len(all_players)} detections, {len(ball_trail)} ball positions)")


if __name__ == "__main__":
    main()
