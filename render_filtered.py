#!/usr/bin/env python3
"""
Re-render a processed video filtered to one team.

Reads the results JSON from a completed GoalHub analysis and re-renders
the video showing only the selected team's player bounding boxes.

Usage:
    python render_filtered.py --results path/to/results.json --team "My Team"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import PLAYER
from stats_computer import PitchMapper

RENDER_TEAM_COLORS = {
    "My Team": (0, 180, 255),     # orange
    "Team 2": (255, 50, 100),      # pinkish-red
    "Referee": (255, 255, 50),     # cyan/light blue
    "Unknown": (200, 200, 200),    # grey
}


def main():
    ap = argparse.ArgumentParser(description="Re-render video filtered to one team")
    ap.add_argument("--results", required=True, help="Path to results JSON from process.py")
    ap.add_argument("--team", required=True, choices=["My Team", "Team 2"],
                    help="Which team's players to show")
    ap.add_argument("--attacking-goal", default=None, choices=["left", "right"],
                    help="Which goal your team is attacking (determines which GK is yours)")
    ap.add_argument("--output-dir", default=None, help="Output directory")
    args = ap.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    video_path = data.get("video")
    if not video_path or not Path(video_path).exists():
        print(f"Source video not found: {video_path}")
        sys.exit(1)

    # Determine which goalkeepers to include
    gk_track_ids_in_data = set(data.get("goalkeeper_ids", []))
    gk_by_goal = data.get("gk_by_goal", {})
    gk_to_include = set()
    if args.attacking_goal and gk_by_goal:
        # GK defends the OPPOSITE goal from where we're attacking
        defending_goal = "right" if args.attacking_goal == "left" else "left"
        our_gk_tid = gk_by_goal.get(defending_goal)
        if our_gk_tid:
            gk_to_include.add(our_gk_tid)
            print(f"  Including our GK: #{our_gk_tid} (defends {defending_goal} goal)")
    else:
        # No attacking direction = show all GKs
        gk_to_include = gk_track_ids_in_data

    # Build frame -> players lookup, filtered to selected team + our GK
    team_tracks = set()
    all_detections = data.get("detections", [])
    frame_players = defaultdict(list)
    for det in all_detections:
        team = det.get("team", "Unknown")
        is_our_gk = det["track_id"] in gk_to_include
        if team == args.team or is_our_gk:
            team_tracks.add(det["track_id"])
            frame_players[det["frame"]].append(det)

    ball_trail = data.get("ball_trail", [])
    calibration = data.get("calibration", {})
    polygon = np.array(calibration.get("pitch_polygon", []), dtype=np.int32)
    goals = calibration.get("goals", [])

    if len(polygon) < 4:
        print("Invalid pitch polygon in results")
        sys.exit(1)

    mapper = PitchMapper(polygon)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Detect skip from the results data (gap between consecutive frames)
    frames_in_data = sorted(frame_players.keys())
    frame_skip = max(1, min((frames_in_data[i+1] - frames_in_data[i] for i in range(len(frames_in_data)-1)), default=1))

    print(f"Re-rendering video: {w}x{h} @ {fps:.1f} fps, {total} frames, skip={frame_skip}")
    print(f"Filter: {args.team} ({len(team_tracks)} players, {len(frame_players)} frames with detections)")

    # Output path
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.results).parent
    stem = Path(video_path).stem
    team_slug = args.team.lower().replace(" ", "_")
    attack_slug = f"_{args.attacking_goal}" if args.attacking_goal else ""
    out_path = str(out_dir / f"{stem}_filtered_{team_slug}{attack_slug}.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(out_path, codec, fps / frame_skip, (w, h))
    frame_idx = 0
    render_processed = 0
    render_distances = {}
    render_prev_pos = {}
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        annotated = frame.copy()

        # Draw pitch polygon
        cv2.polylines(annotated, [polygon.reshape(-1, 1, 2).astype(np.int32)],
                      True, (0, 255, 200), 3)

        # Draw goals
        if len(goals) >= 8:
            colours = [(0, 0, 255), (0, 200, 0)]
            for g_idx in range(2):
                pts = np.array([(int(g[0]), int(g[1])) for g in goals[g_idx * 4:(g_idx + 1) * 4]], dtype=np.int32)
                cv2.polylines(annotated, [pts.reshape(-1, 1, 2)], True, colours[g_idx], 2)
                cv2.line(annotated, tuple(pts[2]), tuple(pts[3]), colours[g_idx], 4)
                label = f"GOAL {g_idx + 1}"
                mx, my = int((pts[2][0] + pts[3][0]) / 2), min(pts[2][1], pts[3][1]) - 12
                cv2.putText(annotated, label, (mx - 30, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw filtered player boxes
        for det in frame_players.get(frame_idx, []):
            tid = det["track_id"]
            if tid <= 0:
                continue
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("confidence", 0)
            cls_id = det.get("class_id", PLAYER)
            cls_name = det.get("class", "Player")

            team = det.get("team", "Unknown")
            is_gk = det.get("goalkeeper", False)
            if is_gk:
                colour = (180, 50, 255)     # purple for all goalkeepers
            elif cls_id == PLAYER:
                colour = RENDER_TEAM_COLORS.get(team, (200, 200, 200))
            else:
                colour = (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
            if is_gk:
                tag = " GK"
            elif team == "My Team":
                tag = " [M]"
            elif team == "Team 2":
                tag = " [T2]"
            else:
                tag = ""
            label = f"#{tid}{tag}"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2)

            # Distance accumulation
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
            bx1, bx2 = w - 340, w - 10
            by1, by2 = 10, 45 + len(sorted_dists) * 28
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, "Distance (m)", (bx1 + 12, by1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            for rank, (tid, d) in enumerate(sorted_dists):
                y = by1 + 48 + rank * 25
                colour_rank = (0, 255, 0) if rank < 5 else (180, 180, 180)
                cv2.putText(annotated, f"#{tid}: {d:.1f}m",
                            (bx1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour_rank, 1)

        cv2.putText(annotated, f"Frame {frame_idx}/{total}",
                    (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        writer.write(annotated)
        render_processed += 1
        frame_idx += 1

    writer.release()
    cap.release()
    elapsed = time.time() - t_start
    print(f"Done: {render_processed} frames in {elapsed:.0f}s ({render_processed/elapsed:.1f} fps)")
    print(f"Output: {out_path}")

    # Re-encode to H.264 for browser playback
    try:
        h264_path = out_path.replace(".mp4", "_h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path,
             "-c:v", "libx264", "-preset", "slow", "-crf", "18",
             "-maxrate", "50M", "-bufsize", "100M",
             "-movflags", "+faststart",
             h264_path],
            capture_output=True, text=True, timeout=3600,
        )
        os.replace(h264_path, out_path)
        print(f"  Re-encoded to H.264 for browser playback")
    except Exception as e:
        print(f"  [!] H.264 re-encode skipped: {e}")

    # Print the output path so app.py can capture it
    print(f"OUTPUT_PATH:{out_path}")


if __name__ == "__main__":
    main()
