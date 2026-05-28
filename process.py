#!/usr/bin/env python3
"""
GoalHub Process — process a video with a saved calibration (no GUI).

Usage:
    python process.py assets/1.mp4 --calibration calibration.json --threshold 0.4 --skip 3

First time: calibrate once with main.py, then reuse the saved JSON.
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


def main():
    ap = argparse.ArgumentParser(description="Process video with saved calibration")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--calibration", default=None,
                    help="Calibration JSON (pitch polygon + goals)")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--resize", type=int, default=1280)
    args = ap.parse_args()

    # Load calibration
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        polygon = np.array(cal["pitch_polygon"], dtype=np.int32)
        goals = cal.get("goals", [])
        my_team = cal.get("my_team", None)
        print(f"Loaded calibration: {len(polygon)}-point polygon, {len(goals)} goals")
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
    out_path = str(Path(args.video).parent / f"{Path(args.video).stem}_processed.mp4")
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, codec, fps / args.skip, (w, h))

    # Modules
    detector = PlayerDetector(model_size=args.model)
    classifier = TeamClassifier()
    ball_detector = BallDetector()

    all_players = {}
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
        ball_xy = ball_detector.detect(frame, polygon=polygon)

        annotated = frame.copy()
        cv2.polylines(annotated, [polygon.reshape(-1, 1, 2).astype(np.int32)],
                      True, (0, 200, 200), 2)
        for gx, gy in goals:
            cv2.circle(annotated, (int(gx), int(gy)), 8, (0, 255, 255), 2)

        if inside is not None and len(inside) > 0:
            classifier.classify(frame, inside)
            for i in range(len(inside)):
                x1, y1, x2, y2 = map(int, inside.xyxy[i])
                conf = inside.confidence[i]
                team = classifier.get_team_name(i) if (classifier.labels and i in classifier.labels) else "?"

                colour = (0, 255, 0) if team == "My Team" else (100, 100, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(annotated, f"{team[:5]} {conf:.2f}",
                            (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

                # Track by approximate spatial ID
                player_key = f"{frame_idx}_{i}"
                all_players[player_key] = {
                    "frame": frame_idx,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "team": team,
                }

        # Ball
        if ball_xy is not None:
            cx, cy, conf = ball_xy
            cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 200, 255), -1)
            cv2.putText(annotated, f"ball {conf:.2f}",
                        (int(cx) + 10, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            # Trail
            for j, (tx, ty) in enumerate(ball_detector.trail):
                alpha = j / len(ball_detector.trail)
                cv2.circle(annotated, (int(tx), int(ty)), 3,
                           (0, int(200 * alpha), 255), -1)

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

    # Save JSON
    json_path = Path(out_path).with_suffix(".json")
    data = {
        "video": args.video,
        "output": out_path,
        "calibration": {"pitch_polygon": polygon.tolist(), "goals": goals},
        "detections": list(all_players.values()),
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data: {json_path} ({len(all_players)} detections)")


if __name__ == "__main__":
    main()
