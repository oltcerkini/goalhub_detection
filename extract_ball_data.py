#!/usr/bin/env python3
"""Extract ball training data from match videos for Phase 3 ball-specific model.

Takes a match video + calibration, runs YOLO on pitch crops, and saves
ball detections as a YOLO-format dataset. The resulting dataset is used
to fine-tune a dedicated ball-detection model.

Usage:
    # Single video
    python extract_ball_data.py --video match.mp4 --calibration cal.json

    # Multiple videos (auto-skip those without calibration)
    python extract_ball_data.py --video-dir youtube_matches_1080p/ --cal-dir app_data/calibrations/

Output:
    ball_dataset/
        images/train/   # pitch crops with ball
        images/val/
        labels/train/   # YOLO-format labels (class 0 = ball)
        labels/val/
        data.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detector import YOLODetector


def main():
    ap = argparse.ArgumentParser(description="Extract ball training data from match videos")
    ap.add_argument("--video", help="Single video file")
    ap.add_argument("--video-dir", help="Directory of match videos")
    ap.add_argument("--calibration", help="Calibration JSON for single video")
    ap.add_argument("--cal-dir", default="app_data/calibrations",
                    help="Directory of calibration files (for --video-dir)")
    ap.add_argument("--output", default="ball_dataset", help="Output dataset directory")
    ap.add_argument("--skip", type=int, default=15,
                    help="Process every Nth frame (lower = more data but slower)")
    ap.add_argument("--min-conf", type=float, default=0.3,
                    help="Minimum ball confidence to accept as label")
    ap.add_argument("--max-per-video", type=int, default=2000,
                    help="Maximum samples per video")
    args = ap.parse_args()

    # ── Collect videos ────────────────────────────────────────────────
    videos = []
    if args.video:
        videos.append(Path(args.video))
    elif args.video_dir:
        videos = sorted(Path(args.video_dir).glob("*.mp4"))
    else:
        print("Provide --video or --video-dir")
        sys.exit(1)

    if not videos:
        print("No videos found")
        sys.exit(1)

    # ── Output structure ──────────────────────────────────────────────
    out_dir = Path(args.output)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Write data.yaml
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        'names: ["ball"]\n'
    )
    print(f"Dataset: {out_dir}", flush=True)
    print(f"  Train: {(out_dir / 'images/train').resolve()}", flush=True)
    print(f"  Val:   {(out_dir / 'images/val').resolve()}", flush=True)

    # ── Detection model (COCO-pretrained for ball detection) ──────────
    detector = YOLODetector(model_path="yolo26l.pt", conf=0.05, imgsz=3840)

    total_samples = 0
    t_start = time.time()

    for video_path in videos:
        if total_samples >= args.max_per_video * len(videos[:1]):
            print(f"\nReached sample limit, stopping.")
            break

        # ── Load calibration matching the video by filename stem ──────────
        cal = None
        polygon = None
        cal_found_name = None
        if args.calibration:
            cal_path = Path(args.calibration)
        elif args.cal_dir:
            cal_dir = Path(args.cal_dir)
            # Match calibration to video by filename stem (both use the same UUID)
            cal_path = cal_dir / f"{video_path.stem}.json"
            if not cal_path.exists():
                print(f"  [!] No matching calibration for {video_path.name}, skipping")
                cap.release()
                continue
        else:
            cal_path = None

        if cal_path and cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            polygon = np.array(cal["pitch_polygon"], dtype=np.int32)
            print(f"\nCalibration: {cal_path.name} ({len(polygon)} pts)")
        else:
            print(f"\nNo calibration — playing {video_path.name} for manual calibration")
            from pitch_calibrator import PitchCalibrator
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"  Cannot read {video_path.name}, skipping")
                continue
            calibrator = PitchCalibrator()
            if not calibrator.calibrate(frame, window="Calibrate — click pitch corners"):
                print(f"  Calibration cancelled for {video_path.name}, skipping")
                continue
            polygon = calibrator.polygon
            print(f"  Calibrated: {len(polygon)}-point polygon")

        if polygon is None or len(polygon) < 4:
            print(f"  Invalid polygon, skipping")
            continue

        # ── Process video ─────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  {video_path.name}: {w}x{h} @ {fps:.1f}fps, {total} frames")

        frame_idx = 0
        video_samples = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.skip != 0:
                frame_idx += 1
                continue

            # Full-frame detection (player + ball + referee)
            full_dets, pitch_dets = detector.detect_and_filter(frame, polygon=polygon)

            # Look for ball in full-frame detections
            ball = None
            if full_dets is not None:
                ball = detector.get_ball(full_dets)
                ball_idx = ball.confidence.argmax() if ball is not None and len(ball) > 0 else -1

            if ball is not None and len(ball) > 0 and ball.confidence[ball_idx] >= args.min_conf:
                cx = (ball.xyxy[ball_idx][0] + ball.xyxy[ball_idx][2]) / 2
                cy = (ball.xyxy[ball_idx][1] + ball.xyxy[ball_idx][3]) / 2
                bw = ball.xyxy[ball_idx][2] - ball.xyxy[ball_idx][0]
                bh = ball.xyxy[ball_idx][3] - ball.xyxy[ball_idx][1]

                # Crop to pitch (same as ball_detector.py's _crop_to_pitch)
                poly = polygon.astype(np.int32)
                bx, by, bw_r, bh_r = cv2.boundingRect(poly)
                margin_x, margin_y = int(bw_r * 0.15), int(bh_r * 0.15)
                x1 = max(0, bx - margin_x)
                y1 = max(0, by - margin_y)
                x2 = min(frame.shape[1], bx + bw_r + margin_x)
                y2 = min(frame.shape[0], by + bh_r + margin_y)
                crop = frame[y1:y2, x1:x2].copy()

                if crop.size == 0:
                    frame_idx += 1
                    continue

                # Ball position in crop coordinates → YOLO format
                bcx = (cx - x1) / crop.shape[1]
                bcy = (cy - y1) / crop.shape[0]
                bw_n = bw / crop.shape[1]
                bh_n = bh / crop.shape[0]

                # Clamp to [0, 1]
                bcx = max(0.001, min(0.999, bcx))
                bcy = max(0.001, min(0.999, bcy))
                bw_n = max(0.001, min(0.999, bw_n))
                bh_n = max(0.001, min(0.999, bh_n))

                # Split: every 10th sample goes to val
                split = "val" if video_samples % 10 == 0 else "train"
                fname = f"{video_path.stem.replace(' ', '_')[:30]}_f{frame_idx:06d}"

                img_path = out_dir / "images" / split / f"{fname}.jpg"
                lbl_path = out_dir / "labels" / split / f"{fname}.txt"

                cv2.imwrite(str(img_path), crop)
                with open(lbl_path, "w") as f:
                    f.write(f"0 {bcx:.6f} {bcy:.6f} {bw_n:.6f} {bh_n:.6f}\n")

                total_samples += 1
                video_samples += 1

            frame_idx += 1

            if video_samples >= args.max_per_video:
                break

        cap.release()
        print(f"  → {video_samples} samples from {video_path.name}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {total_samples} samples extracted in {elapsed:.0f}s")

    # Print counts
    for split in ["train", "val"]:
        n = len(list((out_dir / "images" / split).glob("*.jpg")))
        print(f"  {split}: {n} images")
    print(f"  data.yaml: {yaml_path}")
    print(f"\nNext step: python train_ball_detector.py")


if __name__ == "__main__":
    main()
