#!/usr/bin/env python3
"""Extract ball training data from YouTube tactical-cam matches WITHOUT calibration.

Unlike extract_ball_data.py which needs pitch calibration + crop, this script
detects balls on the full frame and saves crops around each detection. This
lets us use all 41 YouTube downloads without manual calibration.

Usage:
    python extract_ball_youtube.py --video-dir youtube_matches_1080p/
    python extract_ball_youtube.py --video-dir youtube_matches_1080p/ --max-per-video 10000
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detector import YOLODetector


def main():
    ap = argparse.ArgumentParser(description="Extract ball data from YouTube matches (no calibration)")
    ap.add_argument("--video-dir", default="youtube_matches_1080p",
                    help="Directory of match videos")
    ap.add_argument("--output", default="ball_dataset", help="Output dataset directory")
    ap.add_argument("--skip", type=int, default=30,
                    help="Process every Nth frame (30 = ~1 fps for 30fps video)")
    ap.add_argument("--min-conf", type=float, default=0.06,
                    help="Minimum ball confidence (uses per-class conf with 0.06 default for ball)")
    ap.add_argument("--max-per-video", type=int, default=5000,
                    help="Maximum samples per video")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="Max videos to process (0 = all)")
    ap.add_argument("--crop-size", type=int, default=400,
                    help="Crop size in pixels around ball center")
    args = ap.parse_args()

    out_dir = Path(args.output)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Count existing
    existing = len(list((out_dir / "images" / "train").glob("*.jpg"))) + \
               len(list((out_dir / "images" / "val").glob("*.jpg")))
    print(f"Existing dataset: {existing} images", flush=True)

    # Use COCO yolo26l for extraction (best general ball recall at full resolution)
    detector = YOLODetector(model_path="yolo26l.pt", conf=0.05, imgsz=3840)
    print(f"Detection model: yolo26l.pt at 3840px (COCO class 32 = sports ball)", flush=True)

    videos = sorted(Path(args.video_dir).glob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"Videos to process: {len(videos)}", flush=True)

    total_before = existing
    total_samples = 0
    t_start = time.time()

    for vid_idx, video_path in enumerate(videos):
        if total_samples >= args.max_per_video * len(videos):
            print(f"\nReached global sample limit, stopping.", flush=True)
            break

        # Skip if this video already has data in the dataset
        stem_safe = video_path.stem.replace(" ", "_").replace("|", "").replace("⧸", "")[:40]
        existing_count = len(list((out_dir / "images" / "train").glob(f"yt_{stem_safe}_*.jpg")))
        if existing_count > 0:
            print(f"[{vid_idx+1}/{len(videos)}] {video_path.name}: already has {existing_count} samples, skipping", flush=True)
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total < 100:
            cap.release()
            continue

        print(f"\n[{vid_idx+1}/{len(videos)}] {video_path.name}: {w}x{h} @ {fps:.1f}fps, {total} frames", flush=True)

        frame_idx = 0
        video_samples = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.skip != 0:
                frame_idx += 1
                continue

            # Detect balls using per-class confidence (ball threshold=0.06)
            dets = detector.detect_with_per_class_conf(frame)
            ball_dets = detector.get_ball(dets) if dets is not None else None

            if ball_dets is None or len(ball_dets) == 0:
                frame_idx += 1
                continue

            # Pick highest confidence ball
            best = np.argmax(ball_dets.confidence)
            x1, y1, x2, y2 = ball_dets.xyxy[best]
            conf = float(ball_dets.confidence[best])
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Skip tiny detections
            if bw < 3 or bh < 3:
                frame_idx += 1
                continue

            # Crop a window around ball center
            half = args.crop_size // 2
            rx1 = max(0, int(cx - half))
            ry1 = max(0, int(cy - half))
            rx2 = min(frame.shape[1], int(cx + half))
            ry2 = min(frame.shape[0], int(cy + half))

            if rx2 - rx1 < 50 or ry2 - ry1 < 50:
                frame_idx += 1
                continue

            crop = frame[ry1:ry2, rx1:rx2].copy()
            if crop.size == 0:
                frame_idx += 1
                continue

            # Ball in crop coordinates → YOLO format (normalized)
            bcx = (cx - rx1) / crop.shape[1]
            bcy = (cy - ry1) / crop.shape[0]
            bw_n = bw / crop.shape[1]
            bh_n = bh / crop.shape[0]

            bcx = max(0.001, min(0.999, bcx))
            bcy = max(0.001, min(0.999, bcy))
            bw_n = max(0.001, min(0.999, bw_n))
            bh_n = max(0.001, min(0.999, bh_n))

            split = "val" if video_samples % 10 == 0 else "train"
            stem = video_path.stem.replace(" ", "_").replace("|", "").replace("⧸", "")[:40]
            fname = f"yt_{stem}_f{frame_idx:06d}"

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

            if total_samples % 1000 == 0:
                elapsed = time.time() - t_start
                rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"  ... {total_samples} samples ({rate:.1f}/s)", flush=True)

        cap.release()
        print(f"  → {video_samples} samples from {video_path.name}", flush=True)

    elapsed = time.time() - t_start
    total_now = existing + total_samples
    print(f"\n{'='*60}")
    print(f"Done: {total_samples} samples extracted in {elapsed:.0f}s ({total_samples/elapsed:.1f}/s)")
    print(f"Dataset: {total_before} → {total_now} (+{total_samples})")
    for split in ["train", "val"]:
        n = len(list((out_dir / "images" / split).glob("*.jpg")))
        print(f"  {split}: {n} images")
    print(f"\nNext: python train_ball_detector.py --data ball_dataset/data.yaml --epochs 100 --batch 4 --model ball_detector_yolo26m.pt --name ball_detector_r2")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
