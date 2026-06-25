"""Extract 1 fps from video files for semi-supervised learning."""
import cv2
import os
from pathlib import Path

ASSETS = Path("assets")
OUTPUT = Path("semi_supervised_data/raw_frames_3fps")
FPS = 3

VIDEOS = [
    "ulpiana.mp4",
    "ulpiana_1min.mp4",
    "ulpiana_15sec.mp4",
    "ulpiana_7sec.mp4",
]


def extract_frames(video_path, output_dir, fps=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  SKIP: cannot open {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))
    if frame_interval < 1:
        frame_interval = 1

    stem = video_path.stem
    out = output_dir / stem
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = f"{stem}_frame{count:06d}.jpg"
            cv2.imwrite(str(out / fname), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            count += 1
        frame_idx += 1

    cap.release()
    print(f"  {video_path.name}: extracted {count} frames ({frame_idx} total frames @ {video_fps:.1f} fps)")
    return count


def main():
    total = 0
    for vname in VIDEOS:
        vpath = ASSETS / vname
        if not vpath.exists():
            print(f"  NOT FOUND: {vpath}")
            continue
        total += extract_frames(vpath, OUTPUT, fps=FPS)
    print(f"\nDone. Total frames extracted: {total}")


if __name__ == "__main__":
    main()
