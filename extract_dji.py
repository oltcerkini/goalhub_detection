"""Extract frames from DJI drone footage of Ulpiana matches."""
import cv2
from pathlib import Path

SRC = Path(r"assets\more ulpiana vids")
OUT = Path("semi_supervised_data/dji_frames")
FPS = 1  # 1fps since these are long high-res videos

videos = sorted(SRC.glob("*.MP4")) + sorted(SRC.glob("*.mp4"))
print(f"Found {len(videos)} DJI videos")

total = 0
for vpath in videos:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  SKIP: {vpath.name}")
        continue

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    interval = max(1, int(round(video_fps / FPS)))
    out = OUT / vpath.stem
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % interval == 0:
            # Save with moderate quality to save space
            cv2.imwrite(str(out / f"dji{count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            count += 1
        f_idx += 1

    cap.release()
    total += count
    print(f"  {vpath.name}: {count} frames ({w}x{h}, {duration/60:.0f}min @ {video_fps:.1f}fps)")

print(f"\nDone. Total DJI frames: {total}")
