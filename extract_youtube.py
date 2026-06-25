"""Extract 2fps from downloaded YouTube tactical cam matches."""
import cv2
import sys
from pathlib import Path

SRC = Path("youtube_matches")
OUT = Path("semi_supervised_data/youtube_frames")
FPS = 2

videos = sorted(SRC.glob("*.mp4"))
print(f"Found {len(videos)} YouTube videos")

total = 0
for vpath in videos:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  SKIP: {vpath.name[:60]}")
        continue

    vfps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(round(vfps / FPS)))
    safe_name = vpath.stem.replace("｜", "_").replace("|", "_").replace("/", "_")[:80]
    vout = OUT / safe_name
    vout.mkdir(parents=True, exist_ok=True)

    count = 0
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            cv2.imwrite(str(vout / f"yt{count:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            count += 1
        fi += 1

    cap.release()
    total += count
    sz = vpath.stat().st_size / (1024 * 1024)
    print(f"  {safe_name[:50]}: {count} frames ({sz:.0f}MB)")
    sys.stdout.flush()

print(f"\nTotal YouTube frames: {total}")
