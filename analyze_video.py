"""Analyze the user's video frame by frame to see what's in it."""
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")

cap = cv2.VideoCapture("app_data/uploads/50946471.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h}, {fps}fps, {total} frames")

# Sample every 30th frame
max_players = 0
max_frame = 0
frame_counts = []

for f in range(0, total, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, conf=0.2, verbose=False, imgsz=1472)
    boxes = r[0].boxes
    if boxes is not None:
        players = sum(1 for c in boxes.cls if int(c) == 0)
        ball = sum(1 for c in boxes.cls if int(c) == 1)
        refs = sum(1 for c in boxes.cls if int(c) == 2)
    else:
        players, ball, refs = 0, 0, 0

    frame_counts.append((f, players, ball, refs))
    if players > max_players:
        max_players = players
        max_frame = f

cap.release()

print(f"\nFrames sampled: {len(frame_counts)}")
print(f"Max players in any frame: {max_players} (frame {max_frame})")
print(f"Avg players per frame: {sum(p for _, p, _, _ in frame_counts)/len(frame_counts):.1f}")

# Show frames with most players
sorted_frames = sorted(frame_counts, key=lambda x: -x[1])
print(f"\nTop 10 frames by player count:")
for f, p, b, r in sorted_frames[:10]:
    print(f"  Frame {f}: {p} players, {b} ball, {r} refs")
