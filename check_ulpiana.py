"""Check ulpiana_7sec for player count with R3 + 1280px inference."""
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")
cap = cv2.VideoCapture("assets/ulpiana_7sec.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h}, {total} frames")

max_p = 0
for f in range(0, total, 15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret:
        break
    r = model(frame, conf=0.2, verbose=False, imgsz=1280)
    boxes = r[0].boxes
    if boxes is not None:
        players = sum(1 for c in boxes.cls if int(c) == 0)
        ball = sum(1 for c in boxes.cls if int(c) == 1)
    else:
        players, ball = 0, 0
    if players > max_p:
        max_p = players
    if f < 60:
        print(f"  Frame {f}: {players} players, {ball} ball")

cap.release()
print(f"\nMax players in any frame: {max_p}")
