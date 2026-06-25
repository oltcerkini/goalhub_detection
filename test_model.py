"""Quick test comparing R6 vs R3 detections on a frame."""
from ultralytics import YOLO
import cv2

# Load models
r6 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")
r3 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")

# Grab a frame
cap = cv2.VideoCapture("assets/1_annotated.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

def count(results):
    boxes = results[0].boxes
    if boxes is None:
        return 0, 0, 0
    players = sum(1 for c in boxes.cls if int(c) == 0)
    ball = sum(1 for c in boxes.cls if int(c) == 1)
    refs = sum(1 for c in boxes.cls if int(c) == 2)
    return len(boxes), players, ball, refs

# R6 at various conf thresholds
for conf in [0.25, 0.1, 0.05]:
    r = r6(frame, conf=conf, verbose=False)
    total, pl, ba, rf = count(r)
    print(f"R6 @ conf={conf}: {total} total, {pl} players, {ba} ball, {rf} refs")

# R3 at various conf thresholds
for conf in [0.25, 0.1, 0.05]:
    r = r3(frame, conf=conf, verbose=False)
    total, pl, ba, rf = count(r)
    print(f"R3 @ conf={conf}: {total} total, {pl} players, {ba} ball, {rf} refs")

print(f"R6 names: {r6.names}")
print(f"R3 names: {r3.names}")
print(f"R6 imgsz: {r6.model.args.get('imgsz', 'default')}")
