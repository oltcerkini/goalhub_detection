"""Test R6 at different inference resolutions vs R3."""
from ultralytics import YOLO
import cv2

r6 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")
r3 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")

cap = cv2.VideoCapture("app_data/uploads/662ae53f.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Comparing R6 at 1024 vs 1472 vs R3 at 1472...")
for f in range(0, total, 30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret:
        break

    r6_1024 = r6(frame, conf=0.2, verbose=False, imgsz=1024)
    r6_1472 = r6(frame, conf=0.2, verbose=False, imgsz=1472)
    r3_1472 = r3(frame, conf=0.2, verbose=False, imgsz=1472)

    r6_1024_p = sum(1 for c in r6_1024[0].boxes.cls if int(c) == 0) if r6_1024[0].boxes else 0
    r6_1472_p = sum(1 for c in r6_1472[0].boxes.cls if int(c) == 0) if r6_1472[0].boxes else 0
    r3_1472_p = sum(1 for c in r3_1472[0].boxes.cls if int(c) == 0) if r3_1472[0].boxes else 0

    print(f"Frame {f}: R6@1024={r6_1024_p:2d}  R6@1472={r6_1472_p:2d}  R3@1472={r3_1472_p:2d}")

cap.release()
