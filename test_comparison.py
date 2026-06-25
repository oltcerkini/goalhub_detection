"""Compare R6 vs R3 on a proper match video."""
from ultralytics import YOLO
import cv2

r6 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")
r3 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")

# Test on assets/1_annotated.mp4 at multiple frames
cap = cv2.VideoCapture("assets/1_annotated.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

r6_all = 0
r3_all = 0
frames_sampled = 0

for f in range(0, total_frames, 30):  # every 30th frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if not ret:
        break
    frames_sampled += 1

    r6_r = r6(frame, conf=0.25, verbose=False)
    r3_r = r3(frame, conf=0.25, verbose=False)

    r6_n = sum(1 for c in r6_r[0].boxes.cls if int(c) == 0) if r6_r[0].boxes is not None else 0
    r3_n = sum(1 for c in r3_r[0].boxes.cls if int(c) == 0) if r3_r[0].boxes is not None else 0

    r6_all += r6_n
    r3_all += r3_n

    if f < 200:  # show first few
        print(f"  Frame {f}: R6={r6_n} players, R3={r3_n} players")

cap.release()
print(f"\nAverage per frame: R6={r6_all/frames_sampled:.1f}, R3={r3_all/frames_sampled:.1f}")
print(f"Improvement: {((r6_all-r3_all)/r3_all)*100:+.0f}%")
