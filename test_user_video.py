"""Compare R6 vs R3 on the user's actual video frame."""
from ultralytics import YOLO
import cv2
import json

# Load the user's calibration to get the polygon
cal = json.load(open("app_data/calibrations/50946471.json"))
polygon = cal["pitch_polygon"]

# Grab a frame from the user's video
cap = cv2.VideoCapture("app_data/uploads/50946471.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
print(f"User video frame: {w}x{h}")

# R6
r6 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")
r6_r = r6(frame, conf=0.25, verbose=False)
r6_b = r6_r[0].boxes
r6_players = sum(1 for c in r6_b.cls if int(c) == 0) if r6_b is not None else 0
r6_ball = sum(1 for c in r6_b.cls if int(c) == 1) if r6_b is not None else 0
r6_refs = sum(1 for c in r6_b.cls if int(c) == 2) if r6_b is not None else 0
print(f"R6: {len(r6_b)} total, {r6_players} players, {r6_ball} ball, {r6_refs} refs")

# Also check inside-polygon counts
import numpy as np
import supervision as sv
if r6_b is not None:
    dets = sv.Detections(
        xyxy=r6_b.xyxy.cpu().numpy(),
        confidence=r6_b.conf.cpu().numpy(),
        class_id=r6_b.cls.cpu().numpy().astype(int),
    )
    feet = np.column_stack([
        (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2,
        dets.xyxy[:, 3],
    ])
    poly_np = np.array(polygon, dtype=np.int32)
    inside = np.array([
        cv2.pointPolygonTest(poly_np, (float(c[0]), float(c[1])), True) >= -60
        for c in feet
    ])
    inside_players = sum(inside & (dets.class_id == 0))
    print(f"R6 players inside polygon: {inside_players}")

# R3
r3 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")
r3_r = r3(frame, conf=0.25, verbose=False)
r3_b = r3_r[0].boxes
r3_players = sum(1 for c in r3_b.cls if int(c) == 0) if r3_b is not None else 0
r3_ball = sum(1 for c in r3_b.cls if int(c) == 1) if r3_b is not None else 0
r3_refs = sum(1 for c in r3_b.cls if int(c) == 2) if r3_b is not None else 0
print(f"R3: {len(r3_b)} total, {r3_players} players, {r3_ball} ball, {r3_refs} refs")

if r3_b is not None:
    dets3 = sv.Detections(
        xyxy=r3_b.xyxy.cpu().numpy(),
        confidence=r3_b.conf.cpu().numpy(),
        class_id=r3_b.cls.cpu().numpy().astype(int),
    )
    feet3 = np.column_stack([
        (dets3.xyxy[:, 0] + dets3.xyxy[:, 2]) / 2,
        dets3.xyxy[:, 3],
    ])
    inside3 = np.array([
        cv2.pointPolygonTest(poly_np, (float(c[0]), float(c[1])), True) >= -60
        for c in feet3
    ])
    inside3_players = sum(inside3 & (dets3.class_id == 0))
    print(f"R3 players inside polygon: {inside3_players}")
