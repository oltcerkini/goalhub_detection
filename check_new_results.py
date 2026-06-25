"""Check the latest processing results."""
import json
import cv2
from ultralytics import YOLO
from pathlib import Path

for vid in ["662ae53f", "707fe1ca", "50946471"]:
    res_path = Path(f"app_data/output/{vid}_processed.json")
    if not res_path.exists():
        print(f"\n{vid}: no results")
        continue
    data = json.load(open(res_path))
    n_dets = len(data.get("detections", []))
    n_ball = len(data.get("ball_trail", []))
    tracks = set(d["track_id"] for d in data.get("detections", [])) if data.get("detections") else set()
    teams = set(d.get("team", "") for d in data.get("detections", [])) if data.get("detections") else set()
    n_frames = len(set(d["frame"] for d in data.get("detections", []))) if data.get("detections") else 0
    print(f"\n{vid}:")
    print(f"  Detections: {n_dets}, Ball: {n_ball}, Tracks: {len(tracks)}, Frames: {n_frames}")
    print(f"  Teams: {teams}")
    if tracks:
        print(f"  Track IDs: {sorted(tracks)[:20]}...")

    # Also check raw detection on a frame
    vid_path = Path(f"app_data/uploads/{vid}.mp4")
    if vid_path.exists():
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  Video: {w}x{h}, {total} frames")

        # Compare with R3 as well
        r6 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r6/weights/best.pt")
        r3 = YOLO("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")

        cap = cv2.VideoCapture(str(vid_path))
        max_r6, max_r3 = 0, 0
        for f in range(0, total, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
            r6_r = r6(frame, conf=0.2, verbose=False, imgsz=1472)
            r3_r = r3(frame, conf=0.2, verbose=False, imgsz=1472)

            r6_p = sum(1 for c in r6_r[0].boxes.cls if int(c) == 0) if r6_r[0].boxes else 0
            r3_p = sum(1 for c in r3_r[0].boxes.cls if int(c) == 0) if r3_r[0].boxes else 0
            max_r6 = max(max_r6, r6_p)
            max_r3 = max(max_r3, r3_p)
        cap.release()
        print(f"  Max R6 players per frame: {max_r6}")
        print(f"  Max R3 players per frame: {max_r3}")
