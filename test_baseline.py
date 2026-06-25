"""Quick test: Run YOLO26m on ulpiana_7sec and show detection stats."""
import cv2, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import YOLODetector, PLAYER, BALL, REFEREE

video = "assets/ulpiana_7sec.mp4"
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detector = YOLODetector(conf=0.25)

frame_idx = 0
sample_count = 0
player_counts = []
ball_counts = []
ref_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 30 != 0:  # every 30th frame (~0.5s)
        frame_idx += 1
        continue

    dets, _ = detector.detect_and_filter(frame)
    n_players = 0
    n_balls = 0
    n_refs = 0
    if dets is not None:
        p = detector.get_players(dets)
        b = detector.get_ball(dets)
        r = detector.get_referees(dets)
        n_players = len(p) if p is not None else 0
        n_balls = len(b) if b is not None else 0
        n_refs = len(r) if r is not None else 0

    player_counts.append(n_players)
    ball_counts.append(n_balls)
    ref_counts.append(n_refs)
    sample_count += 1
    print(f"  Frame {frame_idx}/{total}: {n_players} players, {n_balls} balls, {n_refs} refs")

    frame_idx += 1

cap.release()

print(f"\n=== Results from {sample_count} sample frames ===")
print(f"Players: avg={sum(player_counts)/sample_count:.1f}, min={min(player_counts)}, max={max(player_counts)}")
print(f"Balls:   avg={sum(ball_counts)/sample_count:.1f}, min={min(ball_counts)}, max={max(ball_counts)}")
print(f"Refs:    avg={sum(ref_counts)/sample_count:.1f}, min={min(ref_counts)}, max={max(ref_counts)}")
