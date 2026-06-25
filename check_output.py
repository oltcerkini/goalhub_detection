"""Check recent processing results."""
import json
from pathlib import Path

calibs = list(Path("app_data/calibrations").glob("*.json"))
outputs = list(Path("app_data/output").glob("*.json"))
print("Calibrations:", [c.name for c in calibs])
print("Outputs:", [o.name for o in outputs])
for o in outputs:
    data = json.load(open(o))
    n_dets = len(data.get("detections", []))
    n_ball = len(data.get("ball_trail", []))
    first_frame = min(d["frame"] for d in data.get("detections", [])) if data.get("detections") else "N/A"
    last_frame = max(d["frame"] for d in data.get("detections", [])) if data.get("detections") else "N/A"
    n_tracks = len(set(d["track_id"] for d in data.get("detections", []))) if data.get("detections") else 0
    print(f"  {o.name}: {n_dets} detections, {n_ball} ball positions, {n_tracks} unique tracks, frames {first_frame}-{last_frame}")
