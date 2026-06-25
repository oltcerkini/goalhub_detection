"""Fix calibration coords and process with R6, then show results."""
import json
import subprocess
import sys
from pathlib import Path

# Fix calibration: scale coords from 1920x1080 to 3840x2160
cal = json.load(open("assets/calibration_2.json"))
scaled = {
    "pitch_polygon": [[x*2, y*2] for x, y in cal["pitch_polygon"]],
    "goals": [[x*2, y*2] for x, y in cal["goals"]],
    "my_team": cal["my_team"],
}
fixed_path = "assets/calibration_fixed.json"
json.dump(scaled, open(fixed_path, "w"))
print(f"Fixed calibration saved. Polygon: {scaled['pitch_polygon']}")

# Process with R6
cmd = [
    sys.executable, "process.py",
    "assets/1_annotated.mp4",
    "--calibration", fixed_path,
    "--threshold", "0.2",
    "--skip", "3",
    "--gamma", "0.85",
    "--output-dir", "app_data/output",
    "--no-post-process",
]
print(f"\nRunning: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# Count results
data = json.load(open("app_data/output/1_annotated_processed.json"))
n_dets = len(data.get("detections", []))
n_ball = len(data.get("ball_trail", []))
tracks = set(d["track_id"] for d in data.get("detections", [])) if data.get("detections") else set()
print(f"\n=== RESULTS ===")
print(f"Detections: {n_dets}")
print(f"Ball positions: {n_ball}")
print(f"Unique tracks: {len(tracks)}")
print(f"Tracks: {sorted(tracks)}")

if data.get("stats"):
    dists = data["stats"].get("distances", {})
    print(f"Distances: {dists}")
