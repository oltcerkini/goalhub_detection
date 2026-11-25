#!/usr/bin/env python3
"""Goal-focused ball-annotation sheet (fine-tune for goal-region detection).

Native full 3840x2160 frames at inference scale. Positives: dense frames around
the known right-goal goal (21:25 / 1285 s). Negatives/hard cases: keeper-in-goal
and net/post frames sampled at other high-confidence goal-area events found by
the full-clip scan.

    annotation_goals/frames/f{id:06d}.jpg
    annotation_goals/manifest.js
    annotation_goals/annotate.html
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\PC\goalhub_detection")
os.chdir(r"C:\Users\PC\goalhub_detection")
import cv2

VIDEO = ("assets/180Videos/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUT = Path("annotation_goals")

# small, quick set: 12 frames across the known goal + 6 hard negatives
GOAL_C = 1285.0
times = [GOAL_C - 12.0 + 2.0 * k for k in range(12)]   # 1273..1295, every 2 s
EVENTS = [581, 936, 1223, 1558, 2047, 2175]            # keeper/net negatives
times += EVENTS


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "frames").glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(VIDEO)
    manifest = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(OUT / "frames" / f"f{i:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        manifest.append({"frame": i, "candidate": None, "third": "",
                         "dual": (i % 5 == 0), "time_s": round(t, 1)})
    cap.release()
    (OUT / "manifest.js").write_text(
        "const MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")
    src = Path("annotation/annotate.html").read_text()
    (OUT / "annotate.html").write_text(src.replace("clip:'f5bacf2f'",
                                                   "clip:'barca_goals'"))
    print(f"wrote {len(manifest)} frames -> {OUT/'frames'}")
    print(f"  open {OUT/'annotate.html'} in a browser; click the ball in each")
    print(f"  frame that has one (leave empty for negatives); Save -> rename")
    print(f"  ball_gt.json to {OUT/'ball_gt_goals.json'}")


if __name__ == "__main__":
    main()
