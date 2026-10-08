#!/usr/bin/env python3
"""Full-frame ball-annotation sheet (Phase 7 retrain, Test C).

Native scale: the ball is ~12-17 px in the full 3840x2160 frame -> ~3-4 px at
imgsz 1024, matching the inference input. Frames sampled evenly across the
match (negatives included). Same layout as before:

    annotation_fullframe/frames/f{id:06d}.jpg
    annotation_fullframe/manifest.js
    annotation_fullframe/annotate.html
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\PC\goalhub_detection")
os.chdir(r"C:\Users\PC\goalhub_detection")
import cv2

VIDEO = ("equirect_test/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUT = Path("annotation_fullframe")
N = 200


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total / fps
    times = [dur * i / N for i in range(N)]
    for old in (OUT / "frames").glob("*.jpg"):
        old.unlink()

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
                                                   "clip:'barca_fullframe'"))
    print(f"wrote {len(manifest)} full frames -> {OUT/'frames'} "
          f"({dur:.0f}s clip, {dur/N:.1f}s spacing)")
    print(f"  open {OUT/'annotate.html'}, click the ball, Save JSON -> "
          f"rename to annotation_fullframe/ball_gt_fullframe.json")


if __name__ == "__main__":
    main()
