"""Full-frame ball detection recall + false-positive check.

Runs ball_detector_yolo26m.engine on the FULL equirect frame (no crop, no
tiling) at conf 0.05, then through the existing temporal validator
(ball_detector.validate_ball_trail). No pipeline module is modified.

    python test_ball_recall.py [--start 70] [--step 6] [--n 120]
"""

import argparse
import os

import cv2
import numpy as np

from ball_detector import validate_ball_trail
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
CONF = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=70.0)
    ap.add_argument("--step", type=int, default=6)      # sample every Nth frame
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    ball = YOLO("ball_detector_yolo26m.engine")
    cap = cv2.VideoCapture(V)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    f0 = int(args.start * fps)

    raw_trail, per_frame = [], {}
    big = []
    for k in range(args.n):
        fidx = f0 + k * args.step
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        r = ball.predict(frame, imgsz=1024, conf=CONF, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        n = 0
        for s, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            raw_trail.append((float(cx), float(cy), fidx, float(s)))
            n += 1
            if max(x2 - x1, y2 - y1) > 100:
                big.append((fidx, float(s), float(x2 - x1), float(y2 - y1)))
        per_frame[fidx] = n
    cap.release()

    n_frames = args.n
    hit = len(per_frame)
    multi = sum(1 for v in per_frame.values() if v >= 2)
    print(f"window: {n_frames} frames, every {args.step}, from {args.start:.0f}s "
          f"({n_frames*args.step/fps:.1f}s span), conf>={CONF}")
    print(f"raw detections: {len(raw_trail)}")
    print(f"frames with >=1 detection: {hit}/{n_frames}  (coverage {hit/n_frames*100:.0f}%)")
    print(f"frames with 0 detections : {n_frames-hit}/{n_frames}  "
          f"(miss rate {(n_frames-hit)/n_frames*100:.0f}%)")
    print(f"frames with >=2 detections: {multi}  "
          f"(multi-detection/FP rate {multi/n_frames*100:.0f}%)")
    conf = np.array([t[3] for t in raw_trail])
    if len(conf):
        print(f"raw conf: min {conf.min():.3f} median {np.median(conf):.3f} max {conf.max():.3f}")

    filt = validate_ball_trail(raw_trail)
    ff = {}
    for t in filt:
        ff[t[2]] = ff.get(t[2], 0) + 1
    print(f"\nafter validate_ball_trail: {len(filt)}/{len(raw_trail)} detections kept "
          f"({len(filt)/max(len(raw_trail),1)*100:.0f}%)")
    print(f"  frames with a surviving detection: {len(ff)}/{n_frames} "
          f"(coverage {len(ff)/n_frames*100:.0f}%)")

    print(f"\nimplausible-size detections (>100px = not a ball): {len(big)}")
    for fidx, s, w, h in big[:8]:
        print(f"  frame {fidx} conf {s:.2f} size {w:.0f}x{h:.0f}")

    # persist for visual check + comparison
    with open("equirect_test/ball_recall_raw.txt", "w") as f:
        for t in raw_trail:
            f.write(f"{t[2]} {t[0]:.1f} {t[1]:.1f} {t[3]:.3f}\n")
    print("\nraw trail -> equirect_test/ball_recall_raw.txt")


if __name__ == "__main__":
    main()
