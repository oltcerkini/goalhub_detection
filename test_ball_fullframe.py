"""Does the ball detector work on the whole equirect frame, no tiling?

Runs ball_detector_yolo26m.engine at imgsz=1024 on (a) the full VR180 frame and
(b) the letterbox-cropped content, for 5 frames, and prints detections.
Compare against the FOV-60 tiled numbers from Phase 6.
"""

import cv2
import numpy as np

from spherical_tiling import detect_content_region, apply_region, load_sidecar
from ultralytics import YOLO

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
TIMES = (70.0, 85.0, 100.0, 120.0, 140.0)
CONF = 0.06


def run(model, img, tag, W):
    r = model.predict(img, imgsz=1024, conf=CONF, verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        print(f"    {tag:22s} {img.shape[1]}x{img.shape[0]}: 0 detections")
        return 0
    n = 0
    for s, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = b
        n += 1
        print(f"    {tag:22s} {img.shape[1]}x{img.shape[0]}: conf {s:.3f} "
              f"box {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f} size {x2-x1:.0f}x{y2-y1:.0f}px "
              f"centre ({((x1+x2)/2):.0f},{((y1+y2)/2):.0f})")
    return n


def main():
    sc = load_sidecar(V) or {}
    region = sc.get("crop") or None
    ball = YOLO("ball_detector_yolo26m.engine")
    cap = cv2.VideoCapture(V)
    first = cv2.VideoCapture(V); first.set(cv2.CAP_PROP_POS_MSEC, TIMES[0]*1000)
    _, f0 = first.read(); first.release()
    print(f"full frame {f0.shape[1]}x{f0.shape[0]}; "
          f"content region {region if region else detect_content_region(f0)}")

    tot_full = tot_crop = 0
    for t in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        print(f"\n  --- frame {t:.0f}s ---")
        tot_full += run(ball, frame, "FULL frame", frame.shape[1])
        crop = apply_region(frame, region or detect_content_region(frame))
        tot_crop += run(ball, crop, "CROPPED content", crop.shape[1])
    cap.release()
    print(f"\ntotals: full-frame {tot_full} detections, cropped {tot_crop} "
          f"(vs Phase 6 tiled: 7/3/5 on 70/85/100s)")


if __name__ == "__main__":
    main()
