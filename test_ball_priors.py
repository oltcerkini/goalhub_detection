"""Tests A + B for the ball path (no retrain, no module edits).

A: re-tune validate_ball_trail's conf_score reference from the drone regime
   (0.3) to the equirect regime (observed median ~0.11). Reimplemented locally so
   ball_detector.py is untouched.
B: player-proximity prior — drop ball detections farther than N equirect px from
   any merged player detection.

Reuses the saved full-frame ball trail (equirect_test/ball_recall_raw.txt).
"""

import os

import cv2
import numpy as np

from ball_detector import validate_ball_trail as validate_orig
from spherical_tiling import (
    apply_region, build_grid, tile_equirect, merge_to_equirect, load_sidecar,
)

V = ("equirect_test/Barca Academy Austin 04B Blue vs "
     "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
N_FRAMES, STEP, F0 = 120, 6, 1680          # 70s..100s, every 6th frame


def validate_tuned(ball_trail, conf_ref=0.11, min_fragment_length=4,
                   max_gap_frames=8, max_frame_jump_px=300):
    """Same as ball_detector.validate_ball_trail but conf_score is normalised by
    conf_ref instead of the hard-coded 0.3."""
    if not ball_trail:
        return []
    st = sorted(ball_trail, key=lambda t: t[2])
    frags, cur = [], [st[0]]
    for i in range(1, len(st)):
        if st[i][2] - st[i - 1][2] > max_gap_frames:
            frags.append(cur); cur = []
        cur.append(st[i])
    frags.append(cur)

    kept = []
    for frag in frags:
        length = len(frag)
        smooth = sum(1 for i in range(1, length)
                     if np.hypot(frag[i][0] - frag[i - 1][0],
                                 frag[i][1] - frag[i - 1][1]) < max_frame_jump_px)
        smooth_ratio = smooth / max(length - 1, 1)
        avg_conf = sum(t[3] for t in frag) / length
        length_score = min(length / min_fragment_length, 2.0)
        conf_score = min(avg_conf / conf_ref, 1.0)
        comp = length_score * smooth_ratio * conf_score
        if comp >= 1.0 or (length >= min_fragment_length and smooth_ratio >= 0.6):
            kept.extend(frag)
    return kept


def stats(trail, n=N_FRAMES):
    fr = {}
    for t in trail:
        fr[t[2]] = fr.get(t[2], 0) + 1
    cov = len(fr)
    multi = sum(1 for v in fr.values() if v >= 2)
    return cov, cov / n * 100, multi, multi / n * 100


def main():
    sc = load_sidecar(V) or {}
    offx, offy = sc["crop"][0], sc["crop"][1]        # full-frame -> crop coords
    rows = [l.split() for l in open("equirect_test/ball_recall_raw.txt")]
    raw = [(float(x), float(y), int(f), float(c)) for f, x, y, c in rows]
    print(f"raw trail: {len(raw)} detections over {N_FRAMES} frames "
          f"(full-frame coords; crop offset {offx},{offy})\n")

    print("=" * 68)
    print("A  validator re-tuned (conf_ref 0.3 -> 0.11)")
    print(f"{'conf>=':>8} {'raw cov':>8} {'raw mlt':>8} {'orig cov':>9} {'tuned cov':>10} {'tuned mlt':>10}")
    for thr in (0.05, 0.08, 0.10, 0.12):
        s = [t for t in raw if t[3] >= thr]
        _, rc, _, rm = stats(s)
        _, oc, _, _ = stats(validate_orig(s))
        _, tc, _, tm = stats(validate_tuned(s, conf_ref=0.11))
        print(f"{thr:8.2f} {rc:7.0f}% {rm:7.0f}% {oc:8.0f}% {tc:9.0f}% {tm:9.0f}%")

    print("\n" + "=" * 68)
    print("B  player-proximity prior (needs player positions)")
    from ultralytics import YOLO
    pl = YOLO("yolo26l.pt")
    cap = cv2.VideoCapture(V)
    players = {}          # frame -> np.array of (x,y) equirect centres
    for k in range(N_FRAMES):
        fidx = F0 + k * STEP
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        fr = apply_region(frame, sc["crop"])
        dets = []
        for img, yaw, pitch, fov, tid in tile_equirect(fr, build_grid(180, 60.0), 180):
            r = pl.predict(img, imgsz=1280, conf=0.10, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                if c == 0:
                    dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                                 "bbox": tuple(map(float, b)), "conf": float(s), "cls": 0})
        mg, _ = merge_to_equirect(dets, 180, fr.shape[1], fr.shape[0], criterion="center")
        players[fidx] = np.array([m["center"] for m in mg]) if mg else np.zeros((0, 2))
    cap.release()
    import json
    json.dump({str(k): v.tolist() for k, v in players.items()},
              open("equirect_test/players_crop.json", "w"))
    pn = [len(v) for v in players.values()]
    print(f"  player dets/frame: mean {np.mean(pn):.1f} (min {min(pn)} max {max(pn)})")

    def prox(trail, N):
        """Ball detections are full-frame; shift into crop coords to match players."""
        out = []
        for x, y, f, c in trail:
            P = players.get(f)
            if P is None or len(P) == 0:
                continue
            bx, by = x - offx, y - offy
            if np.min(np.hypot(P[:, 0] - bx, P[:, 1] - by)) <= N:
                out.append((x, y, f, c))
        return out

    print(f"\n{'N px':>6} {'cov':>7} {'mlt':>7} {'| +tuned cov':>13} {'tuned mlt':>10}")
    for N in (100, 200, 400):
        fp = prox([t for t in raw if t[3] >= 0.05], N)
        c, cp, m, mp = stats(fp)
        vt = validate_tuned(fp, conf_ref=0.11)
        _, tc, _, tm = stats(vt)
        print(f"{N:6d} {cp:6.0f}% {mp:6.0f}% {tc:12.0f}% {tm:9.0f}%")


if __name__ == "__main__":
    main()
