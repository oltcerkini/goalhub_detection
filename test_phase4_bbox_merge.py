"""Phase 4 — exact bbox projection (4.1), merge verification (4.2),
FOV recommendation inputs (4.3). New-module usage only.

    python test_phase4_bbox_merge.py [--video PATH] [--time 85]
"""

import argparse
import os

import cv2
import numpy as np
import supervision as sv

from spherical_tiling import (
    detect_content_region, apply_region, build_grid, tile_equirect,
    merge_to_equirect, load_sidecar, _iou, GRID_180,
)
from player_tracker import PlayerTracker

VIDEO = ("equirect_test/Barca Academy Austin 04B Blue vs "
         "FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv")
OUTDIR = "equirect_test"
CONF = 0.10
EXPECTED_PLAYERS = 20        # visible on the pitch, frame 85s (~18-22)


def grab(video, t, region=None):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, raw = cap.read(); cap.release()
    assert ok
    return apply_region(raw, region or detect_content_region(raw))


def make_model():
    from ultralytics import YOLO
    return YOLO("yolo26l.pt")


def detect_tiles(model, tiles):
    dets = []
    for img, yaw, pitch, fov, tid in tiles:
        r = model.predict(img, imgsz=1280, conf=CONF, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for c, s, b in zip(cls, conf, xyxy):
            if c == 0:
                dets.append({"tile_id": tid, "yaw": yaw, "pitch": pitch, "fov": fov,
                             "bbox": tuple(float(v) for v in b), "conf": float(s), "cls": 0})
    return dets


def bbox_error(mapped):
    """Exact vs linear bbox: width/height % error and IoU, per detection."""
    we, he, ious = [], [], []
    for m in mapped:
        a, b = np.array(m["bbox"]), np.array(m["bbox_linear"])
        wa, ha = a[2] - a[0], a[3] - a[1]
        wb, hb = b[2] - b[0], b[3] - b[1]
        we.append(abs(wb - wa) / max(wa, 1e-6) * 100)
        he.append(abs(hb - ha) / max(ha, 1e-6) * 100)
        ious.append(_iou(list(a), list(b)))
    we, he, ious = np.array(we), np.array(he), np.array(ious)
    return we, he, ious


def merged_centers(merged):
    return np.array([m["center"] for m in merged])


def diff_merges(m_exact, m_lin, tol=25.0):
    """Match kept detections by centre; report how many merges differ."""
    ca, cb = merged_centers(m_exact), merged_centers(m_lin)
    matched = 0
    for p in ca:
        if len(cb) and np.min(np.hypot(*(cb - p).T)) < tol:
            matched += 1
    return matched, len(ca), len(cb)


def run(frame, span, fov):
    model = make_model()
    tiles = tile_equirect(frame, build_grid(span, fov), span)
    dets = detect_tiles(model, tiles)
    merged_e, mapped = merge_to_equirect(dets, span, frame.shape[1], frame.shape[0],
                                         0.5, bbox_key="bbox")
    merged_l, _ = merge_to_equirect(dets, span, frame.shape[1], frame.shape[0],
                                    0.5, bbox_key="bbox_linear")
    return dets, merged_e, merged_l, mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--time", type=float, default=85.0)
    ap.add_argument("--span", type=float, default=180.0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    sc = load_sidecar(args.video) or {}
    span = float(sc.get("span", args.span))
    frame = grab(args.video, args.time, sc.get("crop"))
    eq_h, eq_w = frame.shape[:2]

    print("=" * 70)
    print("4.1  EXACT vs LINEAR bbox projection (FOV 60 and 40)")
    for fov in (60.0, 40.0):
        dets, me, ml, mapped = run(frame, span, fov)
        we, he, ious = bbox_error(mapped)
        print(f"\n  FOV {fov:.0f}: {len(dets)} raw, {len(me)} merged")
        print(f"    width err  : median {np.median(we):5.1f}%  max {we.max():6.1f}%")
        print(f"    height err : median {np.median(he):5.1f}%  max {he.max():6.1f}%")
        print(f"    IoU(lin vs exact): median {np.median(ious):.3f}  min {ious.min():.3f}")
        m, na, nb = diff_merges(me, ml)
        print(f"    merge decisions: exact kept {na}, linear kept {nb}, "
              f"matched {m} -> {'UNCHANGED' if na == nb and m == na else 'CHANGED'}")
        if fov == 40.0:
            f40 = dict(dets=dets, merged=me, mapped=mapped)

    print("\n" + "=" * 70)
    print("4.2  MERGE VERIFICATION (FOV 40, frame 85s)")
    merged = f40["merged"]
    print(f"  merged detections: {len(merged)}  (expected ~{EXPECTED_PLAYERS-2}-{EXPECTED_PLAYERS+2})")

    # Suspicious near-duplicates: different detections with IoU just under threshold.
    susp = []
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            iou = _iou(merged[i]["bbox"], merged[j]["bbox"])
            if iou >= 0.25:
                susp.append((iou, merged[i], merged[j]))
    print(f"  suspicious near-duplicate pairs (0.25<=IoU<0.5): {len(susp)}")
    for iou, a, b in sorted(susp, key=lambda x: -x[0])[:6]:
        print(f"    IoU {iou:.2f}  conf {a['conf']:.2f}/{b['conf']:.2f}  "
              f"tiles {a['tile_id']}/{b['tile_id']}  centres "
              f"({a['center'][0]:.0f},{a['center'][1]:.0f})/({b['center'][0]:.0f},{b['center'][1]:.0f})")

    # 4.2 ByteTrack over 30 consecutive frames.
    print("\n  4.2b ByteTrack over 30 frames (times ~83.8-85.0s):")
    tracker = PlayerTracker(max_missed=90, proximity_px=250,
                            appearance_threshold=0.4, match_distance_weight=0.5)
    model = make_model()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    f0 = int(args.time * fps) - 29
    all_ids, per_frame_n = set(), []
    for k in range(30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + k)
        ok, raw = cap.read()
        if not ok:
            break
        fr = apply_region(raw, sc.get("crop")) if sc.get("crop") else \
             apply_region(raw, detect_content_region(raw))
        dets = detect_tiles(model, tile_equirect(fr, build_grid(span, 40.0), span))
        m, _ = merge_to_equirect(dets, span, fr.shape[1], fr.shape[0], 0.5, bbox_key="bbox")
        per_frame_n.append(len(m))
        if m:
            d = sv.Detections(
                xyxy=np.array([x["bbox"] for x in m], dtype=float),
                confidence=np.array([x["conf"] for x in m], dtype=float),
                class_id=np.zeros(len(m), dtype=int))
            t = tracker.update(d, frame=None)
            if t is not None and t.tracker_id is not None:
                all_ids.update(int(i) for i in t.tracker_id)
    cap.release()
    print(f"    merged per frame: mean {np.mean(per_frame_n):.1f} "
          f"(min {min(per_frame_n)} max {max(per_frame_n)})")
    print(f"    unique track IDs: {len(all_ids)}  vs expected ~{EXPECTED_PLAYERS}")
    ratio = len(all_ids) / EXPECTED_PLAYERS
    print(f"    ratio {ratio:.2f}x  -> {'OK (<=1.3x)' if ratio <= 1.3 else 'LEAKING (>1.3x)'}")

    print("\n" + "=" * 70)
    print("4.3  FAR analysis by PLAYER SIZE (corrected: pitch row != distance)")
    mapped = f40["mapped"]
    h = np.array([m["bbox"][3] - m["bbox"][1] for m in mapped])
    c = np.array([m["conf"] for m in mapped])
    for lo, hi, name in ((0, 15, "small"), (15, 30, "medium"), (30, 1e9, "large")):
        mask = (h >= lo) & (h < hi)
        if mask.sum():
            print(f"    equirect box height {name:6s} ({lo}-{hi if hi<1e5 else 'inf'}px): "
                  f"n={int(mask.sum()):3d}  conf p10 {np.percentile(c[mask],10):.3f} "
                  f"p50 {np.percentile(c[mask],50):.3f} p90 {np.percentile(c[mask],90):.3f}")

    # visual for manual false-merge inspection
    vis = frame.copy()
    for x in merged:
        a = [int(v) for v in x["bbox"]]
        cv2.rectangle(vis, (a[0], a[1]), (a[2], a[3]), (0, 0, 255), 3)
        cv2.putText(vis, f'{x["conf"]:.2f}', (a[0], max(6, a[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTDIR, "phase4_merged_fov40.png"), vis)
    print(f"\n  annotated -> {OUTDIR}/phase4_merged_fov40.png")


if __name__ == "__main__":
    main()
