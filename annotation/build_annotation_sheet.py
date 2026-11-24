#!/usr/bin/env python3
"""Build a ball ground-truth annotation sheet from a clip.

Seeds candidate frames from the arm-A whole-pitch ball detector (as a *candidate*
position, NOT a label), stratified by pitch third, plus a control set of frames
where arm A is silent. Also writes the 10-frame dual-annotator subset.

Outputs:
    annotation/frames/f{frame:06d}.jpg   full-resolution frames
    annotation/manifest.js               frame list for annotate.html

Usage:  python annotation/build_annotation_sheet.py
"""
import os, sys, json
from pathlib import Path

sys.path.insert(0, r'C:\Users\PC\goalhub_detection')
os.chdir(r'C:\Users\PC\goalhub_detection')
import cv2, numpy as np
from ultralytics import YOLO

CLIP = 'app_data/uploads/f5bacf2f.mp4'
CAL  = 'app_data/calibrations/f5bacf2f.json'
OUT  = Path('annotation')
TARGET      = 50      # total frames in the sheet
N_PER_THIRD = 14      # ~17 requested; 14 gives room for the silent control
N_SILENT    = 10      # control: frames where arm A finds nothing (recall test)
DUAL_EVERY  = 5       # every 5th selected frame -> ~10 dual-annotated

(OUT / 'frames').mkdir(parents=True, exist_ok=True)
poly = np.array(json.load(open(CAL))['pitch_polygon'], dtype=np.float32)
model = YOLO('ball_detector_yolo26m.engine')
inv = 1/0.85
lut = np.array([(i/255.0)**inv*255 for i in range(256)], dtype=np.uint8)

x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
mx, my = int(w*0.15), int(h*0.15)
x1, y1 = max(0, x-mx), max(0, y-my)

def in_poly(cx, cy):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), True) >= -10

def dets(res, ox, oy):
    out = []; b = res[0].boxes
    if b is None or len(b) == 0: return out
    xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy().astype(int)
    for i in range(len(xy)):
        if cl[i] == 0:
            cx = (xy[i,0]+xy[i,2])/2 + ox; cy = (xy[i,1]+xy[i,3])/2 + oy
            if in_poly(cx, cy): out.append((cx, cy, float(cf[i])))
    return out

# ── Pass 1: arm-A candidates over the whole clip ─────────────────────────
cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cand = {}   # frame -> (cx, cy, conf)
for fidx in range(total):
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.LUT(frame, lut)
    crop = frame[y1:min(frame.shape[0], y+my+h), x1:frame.shape[1]]
    d = dets(model(crop, conf=.05, imgsz=1024, verbose=False), x1, y1)
    if d: cand[fidx] = max(d, key=lambda z: z[2])
cap.release()

# ── Stratify arm-A candidates into pitch thirds (by detection y) ──────────
yv = poly[:, 1]; y0, y1p = float(yv.min()), float(yv.max())
def third_of(cy):
    f = (cy - y0) / max(y1p - y0, 1e-6)
    return 'far' if f < 1/3 else ('mid' if f < 2/3 else 'near')

buckets = {'far': [], 'mid': [], 'near': []}
for fidx, (cx, cy, cf) in sorted(cand.items()):
    buckets[third_of(cy)].append(fidx)

def spread(lst, n):
    if not lst or n <= 0: return []
    if len(lst) <= n: return lst
    step = len(lst) / n
    return [lst[int(i*step)] for i in range(n)]

silent = [f for f in range(total) if f not in cand]

sel = {}   # frame -> (third_guess, candidate_or_None)
for t in ('far', 'mid', 'near'):
    for f in spread(buckets[t], N_PER_THIRD):
        sel[f] = (t, cand[f])
for f in spread(silent, N_SILENT):
    sel.setdefault(f, ('', None))
# Even-spread fill: guarantees far-third *scenes* are represented even though
# arm A never fires there (far=0 candidates). Annotator flags the true third.
if len(sel) < TARGET:
    for f in spread(list(range(total)), total // 4):
        if len(sel) >= TARGET: break
        sel.setdefault(f, (third_of(cand[f][1]) if f in cand else '', cand.get(f)))
selected = sorted(sel.items())

# ── Render frames + manifest ──────────────────────────────────────────────
for old in (OUT / 'frames').glob('*.jpg'):
    old.unlink()
cap = cv2.VideoCapture(CLIP)
manifest = []
for i, (f, (t, c)) in enumerate(selected):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ret, frame = cap.read()
    if not ret: continue
    name = f'f{f:06d}.jpg'
    cv2.imwrite(str(OUT / 'frames' / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    manifest.append({
        'frame': f,
        'candidate': [round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 3)] if c else None,
        'third': t,
        'dual': (i % DUAL_EVERY == 0),
    })
cap.release()

(OUT / 'manifest.js').write_text('const MANIFEST = ' + json.dumps(manifest, indent=1) + ';\n')
print(f'clip {CLIP}: {total} frames, arm-A candidates on {len(cand)}')
print(f'  thirds: far={len(buckets["far"])} mid={len(buckets["mid"])} near={len(buckets["near"])}  silent={len(silent)}')
print(f'  selected {len(manifest)} frames -> {OUT/"frames"}')
print(f'  dual subset: {sum(1 for m in manifest if m["dual"])} frames')
print(f'  open {OUT/"annotate.html"} in a browser (no server needed)')
