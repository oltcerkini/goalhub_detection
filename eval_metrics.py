"""Evaluation metrics for tiny-object (ball) detection at training scale.

IoU is degenerate at ~2.6 px (any sub-pixel offset collapses it to 0, any
overlap to 1), so we use Normalized Wasserstein Distance instead:

    W^2(A,B) = (dcx)^2 + (dcy)^2 + ((wA-wB)^2 + (hA-hB)^2) / 4
    NWD(A,B) = exp( -sqrt(W^2) / C )

Boxes are treated as axis-aligned Gaussians. NWD in (0, 1]; 1 = identical.

**C is dataset-specific.** The NWD paper's C=12.8 is the mean absolute object
size of AI-TOD. At our evaluation scale the ball is ~2.6 px, so C must be set
to the *measured* median ball size, not the paper default. With C=12.8 the
metric is far too permissive at this scale (2 px error -> NWD 0.86). See
`median_ball_C()` -- call it once the GT exists and pass the result as C.

Boxes are (cx, cy, w, h) in the same pixel space for both args.
"""
import sys
import numpy as np

# Fallback until GT exists; MUST be replaced by a measured value. (report as C)
DEFAULT_C = 2.6


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def w2(a, b):
    acx, acy, aw, ah = a
    bcx, bcy, bw, bh = b
    return (acx - bcx) ** 2 + (acy - bcy) ** 2 + ((aw - bw) ** 2 + (ah - bh) ** 2) / 4.0


def nwd(a, b, C=DEFAULT_C):
    """Normalized Wasserstein Distance between two (cx,cy,w,h) boxes."""
    return float(np.exp(-np.sqrt(w2(a, b)) / C))


def median_ball_C(gt_boxes):
    """C = median absolute ball size (px) at eval scale, per the NWD paper."""
    if not gt_boxes:
        raise ValueError("no GT boxes")
    sizes = [np.sqrt(w * h) for (_, _, w, h) in gt_boxes]
    return float(np.median(sizes))


def match(preds, gts, C=DEFAULT_C, thr=0.5):
    """Greedy highest-NWD-first matching. Returns [(pred_i, gt_j, nwd), ...]."""
    pairs = [(nwd(p, g, C), i, j)
             for i, p in enumerate(preds) for j, g in enumerate(gts)]
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); out.append((i, j, s))
    return out


def precision_recall(preds, gts, C=DEFAULT_C, thr=0.5):
    """Returns (precision, recall, tp, fp, fn). Empty-side => 1.0 by convention."""
    m = match(preds, gts, C, thr)
    tp = len(m)
    fp = len(preds) - tp
    fn = len(gts) - tp
    prec = tp / len(preds) if preds else 1.0
    rec = tp / len(gts) if gts else 1.0
    return prec, rec, tp, fp, fn


def sweep(preds_per_frame, gts_per_frame, C=DEFAULT_C, thresholds=None):
    """Precision/recall curve over thresholds. Frames keyed identically."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    keys = sorted(set(preds_per_frame) | set(gts_per_frame))
    rows = []
    for thr in thresholds:
        TP = FP = FN = 0
        for k in keys:
            p = preds_per_frame.get(k, []); g = gts_per_frame.get(k, [])
            _, _, tp, fp, fn = precision_recall(p, g, C, thr)
            TP += tp; FP += fp; FN += fn
        prec = TP / (TP + FP) if (TP + FP) else 1.0
        rec = TP / (TP + FN) if (TP + FN) else 1.0
        rows.append((thr, prec, rec, TP, FP, FN))
    return rows


def assert_no_training_overlap(eval_stems, dataset_dir="ball_dataset"):
    """Hard fail if any eval clip's stem appears in the training dataset.

    Silent contamination is the failure mode this guards. Raises SystemExit.
    """
    from pathlib import Path
    tr = set()
    d = Path(dataset_dir)
    for s in ("train", "val"):
        for p in (d / "images" / s).glob("*.jpg"):
            tr.add(p.stem.rsplit("_f", 1)[0])
    bad = set(eval_stems) & tr
    if bad:
        raise SystemExit(
            f"FATAL: eval clip(s) present in training set {sorted(bad)} -- "
            f"refusing to run. Pick a truly held-out clip."
        )
    return True


if __name__ == "__main__":
    C = DEFAULT_C
    box = (1000.0, 1500.0, 9.7, 9.7)
    cases = {
        "identical":      (0.0, 9.7, 9.7),
        "0.5 px apart":   (0.5, 9.7, 9.7),
        "1 px apart":     (1.0, 9.7, 9.7),
        "2 px apart":     (2.0, 9.7, 9.7),
        "5 px apart":     (5.0, 9.7, 9.7),
        "20 px apart":    (20.0, 9.7, 9.7),
        "1 px apart, halved size": (1.0, 4.85, 4.85),
    }
    print(f"{'case':26s} {'NWD@C=2.6':>10s} {'NWD@C=12.8':>11s}")
    for name, (d, w, h) in cases.items():
        a = box; b = (box[0] + d, box[1], w, h)
        print(f"{name:26s} {nwd(a,b,2.6):10.3f} {nwd(a,b,12.8):11.3f}")
    print()
    print("the 2 px row is the point: C=12.8 calls a near-miss (0.86) 'good';")
    print("C=2.6 scores it 0.46, borderline at a 0.5 threshold.")
