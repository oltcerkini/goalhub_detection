#!/usr/bin/env python3
"""Solve a VR180 calibration from the pencil-tool trace file.

    python calib/calibrate.py [calib/calib_equirect.json] [-o calib/calibration.json]
    python calib/calibrate.py --synthetic      # self-test, writes a synthetic trace

Reads calib_equirect.json (produced by calibrate.html: traced pitch curves +
the user-supplied pitch dimensions) and writes a versioned calibration contract
with the recovered camera pose and quality metrics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spherical_calib import calibration_from_traces, pitch_geometry, pitch_to_equirect, rotmat  # noqa: E402


def make_synthetic_trace(path, W=3401, H=1586, span=180.0, dims=None, n=25, noise=0.0):
    """Write a calib_equirect.json as if the user had traced the true lines."""
    import numpy as np
    rng = np.random.default_rng(1)
    P = np.array([26.0, 9.5, -58.0])
    rv = np.radians([18.0, 62.0, 7.0])
    geom = pitch_geometry(dims)
    from spherical_calib import _sample_spec
    lines = []
    for lab, spec in geom.items():
        pts = _sample_spec(spec)
        idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
        xs, ys = pitch_to_equirect(pts[idx, 0], pts[idx, 1], {"P": P, "rvec": rv}, W, H, span)
        pl = [[float(np.ravel(xs)[i] + rng.normal(0, noise)),
               float(np.ravel(ys)[i] + rng.normal(0, noise))]
              for i in range(len(np.ravel(xs)))
              if np.isfinite(np.ravel(xs)[i]) and 0 <= np.ravel(xs)[i] < W]
        if len(pl) >= 2:
            lines.append({"label": lab, "points": pl})
    cfg = {"image": "synthetic", "size": [W, H], "span": span, "crop": None,
           "pitch": {**(dims or {}), "length": (dims or {}).get("length", 105.0),
                     "width": (dims or {}).get("width", 68.0)},
           "lines": lines}
    Path(path).write_text(json.dumps(cfg, indent=1))
    return cfg


def main():
    args = [a for a in sys.argv[1:]]
    synthetic = "--synthetic" in args
    args = [a for a in args if a != "--synthetic"]
    out = "calib/calibration.json"
    if "-o" in args:
        i = args.index("-o"); out = args[i + 1]; del args[i:i + 2]
    src = args[0] if args else "calib/calib_equirect.json"

    if synthetic:
        src = "calib/calib_equirect_synthetic.json"
        make_synthetic_trace(src, noise=2.0)
        print(f"wrote synthetic trace {src}")

    cfg = calibration_from_traces(src)
    Path(out).write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")
    if not cfg["quality"]["valid"]:
        print("WARNING: calibration_valid = false  ->  metric analytics must NOT be trusted")


if __name__ == "__main__":
    main()
