#!/usr/bin/env python3
"""Step 2 sampler: run a short slice of every clip in assets/180Videos.

Emulates the app's defaults (auto region/stereo via normalize_frame, vspan
prefilled only for very wide strips) so cheap failures surface the way a user
would hit them.

    python scripts/sample_clips.py [--frames 12] [--step 2] [--no-ball]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from equirect import normalize_frame                        # noqa: E402
import vr180_pipeline                                       # noqa: E402

CLIPS = [
    ("barca_04b_blue",   "Barca Academy Austin 04B Blue vs FC DallasAlamo 4K 180VR [llQXdpryFjI].mkv", 1200),
    ("barca_04b_chall",  "Barca Academy Austin 04B vs Challenger United 04B Predators 4k 180VR [qZiuleQYujQ].mkv", 1800),
    ("barca_11b_garnet", "Barca Academy Austin 11B Garnet vs Titans 4K 180VR [4HSjuJZR3p8].mkv", 1000),
    ("night_wolves",     "D.D.WOLVES FC - 19th January 2026 4K - Full View (VR-180) [A1WwetUCZoA].mkv", 900),
    ("portrait_pasc",    "PASC Fire vs Lower Mac Impact - 2nd Half VR180 [9MzVv9Aq9sk].mkv", 600),
    ("stereo_viewpt",    "Sports Football -- Viewpt Nano [j2iM3LCEjK0].mkv", 150),
]


def probe(path, start_s):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    ok, raw = cap.read()
    cap.release()
    if not ok:
        return None
    content, meta = normalize_frame(raw)
    h, w = content.shape[:2]
    implied = 180.0 * h / w
    vspan = round(implied) if implied < 60 else None
    return {"layout": meta["layout"], "region": meta["region"],
            "content": [w, h], "implied_vspan": round(implied, 1), "vspan": vspan,
            "mean_luma": round(float(cv2.cvtColor(content, cv2.COLOR_BGR2GRAY).mean()), 1)}


def summarize(res, probe_info, secs):
    frames = res["frames"]
    n = len(frames)
    per_frame = [len(f["players"]) for f in frames]
    ids = {p["id"] for f in frames for p in f["players"]}
    ball_frames = sum(1 for f in frames if f["ball"])
    return {
        "layout": res["layout"], "region": res["region"], "image_size": res["image_size"],
        "implied_vspan": probe_info["implied_vspan"], "vspan": res["vspan_deg"],
        "vspan_source": res.get("vspan_source"),
        "enhance": res["enhance_counts"], "roi_mode": res["roi_mode"],
        "calibration_valid": res["calibration_valid"],
        "frames": n, "players_median": statistics.median(per_frame) if per_frame else 0,
        "players_min": min(per_frame) if per_frame else 0,
        "players_max": max(per_frame) if per_frame else 0,
        "unique_ids": len(ids), "ball_frames": ball_frames,
        "secs": round(secs, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated clip keys")
    a = ap.parse_args()

    only = set(a.only.split(",")) if a.only else None
    outdir = ROOT / "out" / "short_samples"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, name, start in CLIPS:
        if only and key not in only:
            continue
        path = ROOT / "assets" / "180Videos" / name
        print(f"\n=== {key} @ {start}s ===", flush=True)
        info = probe(path, start)
        if info is None:
            print("  probe FAILED (no frame)"); summary[key] = {"error": "probe failed"}
            continue
        print(f"  {info}", flush=True)
        t0 = time.time()
        try:
            res = vr180_pipeline.run(str(path), start_s=start, max_frames=a.frames,
                                     step=a.step, ball=not a.no_ball, verbose=False)
        except Exception as e:                              # noqa: BLE001
            import traceback; traceback.print_exc()
            summary[key] = {"error": f"{type(e).__name__}: {e}"}
            continue
        s = summarize(res, info, time.time() - t0)
        s["probe"] = info
        summary[key] = s
        print("  " + json.dumps({k: v for k, v in s.items() if k != "probe"}), flush=True)
        (outdir / f"{key}.json").write_text(json.dumps(res, indent=1))

    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n=== SUMMARY ===")
    for key, s in summary.items():
        if "error" in s:
            print(f"{key:18s} ERROR {s['error']}")
            continue
        print(f"{key:18s} {s['layout']:5s} img={s['image_size']} "
              f"vspan={s['vspan']}({s['vspan_source']}) "
              f"enh={s['enhance']} roi={s['roi_mode']} "
              f"players med/min/max={s['players_median']}/{s['players_min']}/{s['players_max']} "
              f"ids={s['unique_ids']} ball={s['ball_frames']}/{s['frames']} "
              f"{s['secs']}s")
    print(f"\nwrote {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
