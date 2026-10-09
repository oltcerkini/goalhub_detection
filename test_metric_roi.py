#!/usr/bin/env python3
"""Task #11/#12 — metric projection gate + calibration-based pitch ROI.

Verifies the vr180_pipeline wiring against the synthetic calibration:
  (1) vspan from the calibration reaches the projection (not silently 180);
  (2) map round-trip is exact (the two maps are true inverses under the pose);
  (3) above-horizon rays are gated (to_pitch -> None), below-horizon pass;
  (4) pitch membership keeps on-pitch points and drops off-pitch ones.
"""
import json
from pathlib import Path

import numpy as np

from vr180_pipeline import make_metric
from calib.spherical_calib import make_mappers, equirect_to_ground, pitch_to_equirect, DEFAULT_PITCH

CAL = json.loads(Path("calib/calibration.json").read_text())
W, H = CAL["image_size"]
SPAN, VSPAN = float(CAL["span_deg"]), float(CAL["vspan_deg"])
POSE = {"P": np.array(CAL["camera_pose"]["position_m"]),
        "rvec": np.array(CAL["camera_pose"]["rotation_vector"])}
assert CAL["quality"]["valid"], "synthetic calibration must be valid"

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


print(f"calibration: {W}x{H} span {SPAN} vspan {VSPAN} valid={CAL['quality']['valid']}")

# (1) vspan reaches the projection ------------------------------------------
e2p180, _ = make_mappers(POSE, W, H, SPAN, 180.0)
e2p90, _ = make_mappers(POSE, W, H, SPAN, 90.0)
x180, _ = e2p180(W / 2, H * 0.75)
x90, _ = e2p90(W / 2, H * 0.75)
check("vspan propagates into the projection",
      not np.allclose(np.ravel(x180), np.ravel(x90)),
      f"(vspan180 {float(np.ravel(x180)[0]):.1f} m vs vspan90 {float(np.ravel(x90)[0]):.1f} m)")

# make_metric must honour cal vspan_deg (here 180, so compare to a doctored cal)
metric = make_metric(CAL, W, H)
cal90 = json.loads(json.dumps(CAL)); cal90["vspan_deg"] = 90.0
metric90 = make_metric(cal90, W, H)
check("make_metric honours vspan_deg",
      metric["to_pitch"](W / 2, H * 0.75) != metric90["to_pitch"](W / 2, H * 0.75))

# (2) round-trip under the pose ---------------------------------------------
print("round-trip (project a known pitch point, map it back):")
for name, (xm, zm) in {"centre spot": (0, 0), "on-pitch": (20, 10),
                       "pen-box corner": (36, 20.16), "behind far touchline": (0, 50)}.items():
    ex, ey = pitch_to_equirect(xm, zm, POSE, W, H, SPAN, VSPAN)
    ex, ey = float(np.ravel(ex)[0]), float(np.ravel(ey)[0])
    in_img = 0 <= ex < W and 0 <= ey < H
    got = metric["to_pitch"](ex, ey) if in_img else None
    err = float(np.hypot(got[0] - xm, got[1] - zm)) if got else float("nan")
    check(f"round-trip {name}", in_img and got is not None and err < 0.05,
          f"(eq {ex:.0f},{ey:.0f} -> {got} err {err:.3f} m)")

# (3) horizon gate ----------------------------------------------------------
col = [metric["to_pitch"](W / 2, y) for y in range(0, H, 4)]
n_none = sum(v is None for v in col)
check("horizon inside frame (some rays gated)", 0 < n_none < len(col),
      f"({n_none}/{len(col)} sampled rows above horizon)")
check("bottom of frame is a valid ground ray", metric["to_pitch"](W / 2, H - 2) is not None)

# (4) pitch ROI membership --------------------------------------------------
print("pitch ROI:")
for name, (xm, zm), want in [("centre spot", (0, 0), True),
                             ("on-pitch", (20, 10), True),
                             ("behind far touchline", (0, 50), False),
                             ("beyond goal line", (60, 0), False)]:
    ex, ey = pitch_to_equirect(xm, zm, POSE, W, H, SPAN, VSPAN)
    ex, ey = float(np.ravel(ex)[0]), float(np.ravel(ey)[0])
    ok = metric["on_pitch"](ex, ey)
    check(f"on_pitch {name} == {want}", ok == want, f"(got {ok})")

# (5) invalid calibration -> metrics withheld, ROI deferred -----------------
calbad = json.loads(json.dumps(CAL)); calbad["quality"]["valid"] = False
mbad = make_metric(calbad, W, H)
check("invalid cal -> to_pitch None", mbad["to_pitch"](W / 2, H - 2) is None)
check("invalid cal -> on_pitch None (caller falls back)", mbad["on_pitch"](W / 2, H - 2) is None)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)
