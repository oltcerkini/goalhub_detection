# VR180 Pipeline — Status, Roadmap & Context

_Last updated: 2026-10-08. Maintainer note: read this first when resuming work._

## 1. Product goal (the user's words)

> Upload a 180° video → trace the curved pitch lines and enter the pitch dimensions →
> the system detects players, the ball and goals → distances/positions in metres →
> a follow-camera video.

Target user: an amateur club coach — one fixed 180° camera pitch-side, no operator.

**Hard requirements**
- Player detection must not be *degraded* vs the existing drone/flat pipeline.
- Flat pipeline stays default and unchanged.
- No new ML model unless the existing detector demonstrably fails.
- No 4-point homography (pitch lines are curved in equirect space).
- Analytics must be in **metric pitch coordinates**, not pixels.
- Equirect pixels remain the canonical **image-space** representation.

## 2. Current state

### Proven (measured, with evidence)
| Area | Result |
|---|---|
| Projection equirect↔perspective | straight-line residual 0.0000 px; horizon at row 540.0; span 180 for VR180 |
| Spherical tiling | FOV-60 grid (5 yaw × 2 pitch = 10 tiles); exact tile→equirect inverse (round-trip <2 px) |
| Player detection | same model `yolo26l.pt` @1280; tiled path; ByteTrack on equirect coords = 0.90× track ratio |
| Calibration solver | pose recovered from synthetic traces: 0.000 m clean, 0.19–0.31 % @1–2 px click noise |
| Metric mapping | round-trip 0.0000 m; end-to-end < 0.52 m @2 px noise |
| Goal geometry | synthetic PASS (roll-in detected; wide / over-bar / far-away rejected) |

### Open / blocked
| Issue | Status |
|---|---|
| **Ball detection on VR180** | Finds real balls on the **full frame** (visually confirmed) but low confidence (median 0.11). Validator re-tune (Test A) and player-proximity prior (Test B) **both failed** — best 42 % coverage / 28 % multi-detection. **This is the main open problem.** |
| Real calibration | Needs the user to trace `calib/calib_equirect.json`. Tool exists. |
| Goal detection on real footage | Needs calibration + a clip with a confirmed goal. |
| ROI / off-pitch filtering | Not wired into the run; both detection paths fire on spectators/bench. Calibration gives the pitch region. |
| Virtual camera renderer | Only the pose controller (`virtual_camera.py`) exists. No video output. |
| Integration | `--input-mode equirect` in `process.py` **not started**. |
| Events/analytics beyond goals | Not built for equirect. |

### Key measured facts (don't re-litigate)
- **Tiling is the right call.** Full-frame detection on a 180° frame (3401 px ≈ 19 px/deg) downscales ~3× → 40–45 detections for ~20 players (clutter + spectators). Tiled → 25–26. Same model, same confidence range (0.10–0.91).
- Narrower FOV does **not** help: FOV-40 loses far recall; FOV-20 (3× upsampling) is much worse. **FOV-60 is near-optimal** (ideal ≈68° at imgsz 1280).
- The far-side limit is **source angular resolution** (~19 px/deg), not the pipeline. An 8K source would roughly double it.
- Equirect **span=180 has no horizontal seam** (that only exists for 360).
- All equirect coordinates are relative to the **cropped content region**; the ball trail was once in full-frame coords → offset by the crop origin.

## 3. Architecture (module map)

```
spherical_projection.py   equirect <-> perspective (equirect_to_perspective)
spherical_tiling.py       detect_content_region, detect_span, build_grid, tile_equirect,
                          hemisphere_mask, tile_pixel_to_equirect, merge_to_equirect (centre-NMS),
                          grass_roi_polygon, roi_filter, sidecar <video>.equirect.json
virtual_camera.py         compute_camera_pose (smooth follow; tested)
calib/calibrate.html      freehand pen tool: trace curved lines + pitch size -> calib_equirect.json
calib/spherical_calib.py  pose solver (6-DoF, exact line residuals), equirect_to_pitch /
                          pitch_to_equirect, calibration_from_traces (contract v1)
calib/calibrate.py        CLI: trace file -> calibration.json
calib/goal_detect.py      goal plane + ball-crossing logic
test_*.py                 per-phase validations (keep these; they are the evidence)
```

Existing flat pipeline (untouched, default): `process.py`, `detector.py`,
`ball_detector.py`, `player_tracker.py`, `stats_computer.py`, `team_classifier.py`,
`heatmap.py`, `app.py`.

## 4. Roadmap — high-level tasks

### Track A — Make it usable end-to-end (highest user value)
- **A1. Upload → calibrate → process flow in `app.py`.** Upload a 180 video, extract a
  content-cropped frame, open the pen tool, save traces + pitch size, run the solver,
  then process. Mirrors the existing flat flow.
- **A2. ROI from calibration.** Derive the pitch region from the calibration and drop
  off-pitch player/ball detections. Removes the spectator/bench false positives.
- **A3. `--input-mode equirect` in `process.py`.** Flat stays default. Equirect path:
  crop → tile(players) → full-frame(ball) → merge → track → calibrate → metric → output.
- **A4. Output contract.** Same JSON schema as flat, plus `"mode": "equirect"`,
  a version, and metric positions where `calibration_valid = true`.
- **A5. Equirect-aware player path.** Module-ise tiling+merge+tracking behind a clean
  interface; ground-contact point (bbox bottom-centre) for metric projection.

### Track B — Ball (the open blocker)
- **B1. Redesign VR180 ball temporal filtering.** Motion + player proximity + pitch
  constraint + track continuity. The two cheap fixes are spent — this is a real design.
  Do not just raise the threshold.
- **B2. Decide ball strategy**: (i) full-frame low-conf + strong temporal model, or
  (ii) targeted retrain at the *full-frame* scale (~3–4 px at imgsz 1024), NOT tile scale.
- **B3. Emit a clean ball track**: timestamp, equirect pos, confidence, tracking state.

### Track C — Calibration robustness
- **C1. Auto-detect content region + span** for arbitrary VR180 sources (sidecar fallback).
- **C2. Calibration quality gate.** Refuse metric analytics when `calibration_valid = false`.
- **C3. Generalise** across camera height/position; validate on a second pitch.

### Track D — Football intelligence
- **D1. Events** from metric coords: possession, passes (success/fail), interceptions,
  turnovers, shots, goals.
- **D2. Analytics**: heatmaps, distance, speed, sprints, pass maps, team spatial stats.
- **D3. Reuse** `team_classifier` (jersey KMeans) attached per tracked player.

### Track E — Output experience
- **E1. Virtual follow-camera renderer** (independent of detection; uses the filtered
  ball track; configurable FOV/smoothing/dead-zone; no jitter).
- **E2. Overlays** (IDs, team colours, ball, boxes) + H.264 MP4.
- **E3. Heatmap re-enabled** on metric coordinates.

### Track F — Quality & tests
- **F1. Precision/recall** for player detection (tiled vs full-frame) against a counted frame.
- **F2. Geometry unit tests**: equirect↔ray↔tile↔pitch round-trips.
- **F3. Seam tests** (only relevant if 360 support is added).

## 5. Open risks
1. **Ball** is unresolved and gates passes/possession/goal-based events.
2. Source resolution caps far-side quality — higher-res or 8K camera changes the ceiling.
3. Calibration depends on user tracing quality (≤2 px clicks for <1 % pose error).
4. The pitch-size prompt matters: youth pitches (9v9/7v7) differ from 105×68.

## 5b. Footage inventory (`assets/180Videos/`)

Six clips received 2026-10-08. Normalization module `equirect.py` probes each
(robust letterbox detection + stereo layout):

| clip | raw | layout | content | aspect |
|---|---|---|---|---|
| Barca 04B Blue vs FC Dallas | 3840×2160 | mono | 3395×1592 | 2.13 |
| Barca 04B vs Challenger United | 3840×2160 | mono | 3398×1359 | 2.50 |
| Barca 11B Garnet vs Titans | 3840×2160 | mono | 3386×2074 | 1.63 |
| D.D.WOLVES (night) | 3840×2160 | mono | 3622×796 | **4.55** |
| PASC Fire (portrait) | 1920×2160 | mono | 1878×2082 | 0.90 |
| Sports Football / Viewpt Nano | 3840×2160 | **SBS stereo** | 1797×2040 | 0.88 |

**Open issue raised by the inventory — VERTICAL SPAN.** The projection assumes
the content height spans 180° of pitch. Content aspects span 0.88 → 4.55, which
cannot all be 180°×180° with plausibly square-ish pixels. Either the vertical
span differs per clip, or the pixel aspect varies far more than expected. The
Barca 04B Blue vs FC Dallas clip behaved correctly under the 180° assumption
(round centre circle in Phase 1), so it is at least right for that one.

**Proposed fix:** make the vertical span (or a pixel-aspect factor) a *solved
parameter* in `calib/spherical_calib.solve_pose` — the traces then determine it,
and a wrong assumption shows up as a poor fit (`calibration_valid = false`).
Until then, treat metric output as clip-validated only.

**Second finding:** one in six clips is **stereo** (SBS) — `equirect.py` now
auto-detects and de-stacks (takes one eye).

## 6. Waiting on the user
- A VR180 clip **with a clear goal** (validate goal detection on real footage).
- A **second** VR180 clip (different pitch/camera) for calibration generalisation.
- Optional: a flat/drone clip of a similar pitch for a real flat-vs-VR180 A/B.

## 7. Decision log
- Span **180** is the production target; 360 kept as a parameter.
- **Tiling kept** (FOV-60) — it preserves angular resolution; full-frame is worse.
- **Calibration = spherical ground-plane**, not homography.
- **No retraining** for players; ball retrain only if the temporal redesign fails.
- Validity uses the **median** residual (RMS is inflated by near-horizon traces).

## 8. Commit trail (chronological)
`bed22d8` flat fixes → `5fd12cc`/`09d1518` Projection Phase 1 →
`3da9204` tiling → `d811891` FOV-40 → `0164dd0`/`18a20e7` bbox+evidence →
`95ffe1a` FOV-20 falsified → `b4ee790` ball on tiles → `3fb4500` ball on full frame →
`60a865d` ball recall → `fe97d6a` retrain decision →
`42b017c` calib tool → `fceee7f` solver → `8808a95` goals →
`769f929` pitch-size prompt → `e1f53f6` freehand pen.
