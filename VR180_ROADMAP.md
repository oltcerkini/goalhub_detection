# VR180 Pipeline — Status, Roadmap & Context

_Last updated: 2026-11-24. Maintainer note: read this first when resuming work._

## 0. How to work on this project (operating instructions)

- **Read this file first**, and **update it** after every phase / work session.
- **Execute, don't just plan.** Work in small verifiable steps; run the relevant
  test before moving to the next step. Ship working increments.
- **Verify, never assert.** Any claim ("it works", "no degradation") needs a
  number or a measurement behind it. If a result surprises you, check it
  *before* reporting.
- **Report outcomes, not engineering essays.** The user does not need the
  internals. Say what works / what doesn't / what's next, briefly. Only go deep
  when asked.
- **Flat pipeline stays the default and untouched.** All VR180 work is additive.
- **Never retrain a model, never delete the user's footage/datasets, without an
  explicit ask.**
- **Per-video "training" = calibration** (fit the camera pose for that video).
  ML retraining is a *last resort*, only for the ball, and only on the user's PC.
- **Keep the live task list in sync** with section 4 below (harness tasks).

**Live task list (harness):** #9 runnable pipeline · #10 calibration one-command ·
#11 metric projection + gate · #12 ROI · #13 goal validation (36.35 s) ·
#14 ball temporal redesign · #15 app upload→calibrate→process ·
#16 validate all 6 clips · #17 virtual camera + MP4.

## RESUME HERE  (saved 2026-10-09, end of session)

**Do this first:** read this file top-to-bottom. Current blocker: the **real
Barca trace does not calibrate yet** — the fitted pose is degenerate.

| # | task | state |
|---|---|---|
| 9 | VR180 runnable pipeline | ✅ **DONE** — `vr180_pipeline.py` |
| 10 | calibration one-command (traces → calibration.json) | ✅ **DONE** — `calib/calibrate.py`; synthetic self-test recovers the pose |
| 11 | metric projection + calibration gate | ✅ **DONE** — vspan honoured; above-horizon rays gated; invalid cal withholds metres |
| 12 | ROI: drop off-pitch | ✅ **DONE** — calibration-derived pitch ROI; grass fallback |
| **CAL** | **real calibration (calib_equirect.json)** | ⚠ **APPROXIMATE, accepted 2026-10-12** — parity bug fixed → `valid=true` (med 0.82 m) but the trace is imperfect (near touchline inside the paint) so expect metre-level error; `calib/calibration.json` shipped |
| 13 | goal validation on 36.35 s clip | ⚠ **tested 2026-10-12 — 0 goals, BLOCKED on the ball.** With the approximate calibration, the ball trail (`out/track_goal2.json`, 35–39.6 s) sits at ground x∈[-0.5,10.6] z∈[24,34] for all 110 frames — never within 3 m of a goal line (x=±52.5), and dominantly at a few REPEATED pixels (eq(464,1021) ×7, eq(2979,1445) ×6 …) = static false positives. The goal logic can't fire because the ball isn't tracked. Fix #14 first. |
| 14 | ball temporal redesign (Track B) | ⏸ the main open risk |
| 15 | app: upload → calibrate → process | ✅ **DONE 2026-10-12** — `app180.py` (port 8081) + `app/templates/index180.html`: upload 180 video → trace in the embedded `calibrate.html` → calibrate → **detect players → annotated MP4** (`/api/video/{task_id}`; boxes + track id + pitch position, h264). Verified end-to-end over HTTP on a 30 s clip: 30 frames, 21-22 players, 1.25 s mp4. Drone `app.py` untouched. |
| 16 | validate across all 6 clips | ⏸ |
| 17 | virtual camera + MP4 | ⏸ |

**Session 2026-11-24 — calibration is now a true Bézier pen tool.**
- `calib/calibrate.html`: click = corner (straight); click-drag = curve **only the
  outgoing edge**; auto-closes at the required count (pitch 4, halfway 2, centre circle 4).
  Afterwards **drag any line to bend it in place — no mode switch, no new point**.
  Anchors move the point; a handle drag bends only that side; **Alt-drag mirrors** it.
- **Separate goal shapes removed** — the pitch outline already yields both goal lines
  (`goal_line_left/right` along the pitch ends). Tool is now just **Pitch (required) ·
  Halfway · Centre circle**, plus a single **Calibrate** button. Interface simplified.
- Verified with a Node stub harness (20 checks: cubic sampling, out-only pen drag, direct
  drag-to-bend with no added anchor, auto-close, chip add/remove, trace schema) **and in
  real headless Chrome** via CDP mouse events. The first two-mode UI failed the user in
  practice — dragging a line in the default mode did nothing. The Chrome test reproduced
  it, so the mode buttons were removed: drag-to-bend is the default interaction.
  Solver self-test still 0.312 m (`valid=true`).
- `app180.py` serves `/` and `/calib/tool` with `Cache-Control: no-store`; the app
  cache-busts the iframe (`v=Date.now()`), so a reload/re-upload gets the current tool.
- `app/templates/index180.html`: step-2 copy updated (drag any line to bend); `valid=false`
  shows a warning instead of a green success line.

**Session 2026-11-24 — player precision/recall inside the pitch.**
- **The fitted-pose ROI was wrong on the Barca clip.** Rendering `pitch_to_equirect`
  from `calibration.json` put the pitch polygon far outside the paint, so the pose ROI
  kept every foreground spectator (0/26 dropped). The **raw trace** is aligned with the
  paint, so the ROI now comes from the traced outline, not the solved pose.
- New `spherical_tiling.pitch_polygon_from_trace` + `trace_roi_filter`: longest line per
  pitch label (legacy goal-shape duplicates ignored), trace→content scaling, angle-sort
  polygon; ground contact = bbox bottom-centre; outward margin = 0.20 × bbox height
  (near-player overhang). `vr180_pipeline --trace` (also auto-loads
  `<calibration>_trace.json`); `app180` passes it. `test_trace_roi.py` ALL PASS.
- **Measured (Barca 30 s clip, frames 0/240/480):** tiled@0.10 merged 26/31/27; trace ROI
  kept 19/20/17, dropping the bottom spectators and far-sideline people. Lowering conf to
  0.07 added only an 11×4 noise box → **PLAYER_CONF stays 0.10**.
- **The tracker was the recall bottleneck (three separate loss mechanisms):**
  (1) ByteTrack only starts tracks above ~0.35, so 1–4 on-pitch players/frame
  (conf 0.10–0.25) were never tracked; `PlayerTracker` gained an optional
  `track_activation_threshold` (default **0.25 — flat path unchanged**), VR180 uses 0.0.
  (2) supervision's wrapper re-matches detections to tracks at a hard-coded **IoU 0.5**
  and drops the rest; VR180 uses `strict_iou=False`.
  (3) ByteTrack keeps a new track *unconfirmed* for one frame; `_update_tracks` now emits
  the current frame's unconfirmed tracks too.
  Plus a player-shape filter (h≥8 px, aspect 0.6–8) and `--track-min-hits 1`.
  16-frame A/B: old [16,16,18,18,19,17,17,17,17,16,17,16,17,16,16,14] / 19 IDs;
  new [20,19,20,21,21,18,20,19,19,19,19,17,17,18,19,17] / 21 IDs. No ID explosion.
- **Full-frame union tested and REJECTED:** it added ~9–10 boxes/frame, but they were
  subs/coaches outside the far touchline (box bottoms crossed the line within the margin),
  not missing players. Tiled@0.10 already finds the on-pitch players.
- Flat pipeline untouched (defaults preserved); **no model retrain.** Rendered overlays in
  gitignored `out/roi_*.jpg`; `test_trace_roi.py` PASS; app `/api/process` uses `--trace`.

**Calibration session (2026-10-09).** The user traced `calib/calib_equirect.json`
(frame_85s, Barca, 105×68). Visually the near touchline / both goal lines /
halfway DO sit on the painted lines (verified with `out/trace_labels.jpg`). But
the fitted pose is **degenerate** — the model pitch renders off-frame (1/4 corners
in frame) — so `calibration_valid = false`. Root cause: ill-conditioned geometry
(the far touchline sits at the horizon where ~1 px = metres; the camera is
essentially ON the near touchline, so that line barely constrains the pose).

**Solver state (commits `f275157`, `321f05e`):** off-frame trace points dropped;
boundary lines are **segments** not infinite lines (infinite lines let points
slide → degenerate poses); robust loss (soft_l1) + `quality.suspect_labels`;
**geometric gate** `corners_in_frame >= 2` so a degenerate pose cannot pass on a
low metre residual alone. `test_metric_roi.py` is self-contained and **ALL PASS**;
synthetic self-tests **PASS**. The solver REFUSES rather than emit a bogus pose.

**Session 2026-10-12 — the parity bug FIXED; the metre gate is still lying.**
- **Root cause of the degenerate fit found and fixed.** The tool labels goal lines
  by *image side* (`calibrate.html` edge order: far/goal_right/near/goal_left),
  while the model's camera frame puts +x on the image LEFT. A rectangular pitch
  is 180°-rotationally symmetric so this mirror-disagreement is invisible in the
  metres residual — the solver latched onto the mirrored branch and went
  degenerate. `calibration_from_traces` now tries both goal-label parities
  (`GOAL_FLIP`) and keeps the pose that renders the pitch in frame. Recorded as
  `traced_features.label_parity`. **The user's real trace now calibrates**
  (was `valid=false`/1-corner; now `valid=true`, 4 corners, med 0.60 m). No
  regression: `calibrate.py --synthetic` still recovers the pose (0.31 m, no flip)
  and `test_metric_roi.py` ALL PASS.
- **But do NOT trust the metres yet.** With the flip the model mis-places the
  centre circle by ~65 px (rendered overlay `out/circ_model.jpg`): the model
  circle sits above the painted one. med_m 0.60 passes the <1.0 m gate while the
  pose is several metres wrong mid-pitch — the exact trap §4 warns about.
- Ruled out as the sole cause: dropping the suspect `touchline_far` (flagged 19 m,
  vs 1.21 m overall) barely moves it; sweeping span (see below) helps size not
  position; adding a centre-circle constraint does not move the lines-dominated fit.
- The traced `touchline_far` (y≈730–755) sits ABOVE the horizon (row ≈790 in the
  1586-tall crop) — geometrically impossible for a ground line → that trace is wrong.
- **Span: the crop's effective horizontal span is ~159–165°, not 180.**
  sidecar crop = [221,291,3401,1586] against a **3840-wide** frame → 3401/3840·180
  = 159.4°. The old "span=180 verified" check only confirmed the code matches the
  intended math (horizon row / yaw-0 column) — it never pinned the crop's angular
  width. At span≈160 the model's centre circle finally matches the painted circle
  in SIZE (w638/h63 vs ~w640/h65); span=180 is ~45 px small. Centres still ~65 px
  apart → span is a *contributing* error, not the whole story. NOTE: span is a
  sidecar/`normalize_frame` value (currently hard 180) — fixing it means either
  using the full frame or rescaling span by crop/frame width.
- **Best explanation of the residual:** the camera HEIGHT is under-constrained
  (line fit → 5.6 m; 6-pt PnP → 8.7 m) because the camera sits essentially ON the
  near touchline and the far touchline trace is bogus. The centre circle — a
  well-conditioned mid-pitch feature — is the constraint that would pin it, but it
  must be TRACED (the tool + `geom` support `centre_circle`).

**DECISION 2026-10-12: accept an APPROXIMATE calibration and move on.**
- Added a **Centre circle** shape to `calib/calibrate.html` (4 corners + bent edges,
  all edges labelled `centre_circle`; emits `centre_circle` on save). The tool
  previously had NO circle shape — the earlier "trace the centre circle" ask was
  impossible until now. Syntax-checked.
- The user traced the centre circle cleanly (`calib/circle.json`): image
  **c(1705,845) w634 h99**. Clean and believable.
- But **the lines and the circle cannot both be fit**: with all lines + circle no
  pose gets below ~2 m median, and no (span, vspan) setting fixes it (grid over
  span 100–200 × vspan 60–180: best still ~1.9 m). The lines alone fit to 0.82 m
  but then the model circle lands ~80 px ABOVE the traced one. The traced near
  touchline sits **inside** the paint (`out/near_bottom.jpg`) — by ~0..70 px
  growing to the right — and the far corners don't meet (touchline vs goal line
  ~33 px apart). The camera sits essentially ON the near touchline, so that line
  is nearly degenerate and its error dominates.
- Verified with a free-dimension fit that the data wants a ~105×45 pitch, not
  105×68 (the user is adamant the pitch is 105×68) — one more sign the trace,
  not the pitch, is the problem.
- **Shipped:** `calib/calibration.json` = the lines-only (parity-fixed) fit of
  `calib/calib_equirect (1).json`: P=[0.59, 5.32, 33.56], med 0.816 m, 4 corners,
  valid=true. Source copy in `calib/calibration_barca2.json`. It is **approximate**:
  expect metre-level error, worst mid-pitch. End-to-end run OK —
  `python vr180_pipeline.py --video <barca> --calibration calib/calibration.json
  --start 85 --frames 5` → 14 players/frame with on-pitch `pitch_m` coords.
- **NEXT:** #13 goals / #14 ball can now proceed on this clip, but do NOT quote
  distances as accurate. If accuracy is wanted later: re-trace the pitch outline
  tightly on the paint (bottom line especially), and use a clip/camera angle that
  is not degenerate (camera off the touchline).

**Diagnostic artifacts (gitignored `out/`):** `calib_overlay.jpg`, `calib_final.jpg`,
`trace_labels.jpg`, `calib_model_thick.jpg`, `near_touchline_crop.jpg`,
`far_touchline_crop.jpg`. `calib/calib_equirect.json` = the user's real trace
(untracked; keep it).

**Where we are:** the whole chain runs in one command —
`python vr180_pipeline.py --video "assets/180Videos/<clip>.mkv" [--calibration calib/calibration.json]`
→ normalize → FOV-60 player tiling → merge → ROI → ByteTrack → full-frame ball →
tracking JSON (`mode: "equirect"`). `calib/calibration.json` on disk is the
**synthetic** fixture (the real attempt was degenerate, not committed).

**Known state / open items:**
- Ball is the weak link; #14 is the redesign. Detectors are **not** retrained.
- Vertical-span assumption unresolved (§5b): 180 assumed, `fit_vspan` off.
- One clip is SBS stereo (Viewpt) — handled by `equirect.py`.

**Environment / assets:** system Python 3.11 with ultralytics (TensorRT engine
loads), OpenCV, scipy, supervision. Six clips in `assets/180Videos/`. Models at
repo root (`yolo26l.pt`, `ball_detector_yolo26m.engine`).

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
| Metric gate | vspan honoured; above-horizon rays → `pitch_m` omitted; `calibration_valid=false` withholds metres |
| VR180 player ROI | **traced-pitch polygon** (aligned to the paint) drops foreground spectators / far-sideline people: 26→19, 31→20, 27→17 kept on frames 0/240/480. Pose/grass remain fallbacks. |
| VR180 player tracking | VR180-only: activation 0.0, no IoU-0.5 re-match, unconfirmed tracks emitted, min-hits 1, shape filter. 16-frame run: **17–21 players/frame, 21 unique IDs** (old: 14–19, 19 IDs). Flat defaults unchanged. |
| Calibration input | the **4-corner pitch outline alone** recovers the pose to ~0.8 m @2 px click noise; adding the halfway line → ~0.5 m. Penalty boxes / centre circle are NOT needed. |
| Calib tool contract | Bézier-pen trace round-trips through `calib/calibrate.py` (solver self-test 0.312 m); the same trace drives the ROI polygon |
| Goal geometry | synthetic PASS (roll-in detected; wide / over-bar / far-away rejected) |

### Open / blocked
| Issue | Status |
|---|---|
| **Ball detection on VR180** | Finds real balls on the **full frame** (visually confirmed) but low confidence (median 0.11). Validator re-tune (Test A) and player-proximity prior (Test B) **both failed** — best 42 % coverage / 28 % multi-detection. **This is the main open problem.** |
| Real calibration | Needs the user to trace `calib/calib_equirect.json`. Tool exists. |
| Goal detection on real footage | Needs calibration + a clip with a confirmed goal. |
| ROI / off-pitch filtering | ✅ **DONE 2026-11-24** — ROI is the traced pitch polygon (pose/grass fallbacks). Measured to drop foreground + far-sideline spectators. |
| Virtual camera renderer | Only the pose controller (`virtual_camera.py`) exists. No video output. |
| Integration | `--input-mode equirect` in `process.py` **not started**. |
| Events/analytics beyond goals | Not built for equirect. |

### Key measured facts (don't re-litigate)
- **Tiling is the right call.** Full-frame detection on a 180° frame (3401 px ≈ 19 px/deg) downscales ~3× → 40–45 detections for ~20 players (clutter + spectators). Tiled → 25–26. Same model, same confidence range (0.10–0.91).
- Narrower FOV does **not** help: FOV-40 loses far recall; FOV-20 (3× upsampling) is much worse. **FOV-60 is near-optimal** (ideal ≈68° at imgsz 1280).
- The far-side limit is **source angular resolution** (~19 px/deg), not the pipeline. An 8K source would roughly double it.
- **Full-frame union does not add on-pitch recall** (2026-11-24): the extra boxes were subs/coaches behind the far touchline. Tiled@0.10 already finds the players.
- ByteTrack's default **0.25 activation**, the wrapper's hard-coded **IoU-0.5 re-match**, and the one-frame **unconfirmed** delay each silently dropped far players (conf 0.10–0.25). VR180 bypasses all three; the flat pipeline keeps the defaults.
- Equirect **span=180 has no horizontal seam** (that only exists for 360).
- All equirect coordinates are relative to the **cropped content region**; the ball trail was once in full-frame coords → offset by the crop origin.

## 3. Architecture (module map)

```
spherical_projection.py   equirect <-> perspective (equirect_to_perspective)
spherical_tiling.py       detect_content_region, detect_span, build_grid, tile_equirect,
                          hemisphere_mask, tile_pixel_to_equirect, merge_to_equirect (centre-NMS),
                          pitch_polygon_from_trace / trace_roi_filter (trace-derived pitch ROI),
                          grass_roi_polygon, roi_filter, sidecar <video>.equirect.json
virtual_camera.py         compute_camera_pose (smooth follow; tested)
calib/calibrate.html      BÉZIER PEN (2026-11-24): click = corner, click-drag = curve only
                          the outgoing edge; drag ANY line afterwards to bend it in place
                          (no mode, no new point). Pitch (4 anchors) required; halfway (2)
                          and centre circle (4) optional. Auto-closes at the count; anchors
                          move, handles bend one side, Alt-drag mirrors. Saves
                          calib_equirect.json ({lines:[{label,points}]}).
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

**Resolution attempt (commit pending):** added `vspan` as an optional solved
parameter (`solve_pose(fit_vspan=True)`). **Result: it is ill-conditioned** —
with 2 px trace noise it drifts (true 120° → fitted 137.8°; true 90° → 97.7°)
because vspan and camera pose trade off. It *lowers* the residual while being
wrong, so it can pass `calibration_valid` and still corrupt metres. **Therefore
it is OFF by default**: assume vspan = 180 (validated on the Barca clip) and let
a wrong assumption fail loudly (bad fit → `calibration_valid = false`) rather
than overfit silently.
Remaining options: (a) accept 180 and validate per clip via the fit; (b) get the
camera's real vertical FOV from the user/manufacturer; (c) constrain vspan with
a prior (square-pixel assumption: vspan = span * H / W — currently contradicts
the Barca round-circle observation, so unresolved).

**Second finding:** one in six clips is **stereo** (SBS) — `equirect.py` now
auto-detects and de-stacks (takes one eye).

## 6. Waiting on the user
- ~~A VR180 clip with a clear goal~~ — **resolved**: user confirmed a goal at
  **36.35 s** in `Barca Academy Austin 04B Blue vs FC DallasAlamo 4K 180VR`
  (`llQXdpryFjI`): "player shoots the ball and scoring". Verified the frame shows
  the left goal, penalty box and the ball. **This is the real-video validation
  case for goal detection** (needs calibration + ball track to exercise).
- A **second** VR180 clip (different pitch/camera) for calibration generalisation
  — already have 5 more clips in `assets/180Videos/`.
- Optional: a flat/drone clip of a similar pitch for a real flat-vs-VR180 A/B.

## 7. Decision log
- Span **180** is the production target; 360 kept as a parameter.
- **Tiling kept** (FOV-60) — it preserves angular resolution; full-frame is worse.
- **Calibration = spherical ground-plane**, not homography.
- **No retraining** for players; ball retrain only if the temporal redesign fails.
- Validity uses the **median** residual (RMS is inflated by near-horizon traces).
- **Metric gate = below-horizon ray AND `calibration_valid`** — an above-horizon
  ray has no ground intersection, so `pitch_m` is omitted (never clamped garbage).
- **vspan is a first-class projection parameter** — `make_mappers(..., vspan=)` /
  `equirect_to_ground(..., vspan=)`; the pipeline feeds it the calibration's
  `vspan_deg`. Mismatching it scales every metre silently.
- **VR180 ROI = the user's traced pitch polygon, not the fitted pose (2026-11-24).**
  The traced pixels sit on the paint; the solved pose is too inaccurate on touchline
  cameras (the pose polygon missed the pitch entirely on the Barca clip). Ground
  contact = bbox bottom-centre; outward margin = 0.20 × bbox height. Pose rectangle
  and grass hull remain fallbacks when no trace is available. The ball uses the same
  trace polygon (5 px tolerance), with the ground-projection check as fallback.
- **VR180 tracker = activation 0.0, no IoU-0.5 re-match, emit unconfirmed, min-hits 1,
  shape filter (2026-11-24).** Three loss mechanisms: ByteTrack's 0.25 activation, the
  supervision wrapper's hard-coded IoU-0.5 re-match, and one-frame "unconfirmed" new
  tracks. All three are bypassed for VR180 only; `PlayerTracker` defaults (0.25, strict)
  keep the flat pipeline identical. No retrain.
- **Calibration robustness (2026-10-09, real trace):** off-frame trace points
  dropped; boundary lines are modeled as **segments** not infinite lines (infinite
  lines let points slide → degenerate poses); a **robust loss** (soft_l1) down-weights
  a mistraced near-horizon line and reports it in `quality.suspect_labels`; and a
  **geometric gate** requires the fitted pitch to render in-frame (`corners_in_frame
  >= 2`) so a degenerate fit cannot pass on a low metre residual alone.
- **Real Barca trace (frame_85s) does NOT calibrate yet.** The near touchline/goal
  lines/halfway trace correctly, but the fitted pose collapses to a degenerate one
  (pitch off-frame, 1/4 corners). Causes: the far touchline sits at the horizon
  (~1 px = metres) and the camera is essentially on the near touchline, so the
  metre residual is ill-conditioned. Solver now flags this (valid=false) rather
  than emitting a bogus pose. Needs a better trace or a point-correspondence fit.
- **Calibration input = 4 corners per shape + bendable edges** (pen tool), NOT
  freehand tracing. Only the pitch outline is required; the halfway line is an
  optional refinement; penalty boxes and the centre circle were dropped. Edge
  labels are assigned by image position (near = lower in frame), so the recovered
  pose may be the pitch's mirror twin — harmless, the pitch is D2-symmetric.
- **Calibration tool rebuilt as a true Bézier pen (2026-11-24).** Click = corner, drag =
  curve only the outgoing edge; **drag any line afterwards to bend it in place** (no mode
  switch, no new point). Handle drags are single-sided; Alt-drag mirrors. The separate
  goal shapes are gone: tracing the pitch outline already produces both goal lines
  (`goal_line_left/right`). Required = pitch; optional = halfway, centre circle. UI =
  Pitch / Halfway / Centre circle + Calibrate. Verified in headless Node and real headless
  Chrome (drag-to-bend works); solver self-test 0.312 m. The initial two-mode UI confused
  the user — drag-to-bend in the default mode is the supported interaction.

## 8. Commit trail (chronological)
`bed22d8` flat fixes → `5fd12cc`/`09d1518` Projection Phase 1 →
`3da9204` tiling → `d811891` FOV-40 → `0164dd0`/`18a20e7` bbox+evidence →
`95ffe1a` FOV-20 falsified → `b4ee790` ball on tiles → `3fb4500` ball on full frame →
`60a865d` ball recall → `fe97d6a` retrain decision →
`42b017c` calib tool → `fceee7f` solver → `8808a95` goals →
`769f929` pitch-size prompt → `e1f53f6` freehand pen →
`2d862db` normalize + inventory → `bbf8c87` goal event (36.35 s) →
`b4ed028` vspan param → `6231884` runnable pipeline → `3b0d2ce` session state →
`5090e43` metric gate + pitch ROI.
