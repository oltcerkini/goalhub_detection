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
| 13 | goal validation on 36.35 s clip | ⚠ **still BLOCKED 2026-11-25 — goal event not located in the file.** The ball is now tracked, but at 36.35 s it is at **midfield** (~+9 m, near the camera touchline), consistent with the metre mapping. A 3× **zoomed** ball scan of both traced goal mouths over **0–120 s** found **no ball entering either goal** (only a dead-ball clearance by the left goal at ~10.5–12 s; mouths visually empty in montages of 0–13 s / 33–41 s left and 35.5–39.5 s right). So it is **not** a calibration or ball-tracker fault. User gave ground truth **right goal 21:25**; at that far camera the ball is unresolvable/cluttered at the goal → still not validatable here; needs a closer-camera clip or a goal-region ball fine-tune. |
| 14 | ball temporal redesign (Track B) | ✅ **BASELINE DONE 2026-11-24** — `ball_trail.py` temporal tracker: **70% GT coverage, 2.4 px median**; follows the Barca ball 35.3 s+. Remaining: detector gaps (15/50 GT misses are `NO_CAND`, not tracker drops), id splits after gaps, sparse night clips. |
| 15 | app: upload → calibrate → process | ✅ **DONE 2026-10-12** — `app180.py` (port 8081) + `app/templates/index180.html`: upload 180 video → trace in the embedded `calibrate.html` → calibrate → **detect players → annotated MP4** (`/api/video/{task_id}`; boxes + track id + pitch position, h264). Verified end-to-end over HTTP on a 30 s clip: 30 frames, 21-22 players, 1.25 s mp4. Drone `app.py` untouched. |
| 16 | validate across all 6 clips | ⏸ |
| 17 | virtual camera + MP4 | ✅ **DONE 2026-11-25** — `vr180_pipeline.py --render-follow` reprojects the annotated equirect through a smoothed virtual camera (`virtual_camera.compute_camera_pose` → `spherical_projection.equirect_to_perspective`, FOV 70, 1280×720, h264). Targets ball if tracked else player-centroid; snaps to first target then eases. App exposes `/api/follow/{task_id}` + a second video. Verified on Barca 15 frames (Barca centre circle, boxed players, metre labels, clean pan). |

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

**Session 2026-11-24 — detection A/B (night clip, 10 s).**
- Clip: `out/D.D.WOLVES_10s.mp4` (cut from 15:00 of the 51-min night match), 8 sampled
  frames. Same tiled path, conf 0.10, shape filter; no ROI/calibration.
- Baseline (FOV60, imgsz1280): **15 detections**, median conf 0.221, 5 at ≥0.25.
- **CLAHE (clipLimit 2.0): 28 detections**, median 0.293, 18 at ≥0.25 — visually the boxes
  are real players. **Best config.**
- **2× Lanczos content upscale: 15 detections — no gain.**
- **imgsz1536: 9 detections — worse.** CLAHE+1536: 24 — worse than CLAHE alone.
- **Conclusion: do NOT add an upscaler.** Low-light preprocessing (CLAHE) is the cheap win.
- Remaining recall was geometry: content is 3621×796, but the projection assumed 796 px =
  180° vertical (~4.4 px/deg vertical vs ~20 px/deg horizontal).
- **vspan-aware tiling implemented (same session).** `equirect_to_perspective`,
  `tile_equirect`, `tile_pixel_to_equirect`, `hemisphere_mask`, the bbox merge and
  `build_grid` take `vspan`; a narrow strip uses one 36°-tall pitch row. Defaults stay 180.
- Night clip 8-frame A/B: **vspan180+CLAHE 28 detections / median 0.293**;
  **vspan40 246 / 0.769**; **vspan40+CLAHE 255 / 0.726, 214 at ≥0.25**. Visual overlays
  show boxes on distinct players (not duplicates). Full pipeline vspan40+auto (no ROI):
  **27–34 players/frame, 30 unique IDs**; CLAHE fired on all 8 night frames.
- **Conclusion: performance was geometry, not pixels.** Upscaling is falsified; CLAHE +
  the vertical-span correction is the win.
- App: a `vspan°` field auto-prefills ~40 when the content aspect implies <60°; the
  pipeline receives `--vspan`; `--enhance auto` is the default. (The grass fallback is
  unreliable on this night strip; the trace ROI is the path.)
- Caveat: the app's vspan field changes **tiling** only. The calibration pose/metrics
  still use the calibration's `vspan_deg` (180); solving calibration with vspan is the
  remaining step for accurate metres.
- `test_vspan_tiling.py` PASS; projection synthetic + `test_metric_roi` + phase4 PASS.
- Full-strip direct detection finds 20–30 boxes/frame but is clutter-heavy (not a fallback).

**Session 2026-11-24 — ball detection: temporal trail tracker.**
- New `ball_trail.py`: links low-confidence full-frame ball candidates with a
  constant-velocity tracker. Static FPs are rejected by net displacement / speed;
  gaps are predicted; the best track is chosen by confidence + net travel + hits.
- **GT clip `f5bacf2f` (drone, 50 labeled frames, `annotation/ball_gt.json`):
  35/50 tracked (70%), median 2.4 px, 89% ≤10 px, 100% ≤20 px.** The old top-1
  pick/validator failed because static FPs outscored the real ball.
- **Barca VR180 confirmed-goal clip:** the tracker follows the white ball left
  through 36.3 s (zoom confirmed); end-to-end 42/48 frames with a ball, ids 1/2.
- `vr180_pipeline` ball path replaced: full-frame candidates at conf 0.03 with a
  3–100 px size gate + trace/pose ROI -> tracker (gate 100 px, min hits 3); output
  carries `id`/`predicted`; the app no longer runs `--no-ball` and draws the ball.
- Remaining: 30% of GT frames untracked (detector gaps/start), the trail can split
  into ids 1/2 after a lost frame, and the wide night clip is candidate-sparse.
  A retrain is still optional; the tracker is the new baseline.
- `test_ball_trail.py` PASS; GT/Barca evidence images in gitignored `out/`.

**Session 2026-11-25 — ball GT re-measured; goal event not found.**
- `test_ball_trail` and `test_vspan_tiling` **ALL PASS**. GT eval re-run
  (`out/_ball_tracker_eval.py`): **35/50 = 70% coverage, median 2.4 px,
  89% ≤10 px, 100% ≤20 px** — unchanged, confirmed.
- **All 15 GT misses are detector gaps** (`out/_ball_diag.py`): no candidate
  (even before the size gate) within 40 px of the GT ball on those frames; the
  only candidates are a static left-edge blob. Only **2/15** are tracker
  false-drops (frames 28/32): a live track is rejected by the static gate
  `net < min_move_px` even though `speed` is fine — the gate uses `or`; a
  candidate fix is `net < min_move AND speed < min_speed` (not yet applied).
- **Goal #13 re-check — the ball trail does not reach a goal.** At 36.35 s the
  tracked white ball is at ground **(x≈+9 m, z≈28 m)** = midfield, near the
  camera touchline; the metre position **agrees with the image** (calibration
  not at fault). A 3× zoomed ball scan of both traced goal mouths over
  **0–120 s** finds **no ball inside either goal**; the only goal-area motion is
  a dead-ball clearance at the left goal ~10.5–12 s (mouth empty throughout).
  Visual montages (`out/goal_left_montage.jpg`, `goal_right_36_39_montage.jpg`,
  `goal_left_mouth_montage.jpg`, etc.) all show empty mouths. → the "goal at
  36.35 s" does not reproduce; awaiting the user's exact time/screenshot.
- `out/_goal_scan.py` (full-frame) + `out/_goal_zoom_scan.py` (zoomed, the
  sensitive path) are the reusable goal-locating harnesses.
- **Full 66-min scan (both goals, ~16 min) is too noisy to auto-locate a goal:**
  4633 goal-mouth hits → 225 "events", dominated by the ball model firing on the
  **nets / keepers / goal structure** at both ends. The highest-in-confidence
  goal-area object at 36:00 (left, conf 0.84) is the **goalkeeper**. So automated
  goal-finding from this far sideline camera is not reliable.
- **Strategic point:** this FC Dallas clip is a **far sideline** camera (ball ~10 px
  at the goal, goals cluttered). The product direction is **closer 180 cams**,
  where the ball/goal are much larger. Validating goals on this clip may be the
  wrong target — prefer a closer-camera clip. Either way, a single confirmed
  mm:ss is needed to exercise `goal_detect.py` on real footage.
- **User ground truth: RIGHT goal at 21:25 (1285 s).** Checked `out/_goal_validate.py`:
  the pipeline ball tracker over 1267–1307 s **locks onto near-touchline clutter**
  (ground x≈−5 m, z≈33 m) and never nears the right goal → `detect_goals` = 0.
  Ultra-tight montage `out/goal_right_1285_ultra_montage.jpg` (1284–1290 s) shows
  players + neon keeper packed in the goal mouth (consistent with a goal) but the
  **ball is not resolvable** at ~50 m — it blends into white shirts/keeper.
- **Verdict 2026-11-25: goal detection is NOT validatable on this far-camera clip.**
  The blocker is **source angular resolution at the goal + goal-mouth clutter**, not
  the goal logic (synthetic self-test PASS). To validate on real footage: use a
  **closer-camera clip**, or fine-tune the ball model on **goal-region ball-vs-clutter**
  data (ball + negatives: keeper/net/posts), which the user offered to supply.
- **Scale diagnostic on 18 hand-labeled frames (2026-11-25):** full-frame imgsz1024
  detects **0/9** labeled balls (nearest 700–1900 px away); a **3× zoomed** crop around
  each ball detects **9/9 at ~0.7 px, most conf 0.81–0.86**. So the ball IS detectable —
  the model is scale-limited at full-frame (a ~10 px ball → ~2.7 px input).
  `out/_eval_goal_gt.py`, `out/_eval_goal_gt_zoom.py`.
- **Fair multi-scale test (no GT hints) FAILS (2026-11-25):** 600 px native tiles +
  full-frame (`out/_multiscale_eval.py`) finds only **1/9** balls; nearest 134–922 px,
  with high-conf hits on **players**. General tiling/zoom does NOT work with this model
  (confirms the [[ball-native-tiling-falsified]] players↔ball confusion). The ball is
  reliable only in an isolated ball-centred crop. Higher inference res (`.pt` at
  imgsz 1280/1920/2560) also gives **0–1/9** (`out/_highres_eval.py`). **Real fix = a
  scale-aware ball retrain (needs more labels than 18) or a closer camera.**
  Tiling/zoom/high-res exhausted. Note: the far camera is not the product setup.

**=== SESSION 2026-11-27 — STEP 4: GOAL ANCHOR, LOCK-ON, GOAL ENTRY ===**
Working under the product constraint (normal wide camera, [[product-camera-setup]]).

**1. Calibration anchors the painted goal frames (biggest accuracy win).**
The user trace `calib/calib_equirect.json` contains `goal_left`/`goal_right`
(posts+crossbar, 68 pts each = 136 pts) that `calibration_from_traces` used to
drop because they are 3D, not ground lines. Added `goal_frame_specs`,
`_goal_frame_curve`/`_project_curve3d`, and a `frame_w`-weighted residual in
`fit_pixel`; `GOAL_FLIP` now swaps the frame labels with the goal-line parity.
Measured on the Barca clip (default `goal_frame_weight=1.5`):

| feature | before (no anchor) | after |
|---|---|---|
| traced goal frames (px, median) | 17.4 / 40.0 | **2.8 / 4.6** |
| near touchline (px) | 4.8 | 8.0 |
| overall median (px) | 4.96 | **5.11** (valid) |
| camera height | 5.8 m | ~3.9 m (plausible) |

Weight sweep (0.35→3.0) trades near-touchline (~5→10 px) against goal frames
(20→3 px); 1.5 is the chosen balance. **Leave-one-goal-out:** anchoring only one
goal predicts the other at 14–19 px — the far field stays ill-conditioned, so
trust the calibration AT THE TRACED GOALS, not at untraced far locations. The
quality stats now use UNSCALED pixel residuals (the optimizer still uses the
weighted ones).

**2. `goal_detect.py` vspan fix + pixel mouth + entry rule.**
`detect_goals`/`ray_plane_point`/`goal_frame_equirect` ignored vspan (default 180);
now threaded through (v2 clips have vspan≈50). Added `goal_mouth_polygon`,
`in_goal_mouth_px`, `detect_goals_px`, and `detect_goal_entries`. Lateral width
stays from the GROUND position (ray-plane lateral is only valid ON the plane; using
it broke the wide-shot case). `detect_goal_entries` is now a **goal-line CROSSING**
rule: find sign changes of (ground x − goal line), interpolate the ray-plane (y, z)
to the crossing instant, and accept iff that point is inside the posts and under the
bar. This is the only place the ray-plane lateral is valid (before/behind the ball
it is just a projection). It also requires the crossing to be INTO the goal and a
plausible approach speed, and keeps the FIRST inward crossing per ~6 s. Adversarial
tests: wide-fast, over-bar, static keeper, keeper shuffle, ball crossing far in front
of the goal → all 0 events; roll-in and enter-then-exit → 1.

**3. Lock-on acquisition + dense tracking (`ball_lockon.py`).**
Grid scan over the whole content (tiling restricted to the pitch polygon DROPS the
boundary windows and with them the ball at the goal line — the polygon is applied to
candidates instead, ±60 px). Acquisition builds greedy velocity chains over 5 scans
(~0.42 s apart) and scores `mean_conf·(0.6+0.4·straightness)` among chains with
speed 8–500 px/s. This rejects static FPs (posts/keeper/scoreboards) and erratic
ones. `pitch_polygon` returns the ordered projected border loop (off-frame goal
corners kept) instead of a convex hull. Dense follow = one 300 px local window per
frame.

Measured on the 18-label Barca sequence (t=1273–1308), **no GT seeding**:
- Acquisition locks the ball; trail hits **f0–f5** (t=1273–1283) at **0–2 px =
  5/6 goal-sequence labels**, equal to the hand-seeded baseline.
- f7 (1287) misses by 64 px: the ball is occluded by players/keeper exactly at the
  goal. The tracker then latches a static high-conf object near the goal for the
  rest of the clip (no re-acquire wired into the dense loop). Labels 1297–1307
  (later play) miss.
- Goal crossing rule on this trail: **exactly one event, left (image-right) goal,
  t≈1287.56 s, approach 364 px/s, ray-plane z=-2.64 m (inside the 3.66 m posts); zero
  on the right goal.** This matches the labelled goal. Caveat: the ball is occluded
  at the line, so the exact crossing is interpolated between nearby detections
  (including the object the tracker shifted onto); the crossing z/y are physically
  plausible and the timing matches. This partly revisits the "tiling/zoom exhausted"
  note above: general tiling fails, but motion-ranked acquisition + dense local
  tracking works.

Debug/prototypes: `out/_lockon_eval.py`, `out/_goal_rule_eval.py`,
`out/_acq_scans.json`, `out/_lockon_trail.json` (git-ignored). Tests:
`test_goal_detect.py` (new) + the four existing suites ALL PASS.
NOTE: `app180.py` still needs a restart after recalibration to use the anchored fit.

**4. Pipeline integration (optional `--goals`).** `vr180_pipeline.run(goals=True)`
builds a `BallLockOn` on the content frames once W/H are known: while searching it
buffers 5 frames ≥0.3 s apart and calls `acquire` (max 3 attempts, 5 s cooldown on
failure); while locked it follows per frame. After the pass it runs
`detect_goal_entries` on both the lock-on trail and the full-frame `ball_trail`,
dedupes one event per side per 5 s (prefers lock-on), and adds
`goal_events` + `ball_lockon_trail` to the output. Off by default. Smoke run
(170 frames from 1282 s): acquired the ball at 1283.4 s and emitted one left-goal
event at 1287.71 s (approach 312 px/s). Note: 4K player tiling dominates runtime
(~1.5 s/frame); the acquisition scan adds ~10 s once.

**=== SESSION 2026-11-26 (cont.) — STEP 4 PROBE: ZOOM FINDS THE FAR BALL ===**
Product constraint confirmed by the user: **no behind-goal camera** — ball/goal must
work from the normal wide view ([[product-camera-setup]]). New module `ball_zoom.py`.
Validated against the 18 hand labels (`annotation_goals/ball_gt (2).json`, Barca
goal at 21:25 = 1285 s; labels are exact — the old script's fixed goal crop simply
did not cover the ball, hence the earlier 0/9):
- **Patch detection:** a ball detector on a 200 px crop upscaled 3× finds the ball at
  **0.35–0.90 conf within 0–3 px of every positive label (9/9)**. The ball IS there;
  it is just 8–16 px and too small for the full-frame pass.
- **Grid zoom scan** (240 px windows, 120 px stride, 139 windows over the pitch
  polygon, 3× upscale): candidates within 20 px of GT in **6/6** labeled frames of the
  goal sequence (0–2 px), at ~2.1 s/frame. Candidate density ~14/frame after the
  pitch-polygon filter — association, not detection, is the hard part.
- **4 fps association** fails (constant-velocity/beam search: ball lost after the
  first few seconds when it accelerates). Existing `BallTrailTracker` reaches 3/6.
- **Dense local tracking** (one 300 px window around the constant-velocity prediction,
  24 fps): **5/6 within 20 px (0–2 px on five labels)**, 27 ms/frame. Only the
  near-goal frame at 1287 is 73 px off (occlusion/timing). This is the viable design:
  cheap per-frame local tracking + periodic grid scan for re-acquisition.
- Remaining work: acquisition policy (which grid candidate to lock onto — prefer
  fast/persistent tracks, or restrict to goal areas), and goal-crossing logic in
  metres using the now-valid calibration. Prototypes: `out/_zoom_eval.py`,
  `out/_track_assoc.py`, `out/_dense_track.py`.
- **Super-resolution / upscaler repos considered — rejected.** Zoom cannot add real
  pixels (the ball is ~10×10 real px), and SR can hallucinate detail at that size;
  the bottleneck is acquisition (14 candidates/frame), not resolution. Optional free
  test next session: sweep zoom factor (2/3/4×) and imgsz on the 18 labels.

**=== SESSION 2026-11-26 (cont.) — STEP 3: CALIBRATION FITTED IN PIXELS ===**
Diagnosis: the old solver minimised **metres on the ground**, so far-field traces
could be 100+ px wrong while the metric median looked fine (0.60 m / 144 px). The
residual pattern was radial (centre <1 m, goal lines 80 m) → the assumed **span=180
was wrong**, not just vspan. A span sweep with vspan free pinned the effective
projection at ~149°×61° for the Barca clip (vspan re-fit alone barely moved; the
horizontal span was the error).

**New solver** (`spherical_calib.fit_pixel`, used by `calibration_from_traces`):
- Minimises **traced-pixel → densely projected model-curve** distance (KD-tree),
  optimising pose + span + vspan. Robust `soft_l1`.
- Metric solve stays as the seed/parity chooser; pixel fit refines.
- Auto-drops mistraced labels (median > max(10 px, 3× overall); never the last goal
  line; max 2) and refits. A dropped line is still drawn in the overlay as a check.
- Validity gate is now **pixel-based**: median ≤ 0.45 % of image width (~15 px at
  3400) + corners ≥ 2 + ≥ 12 points. Contract version → **v2** (old v1 calibrations
  keep their 180 assumption and are NOT trusted for detection spans).

Measured:
| trace | before | after |
|---|---|---|
| synthetic (pose known, span 180/180, 2 px noise) | 0.31 m / 9.5 px | **1.15 px**, span 179.6/178.5, pose ~0.2 m |
| user trace `calib/calib_equirect.json` | 0.60 m / 144 px | **5.40 px**, span 148.7/61.1, med_m 0.16, drops `touchline_far`, valid |
| app trace `7d1b8b0b_trace.json` | 3.64 m / 875 px, invalid | **10.92 px**, span 153.5/76.4, med_m 1.09, drops `goal_line_right`, valid |

The dropped far touchline is still predicted onto the painted line in the overlay —
an independent validation. The app trace's ~10 px floor is hand wobble on its
145-point near edge (the tool drags a pen path).

**Easier calibration (visual):** `calib/calib_overlay.py` draws the fitted pitch on
the traced frame (JPEG via `/api/calibrate` → `overlay_url`) and returns
`model_curves` so the trace tool draws the fit as **white dashed lines** in the
canvas. The status shows `fit N px`, validity and suspected labels, so the user
knows exactly which edge to redo. Paint-snap prototype: marginal (5.4→4.5 px on the
good trace, no gain on the app trace) — **not adopted**.

**Pipeline integration:** when a v2 calibration is valid, detection uses its solved
span/vspan (26 vs 24 players in a 3-frame spot check) and the follow-cam projects
with the same geometry. Explicit `--span/--vspan` win; v1/invalid calibrations fall
back to span 180 + content-aspect vspan. The app's vspan field is now blank by
default (placeholder shows the implied value). NOTE: a running `app180.py` must be
restarted to pick up all of this.

Tests: `test_metric_roi`, `test_vspan_tiling`, `test_ball_trail`, `test_trace_roi`
ALL PASS. Debug/experiment scripts live in `out/_pixel_fit.py`, `out/_pxdist_lsq.py`,
`out/_snap_paint.py` (git-ignored).

**=== SESSION 2026-11-26 — STEP 2: SHORT SAMPLES ON ALL SIX CLIPS ===**
Harness: `scripts/sample_clips.py` (12 frames, step 2, app-equivalent defaults, no
calibration) → `out/short_samples/summary.json`. Player detections/frame (median):

| clip | before | after |
|---|---|---|
| barca_04b_blue | 21.5 | 27.0 |
| barca_04b_chall | **3.0** | 25.5 |
| barca_11b_garnet | **8.5** | 22.0 |
| night_wolves | **2.0** | 32.5 |
| portrait_pasc | 23.0 | 23.0 |
| stereo_viewpt | 15.5 | 15.5 |

Two cheap bugs found and fixed (visual checks confirm boxes on players):
1. **Vertical span was assumed 180 by default.** Tile projection with the wrong vspan
   stretches players tall-and-thin in the tiles and the detector misses them
   (Challenger: 3 → 29 at vspan 108). `vr180_pipeline` now infers vspan from the
   content aspect (180·H/W, clamped 20–180) unless `--vspan` is given; the app
   prefills the same value (was: only when implied < 60). Recorded as `vspan_source`.
   **Implication for step 3:** the stored calibrations assume vspan 180, which for
   the Barca clip's 2.13-aspect content is likely also wrong — revisit in calibration.
   The pipeline now prefers the aspect value over the calibration's vspan.
2. **Night auto-ROI destroyed recall.** The grass-colour hull segments only the two
   floodlit patches (x≈0–1115 of 3622) and kept 1–2 of ~28 detections. New guard:
   a grass ROI keeping < 15 % of the frame's detections (≥ 8 detections) is ignored
   (`roi_mode = "grass-rejected"`). User traces are still authoritative.

Still open (not cheap): day-clip ball 0/12 (known far-ball, step 4); portrait/stereo
content implies vspan > 180 (clamped to 180 — see step 3 geometry); occasional
duplicate boxes near dense clusters. Unit tests still pass (`test_vspan_tiling`,
`test_ball_trail`, `test_trace_roi`, `test_metric_roi`); full app-path smoke test with
calibration+trace+render+follow OK (23–24 players/frame, follow view sane).

**=== SESSION 2026-11-25 (end) — RESUME STATE ===**
- **Virtual follow-camera + MP4 DONE (task #17):** `vr180_pipeline.py --render-follow`
  (smoothed pose → `equirect_to_perspective`, FOV 70, 1280×720, h264); app `/api/follow/{id}`
  + second video in `index180.html`. Verified on Barca 15 frames. **Flat pipeline untouched.**
- **Ball findings (recorded above):** full-frame 0/9, tiling 1/9, high-res 0–1/9, isolated
  crop 9/9 → current model can't detect far goal balls in context. **Far-camera goals
  PARKED** (retrain or closer cam needed; not the product setup).
- **18 hand labels exist:** `annotation_goals/ball_gt (2).json` (9 pos / 9 neg) — measuring
  stick for any future ball work. `annotation/build_goal_sheet.py` regenerates the sheet.
- **Ball-trail static-gate fix REVERTED** (coverage↑ but 401 px outliers + unit-test fail);
  tracker stays 70% / 2.4 px. `test_ball_trail` + `test_vspan_tiling` PASS.
- **NEXT (say "continue"):** read `PLAN.md` → **Step 2** = run SHORT SAMPLES of the 6 clips
  (night, stereo, portrait, wide-strip) to surface + fix cheap failures. Not full matches.
- **Waiting on user only for:** a closer-camera clip (unlocks goals) and camera specs
  (model/resolution/FOV → makes calibration easier). Neither blocks Step 2.
- **Git:** all changes are in the working tree, UNCOMMITTED (not committed, per no-unasked-commit rule).
- **Naive goal-region zoom + tracker fails (FP lock):** `out/_goal_zoom_track.py` over
  21:15–21:45 locks onto the **static keeper/goal-post** (conf ~0.5, net~0, stuck id)
  instead of the ball → 0 GT matches. The static gate only rejects conf < `high_conf`
  (0.5), so a ~0.5–0.6 keeper slips through. **Fix direction:** multi-scale ball pass +
  stronger static-FP handling (raise `high_conf`, or moving-only for goal-region tracks).
  Consistent with [[ball-native-tiling-falsified]] (players↔ball confusion); no model
  retrain planned yet.

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
| **Ball detection on VR180** | ⚠ **IMPROVED 2026-11-24** — temporal trail tracker on low-confidence full-frame candidates: **70% GT coverage, 2.4 px median** on the labeled drone clip; follows a Barca VR180 ball through 37 s. Confirmed 2026-11-25: the 15 GT misses are **detector gaps**, not tracker drops. Night/wide clips still candidate-sparse; retrain optional. |
| Real calibration | Approximate (`calib/calibration.json`, med 0.82 m). Metre mapping is self-consistent (2026-11-25). |
| Goal detection on real footage | ⚠ **BLOCKED 2026-11-25 — goal event not located.** At 36.35 s the ball is at midfield; a 3× zoomed scan of both traced goal mouths over 0–120 s finds no ball entering a goal. Logic vouched by the synthetic self-test (`calib/goal_detect.py`). Awaiting the exact time/screenshot. |
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
- **Vertical span beats upscaling (2026-11-24):** a 3621×796 night strip at vspan=40 gives **29–36 detections/frame** (median conf 0.73) vs **1–5 at vspan=180**. 2× Lanczos upscaling and imgsz1536 did **not** help; CLAHE does.
- **Ball tracking works on the labeled clip (2026-11-24):** low-confidence candidates + a motion tracker give **70% GT coverage at 2.4 px median**; the old top-1/validator picked static FPs. On the Barca VR180 clip the tracker follows a white ball through 37 s — but at 36.35 s that ball is at **midfield**, not a goal (see #13, 2026-11-25).
- Equirect **span=180 has no horizontal seam** (that only exists for 360).
- All equirect coordinates are relative to the **cropped content region**; the ball trail was once in full-frame coords → offset by the crop origin.

## 3. Architecture (module map)

```
spherical_projection.py   equirect <-> perspective (equirect_to_perspective; hspan+vspan)
spherical_tiling.py       detect_content_region, detect_span, build_grid(vspan), tile_equirect(vspan),
                          hemisphere_mask(vspan), tile_pixel_to_equirect(vspan),
                          merge_to_equirect (centre-NMS, vspan),
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
ball_trail.py             temporal tracker for low-confidence full-frame ball candidates
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

**SUPERSEDED 2026-11-26 (detection):** the default vspan = 180 was badly wrong for
detection — aspect-derived vspan (option c) recovers 2.5–16× more players per frame
and is now the pipeline default (see the 2026-11-26 session block). The "round
centre circle at vspan 180" observation above should be re-checked, since a 2.13
aspect under a 180×180 equirect implies squeezed/non-square pixels; calibration
(step 3) must settle it.

**Second finding:** one in six clips is **stereo** (SBS) — `equirect.py` now
auto-detects and de-stacks (takes one eye).

**2026-11-24 update:** a fixed `vspan` is now propagated through tiling/merge and exposed
as `--vspan` (the app auto-prefills ~40 when the content aspect implies <60°). Solving
vspan jointly with the pose remains off (ill-conditioned); for detection, supply the
camera's real vertical FOV. `--enhance auto` applies CLAHE to dark frames.

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
- **Vertical span is a first-class tiling parameter (2026-11-24).** `vspan` flows through
  the projection, tiling, inverse mapping, mask, merge and grid (default 180). Wide
  letterboxed strips (aspect >3) use one 36° pitch row and vspan≈40; this is a much larger
  detection gain than any upscaler. Calibration metrics still use the calibration's
  `vspan_deg` until calibration-vspan solving is wired.
- **Ball = temporal trail tracker on low-conf full-frame candidates (2026-11-24).**
  No threshold/validator/upscale: `ball_trail.py` links precise low-confidence candidates,
  rejects static FPs by net motion, and predicts gaps. The app ball is enabled; no retrain
  yet. Night/wide clips remain candidate-sparse.
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
