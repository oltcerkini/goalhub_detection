# GoalHub VR180 — Plain-Language Plan

_Read this first. `VR180_ROADMAP.md` keeps the technical detail; this file stays short
and I update it after each step. You do NOT need to track the technical threads._

## The goal

Upload a 180° match video → the app finds **players + ball** → positions/distances in
**metres** → simple **events (goals)** → a **follow-camera video**.

## How we work

- **I pick the next step** from this plan, do it, and update "Status" here.
- **You only do something where a row says "needs you."** Everything else I handle.
- I don't move on claiming success without a measurement.
- If you ever feel lost: just read the **Current step** line and the **Needs you** section.

## Current step

**RESUME HERE (say "continue"):** saved 2026-11-27. **Steps 1 ✅, 2 ✅, 3 ✅ done.
Step 4 prototyped end-to-end.** The far ball is found in 9/9 hand-marked frames when
the program zooms; **lock-on now acquires the right moving spot on its own** and
follows it at 0–2 px on 5 of the 6 goal-sequence checkpoints (same as when seeded by
hand). The **goal rule fires once** at the correct goal in your Barca clip. Nothing
is blocked on you.

**What I built this session (all measured):**
- **Calibration now anchors the painted goal frames.** Your trace already contained
  136 goal-frame points (posts + crossbar) that were being thrown away; they are now
  used, and the far-goal fit error dropped from **17–40 px to ~3–5 px**. That is the
  accuracy the goal rule needed. (Camera height also became plausible, ~4 m.)
- **Lock-on** (`ball_lockon.py`): a scan finds ~20–60 ball-like spots; it now picks
  the ball by requiring confidence **and** smooth, plausible motion. Static lookalikes
  (posts, keeper, scoreboards) and erratic spots are rejected.
- **Goal crossing rule** (`calib/goal_detect.py`): it looks for the exact moment the
  ball's path crosses the goal-line plane and checks the goal mouth *at that instant*
  (the only place the maths is exact). On your clip it reports exactly **one goal at
  the correct (right-in-image) goal at ~1287.6 s** and none at the other goal.
  Synthetic checks all behave: wide shot, over the bar, keeper standing still,
  keeper shuffling, ball crossing far in front → **no goal**; a normal roll-in → one.

**Honest caveat:** the ball is briefly occluded by players/keeper exactly as it
crosses, so at the goal instant the tracker latches onto a high-confidence object
near the goal (the trail is ~65 px off the hand label at 1287). The event is still
right here because the ball approached fast and the goal area was entered — but the
ball's identity *inside the net* is not verified. That is the next thing to harden
(re-acquire + motion-into-goal evidence), not a calibration problem.

- **Wired into the pipeline** as an optional stage: `vr180_pipeline.py --goals`
  emits `goal_events` (plus the lock-on trail). Verified end-to-end on the clip:
  acquired the ball itself at 1283.4 s and reported **one left-goal event** at the
  crossing (approach ~300 px/s). Off by default, so existing behaviour is unchanged.
- **False-positive pass + acquisition hardening:** on ordinary stretches of the full
  match (no goal) the ball is now acquired and followed for **97 % of frames with
  zero goals fired**. This pass also caught a real weakness — the auto-find sometimes
  locked a linesman/foot instead of the ball — so acquisition now only considers
  confident objects well inside the pitch. After the fix the goal clip still gives its
  one correct goal. (No ground truth for the ordinary stretches, so this is
  encouraging, not proof.)
- **App wiring:** `app180` auto-enables `--goals` only when the calibration is
  v2-valid **and** the trace includes the goal frames, and the results panel shows
  "Goals: N — …". Existing saved calibrations are v1, so **recalibrate in the app**
  to switch it on. (The app's stored clip is a 30 s excerpt, so I couldn't exercise
  the goal path through the UI without uploading the full video.)

**Next, in order:**
1. **False-positive pass**: run `--goals` on the other clips / non-goal stretches
   and make sure nothing spurious fires (needs a valid calibration with goal frames
   for at least one more clip).
2. **Re-acquire after loss** in the dense loop (currently the tracker can sit on a
   static object near the goal once the real ball is occluded), and surface goal
   events in the app UI.
3. Then step 5 (stats) — goals are the last detection piece.

✅ **Step 1 done** — virtual follow-camera + MP4 built and verified (clean perspective
view that pans to the action; new video in the app).

✅ **Step 2 done** — all six clips ran for short slices. Two big cheap bugs fixed:
player detection was being crippled by a wrong default vertical span (one clip found
only 3 players/frame where there are ~25) and the night clip's auto pitch-mask was
throwing away almost every player. After the fix every clip finds its players:
~22–33 per frame across normal/night/stereo/portrait/wide.

✅ **Step 3 done** — calibration is now fitted in **pixels** (it lands on the paint,
not just "low metres on average") and the earlier assumption that the video spans
180°×180° was wrong: the solver now finds the real spans (~149°×61° for your Barca
clip). On your original trace the fit error dropped from ~145 px to **5.4 px**, and
the calibration now produces metre positions with a median error under a metre.
You can also **see the fit**: after calibrating, white dashed lines show the fitted
pitch on your frame — if they sit on the painted lines, it's good. No action from you.

## What already works

- ✅ 180° video loads, crops correctly, handles stereo/night/portrait layouts
- ✅ **Players** detected + tracked well across the pitch — now consistent across all
  six clips (normal, night, stereo, portrait, wide) after the vertical-span fix
- ✅ Camera calibration from your traced pitch → positions in metres — now fitted in
  pixels (the projected pitch lands on the painted lines; ~5 px on a good trace) and
  **verifiable on screen** (white dashed overlay after calibrating)
- ✅ Off-pitch filtering (spectators/benches dropped)
- ✅ Web app: upload → trace pitch → process → **annotated MP4** back
- ✅ **Ball** tracked reliably on normal/closer footage (70% frames, ~2 px median)
- ✅ **Virtual follow-camera** — a perspective "follow shot" rendered to MP4, shown in the app

## The plan, in order

| # | Step | Who | Status |
|---|------|-----|--------|
| 1 | **Virtual follow-camera + MP4** — smooth camera that follows the action; render a clean MP4 | me | ✅ **Done** |
| 2 | **Run the pipeline on all your clips** (night, stereo, portrait, wide) and fix the cheap failures | me | ✅ **Done** |
| 3 | **Make calibration easier + more accurate** | me + your camera specs | ✅ **Done** — your specs are optional now; the spans are solved from the pitch |
| 4 | **Ball near goals / goal detection** — must work from the normal match camera (not behind-goal) | me | **in progress** — auto lock-on + goal entry rule prototyped: 5/6 checkpoints at 0–2 px, one correct goal event; next: integrate + false-positive pass |
| 5 | **Events & analytics** (possession, shots, goals, heatmaps) from metre positions | me | after 3/4 |

## The few things I actually need from you (only when listed)

1. **~15–20 labeled frames** — ONLY if we choose to retrain the ball (step 4), from the
   normal match camera view. **Not now** — first I'll try the no-new-data fix.

That's the whole list. If you have none of these handy, it's fine — steps 1–2 need
nothing from you.

## Known limits (so nothing surprises you)

- **The normal match camera means a ~10 px ball at the far goal.** That's the real
  target (a behind-goal camera is not the product), so step 4 has to beat that limit
  with smarter processing — zooming into the goal area and following movement,
  ignoring still things like the keeper/post — or with a small retrain on your view.
  Not failed; a known hard step.
- **Calibration accuracy now depends mainly on tracing neatness** (a 145-point
  hand-drawn edge wobbles by ~10 px). The app shows the fit so you can see when a
  line needs redoing. Camera specs are no longer needed — the spans are solved.
- Player detection is considered **done** and is not being changed.
