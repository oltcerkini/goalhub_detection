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

**RESUME HERE (say "continue"):** saved 2026-11-26 (end of session). **Steps 1 ✅,
2 ✅, 3 ✅ done. Step 4 in progress.** The zoom idea is proven: the far ball is found
in **9/9** hand-marked frames when the program zooms in (it was 0/9 before), and
following it at full frame rate stays on target at **5/6** checkpoints. Nothing is
blocked on you.

**Next, in order:**
1. *(optional, ~5 min)* free zoom-size test — try different zoom amounts on your 18
   marked frames, keep whichever is most reliable. No downloads.
2. **Lock-on**: pick the right moving spot among the white lookalikes and follow it.
3. **Goal rule**: when the followed ball crosses the goal line (in metres), report a goal.
   Step 5 (stats) comes after.

*Note:* an AI upscaler / super-resolution repo was considered and **not adopted** —
zoom cannot add real pixels, and our bottleneck is picking the right spot, not
resolution. Revisit only if a zoom-size test shows the ball is barely visible.

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
| 4 | **Ball near goals / goal detection** — must work from the normal match camera (not behind-goal) | me | **in progress** — zoomed search finds the far ball in 6/6 labeled goal frames (was 0/9); next: auto lock-on + goal crossing |
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
