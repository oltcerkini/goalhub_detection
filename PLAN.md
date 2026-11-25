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

**RESUME HERE (say "continue"):** saved 2026-11-25. **Step 1 ✅ done.** Next = **Step 2**.
All work is saved on disk (nothing was lost when you closed the terminal).

✅ **Step 1 done** — virtual follow-camera + MP4 built and verified (clean perspective
view that pans to the action; new video in the app). ➡️ **Now: step 2** (run short
samples of your clips and fix the cheap failures). No action from you.

## What already works

- ✅ 180° video loads, crops correctly, handles stereo/night/portrait layouts
- ✅ **Players** detected + tracked well across the pitch (this is solid)
- ✅ Camera calibration from your traced pitch → positions in metres (approximate)
- ✅ Off-pitch filtering (spectators/benches dropped)
- ✅ Web app: upload → trace pitch → process → **annotated MP4** back
- ✅ **Ball** tracked reliably on normal/closer footage (70% frames, ~2 px median)
- ✅ **Virtual follow-camera** — a perspective "follow shot" rendered to MP4, shown in the app

## The plan, in order

| # | Step | Who | Status |
|---|------|-----|--------|
| 1 | **Virtual follow-camera + MP4** — smooth camera that follows the action; render a clean MP4 | me | ✅ **Done** |
| 2 | **Run the pipeline on all your clips** (night, stereo, portrait, wide) and fix the cheap failures | me | **Next** |
| 3 | **Make calibration easier + more accurate** | me + your camera specs | needs you (specs) |
| 4 | **Ball near goals / goal detection** — only feasible on closer footage; otherwise retrain | me | **deferred** — needs you *or* a retrain |
| 5 | **Events & analytics** (possession, shots, goals, heatmaps) from metre positions | me | after 3/4 |

## The few things I actually need from you (only when listed)

1. **A closer-camera clip** — any game, ideally one goal, tell me roughly when. Unlocks
   real goal validation and the ball-at-goal path. **Not needed for steps 1–2.**
2. **Camera specs** — model, resolution, nominal field of view (horizontal/vertical).
   Makes calibration much easier and more accurate. **Optional now; big win for step 3.**
3. **~12 labeled frames** — ONLY if we choose to retrain the ball (step 4). **Not now.**

That's the whole list. If you have none of these handy, it's fine — steps 1–2 need
nothing from you.

## Known limits (so nothing surprises you)

- **Far sideline cameras** (e.g. the Barca clip): the ball is ~10 px at the goal — it is
  genuinely too small for the model; closer cameras fix this. Goal detection on far
  footage is **parked**, not failed.
- **Calibration is approximate** right now (metres have error). Camera specs + a tight
  re-trace improve it; a closer/less-degenerate camera angle helps a lot.
- Player detection is considered **done** and is not being changed.
