"""Goal rules: vspan threading, metre-plane rule, and the gap-tolerant entry rule."""
import numpy as np

from calib.goal_detect import (GOAL_HEIGHT, PITCH_LEN, _trail_from_3d, detect_goals,
                               detect_goals_px, detect_goal_entries, synthetic_test,
                               goal_line_x, pitch3d_to_equirect)

W, H = 3401, 1586
POSE = {"P": np.array([26.0, 9.5, -58.0]),
        "rvec": np.radians([18.0, 62.0, 7.0])}
SIDE = "right"
FPS = 24.0


def make_trail(pts3, span=180.0, vspan=180.0, noise=0.0):
    frames = np.arange(len(pts3))
    return _trail_from_3d(pts3, POSE, W, H, span, frames, noise=noise, vspan=vspan)


def cases():
    xg = goal_line_x(SIDE)
    n, f = 60, np.arange(60)
    return {
        "roll-in": np.stack([np.linspace(xg - 12, xg + 3, n), np.zeros(n), np.zeros(n)], 1),
        "wide": np.stack([np.linspace(xg - 12, xg + 3, n), np.zeros(n), np.full(n, 8.0)], 1),
        "over-bar": np.stack([np.linspace(xg - 12, xg + 3, n), np.full(n, 3.5), np.zeros(n)], 1),
        # a keeper standing in the mouth for the whole clip: no approach, no event
        "static-in-mouth": np.stack([np.full(n, xg - 0.5), np.full(n, 1.0),
                                     np.full(n, 0.5)], 1),
    }


def test_metre_rule_fires_only_for_roll_in():
    res = {}
    for name, pts in cases().items():
        trail = make_trail(pts)
        res[name] = len(detect_goals(trail, POSE, SIDE, W, H, 180.0, 180.0, consecutive=3))
    # the metre rule has no approach test, so a static ball in the mouth DOES
    # count here; approach rejection is the entry rule's job (next test).
    assert res["roll-in"] > 0, res
    assert res["wide"] == 0, res
    assert res["over-bar"] == 0, res
    print("  metre rule: PASS", res)


def test_vspan_threading():
    """The same 3D roll-in must be detected when the image uses a narrow vspan."""
    span, vspan = 150.0, 70.0
    trail = make_trail(cases()["roll-in"], span=span, vspan=vspan)
    ev = detect_goals(trail, POSE, SIDE, W, H, span, vspan, consecutive=3)
    assert len(ev) > 0, "vspan not threaded through the metre rule"
    ev_px = detect_goals_px(trail, POSE, SIDE, W, H, span, vspan, consecutive=3)
    assert len(ev_px) > 0, "vspan not threaded through the pixel rule"
    print("  vspan threading: PASS (metre + pixel rules fire)")


def test_entry_rule_one_event_and_rejects_static():
    n = 90
    t = np.arange(n) / FPS
    xg = goal_line_x(SIDE)
    # ball rolls in over ~1.2 s then rests in the net for the rest of the clip
    xs = np.clip(np.linspace(xg - 10, xg + 2, n), xg - 10, xg + 2)
    roll = np.stack([xs, np.zeros(n), np.zeros(n)], 1)
    ev = detect_goal_entries(make_trail(roll), POSE, SIDE, W, H, 180.0, 180.0,
                             fps=FPS, min_approach_px_s=20.0)
    assert len(ev) == 1, f"expected exactly one entry event, got {len(ev)}"
    # the keeper standing in the mouth must never fire (no approach)
    ev2 = detect_goal_entries(make_trail(cases()["static-in-mouth"]), POSE, SIDE,
                              W, H, 180.0, 180.0, fps=FPS)
    assert len(ev2) == 0, f"static keeper produced {len(ev2)} events"
    print("  entry rule: PASS (1 roll-in event, 0 for static keeper)")


def test_entry_rule_false_positives():
    xg = goal_line_x(SIDE)
    fps = FPS
    n = int(3 * fps)
    tt = np.arange(n) / fps
    cases = {
        # fast shot that goes wide of the posts (z beyond 3.66 m)
        "wide-fast": np.stack([np.linspace(xg - 12, xg + 3, n), np.zeros(n),
                               np.full(n, 5.0)], 1),
        # keeper shuffling inside the mouth: no approach from outside
        "keeper-shuffle": np.stack([np.full(n, xg - 0.5), np.full(n, 1.0),
                                    0.6 + 0.5 * np.sin(2 * np.pi * 2 * tt)], 1),
        # ball crossing 30 m in front of the goal (outside the depth gate), even
        # if its pixel ray lines up with the mouth
        "far-across-mouth": np.stack([np.full(n, xg - 30.0), np.zeros(n),
                                      np.linspace(-10, 10, n)], 1),
    }
    for name, pts in cases.items():
        ev = detect_goal_entries(make_trail(pts), POSE, SIDE, W, H, 180.0, 180.0,
                                 fps=fps)
        assert len(ev) == 0, f"{name} produced {len(ev)} events"
    # enters and comes back out (rebound off the net): still one goal event
    xb = np.concatenate([np.linspace(xg - 8, xg + 2, n // 2),
                         np.linspace(xg + 2, xg - 8, n - n // 2)])
    back = np.stack([xb, np.zeros(n), np.zeros(n)], 1)
    ev = detect_goal_entries(make_trail(back), POSE, SIDE, W, H, 180.0, 180.0, fps=fps)
    assert len(ev) == 1, f"enter-then-exit expected 1 event, got {len(ev)}"
    print("  entry-rule false positives: PASS (wide/keeper/far=0, enter-exit=1)")


if __name__ == "__main__":
    print("goal detect tests:")
    r = synthetic_test(verbose=False)
    assert r["pass"], r
    print("  synthetic roll/wide/over/far: PASS")
    test_metre_rule_fires_only_for_roll_in()
    test_vspan_threading()
    test_entry_rule_one_event_and_rejects_static()
    test_entry_rule_false_positives()
    print("ALL PASS")
