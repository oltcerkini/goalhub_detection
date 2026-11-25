"""Synthetic checks for the temporal ball trail tracker."""
from ball_trail import BallTrailTracker


def main():
    # moving ball + a static high-ish confidence FP
    tr = BallTrailTracker(gate_px=30.0, min_hits=3, max_missed=5)
    out = []
    for i in range(25):
        out.append(tr.update(i, [
            {"x": 50.0 + i * 3.0, "y": 200.0, "conf": 0.10},
            {"x": 100.0 + (i % 2), "y": 100.0, "conf": 0.40},
        ]))
    last = out[-1]
    assert last is not None, "moving trail not emitted"
    assert abs(last["x"] - (50.0 + 24 * 3.0)) < 8, last
    assert abs(last["y"] - 200.0) < 8, last

    # static-only candidates must not produce a trail
    tr2 = BallTrailTracker(gate_px=30.0, min_hits=3, max_missed=5)
    for i in range(20):
        assert tr2.update(i, [{"x": 100.0 + (i % 2), "y": 100.0,
                               "conf": 0.40}]) is None

    # prediction across a two-frame candidate gap
    tr3 = BallTrailTracker(gate_px=30.0, min_hits=3, max_missed=5)
    for i in range(10):
        tr3.update(i, [{"x": 50.0 + i * 4.0, "y": 200.0, "conf": 0.20}])
    p10 = tr3.update(10, [])
    p11 = tr3.update(11, [])
    assert p10 is not None and abs(p10["x"] - 90.0) < 12, p10
    assert p11 is not None and abs(p11["x"] - 94.0) < 14, p11
    assert p10.get("last", 9) == 9

    print("test_ball_trail: ALL PASS")


if __name__ == "__main__":
    main()
