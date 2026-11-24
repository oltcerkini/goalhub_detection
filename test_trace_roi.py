"""Test the calibration-trace ROI (no pose involved)."""
import numpy as np

from spherical_tiling import pitch_polygon_from_trace, trace_roi_filter


def _square_trace(size=(1000, 600)):
    w, h = size
    return {"size": list(size), "lines": [
        {"label": "touchline_far", "points": [[0.1 * w, 0.1 * h], [0.9 * w, 0.1 * h]]},
        {"label": "goal_line_right", "points": [[0.9 * w, 0.1 * h], [0.9 * w, 0.9 * h]]},
        {"label": "touchline_near", "points": [[0.9 * w, 0.9 * h], [0.1 * w, 0.9 * h]]},
        {"label": "goal_line_left", "points": [[0.1 * w, 0.9 * h], [0.1 * w, 0.1 * h]]},
        # legacy goal-shape duplicate, shorter -> must be ignored
        {"label": "goal_line_left", "points": [[0.15 * w, 0.85 * h], [0.15 * w, 0.95 * h]]},
    ]}


def main():
    trace = _square_trace()
    poly = pitch_polygon_from_trace(trace, 1000, 600)
    assert poly is not None and len(poly) == 8, poly
    assert poly[:, 0].min() == 100 and poly[:, 0].max() == 900
    assert poly[:, 1].min() == 60 and poly[:, 1].max() == 540

    inside = {"bbox": [400, 200, 500, 450], "conf": 0.9}       # ground (450,450)
    above = {"bbox": [50, 10, 150, 50], "conf": 0.9}           # ground above far line
    right = {"bbox": [950, 200, 1050, 450], "conf": 0.9}       # ground right of goal line
    kept, dropped = trace_roi_filter([inside, above, right], poly, margin_frac=0.0)
    assert kept == [inside], kept
    assert dropped == [above, right], dropped

    # Near the camera a tall box can overhang the line: proportional margin keeps it.
    overhang = {"bbox": [400, 300, 500, 550], "conf": 0.9}     # bottom 10px past near line
    kept, dropped = trace_roi_filter([overhang], poly, margin_frac=0.0)
    assert kept == [], "strict margin must drop it"
    kept, dropped = trace_roi_filter([overhang], poly, margin_frac=0.20)
    assert kept == [overhang], "proportional margin must keep a boundary player"

    # Trace coordinates are scaled to the pipeline's crop size.
    poly2 = pitch_polygon_from_trace(trace, 2000, 1200)
    assert poly2[:, 0].min() == 200 and poly2[:, 0].max() == 1800
    assert poly2[:, 1].min() == 120 and poly2[:, 1].max() == 1080

    # Missing an edge -> no polygon (caller falls back to pose/grass).
    partial = _square_trace()
    partial["lines"] = [ln for ln in partial["lines"] if ln["label"] != "touchline_far"]
    assert pitch_polygon_from_trace(partial, 1000, 600) is None

    print("test_trace_roi: ALL PASS")


if __name__ == "__main__":
    main()
