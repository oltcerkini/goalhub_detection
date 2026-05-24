"""Interactive pitch calibration — user clicks corners and goals via OpenCV."""

import cv2
import numpy as np


class PitchCalibrator:
    """Two-phase calibration: corners (≥4 pts), then optional goals (2 pts)."""

    CORNER_COLOR = (0, 255, 0)    # green
    GOAL_COLOR = (0, 255, 255)    # yellow
    POLYGON_COLOR = (255, 255, 0)  # cyan
    TEXT_COLOR = (255, 255, 255)

    def __init__(self):
        self.polygon = None   # (N, 2) int32 — pitch boundary
        self.goals = []       # list of (x, y) — goal centres
        self.calibrated = False
        self._points = []     # currently placed points
        self._phase = "corners"
        self._image = None
        self._overlay = None
        self._window = "Calibration"

    def calibrate(self, image, window="Calibration"):
        """Launch interactive calibration. Returns True on success."""
        self._image = image.copy()
        self._overlay = image.copy()
        self._window = window
        self._points = []
        self._phase = "corners"
        self.calibrated = False
        self.goals = []

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)

        self._draw_instructions()
        print("\n=== PHASE 1: CORNERS ===")
        print("Click the 4 pitch corners in order (clockwise or counter-clockwise).")
        print("Press  ENTER  when done,  C  to undo last point,  ESC  to cancel.")

        while True:
            cv2.imshow(window, self._overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:   # Enter
                if self._phase == "corners":
                    if len(self._points) >= 4:
                        self._finalise_corners()
                        self._start_goals()
                    else:
                        print(f"  Need ≥4 corners (have {len(self._points)})")
                elif self._phase == "goals":
                    self._finalise_goals()
                    break

            elif key == ord('c'):
                if self._points:
                    self._points.pop()
                    self._refresh()
                else:
                    print("  Nothing to undo")

            elif key == ord('s') and self._phase == "goals":
                # Skip goals
                self._finalise_goals()
                break

            elif key == 27:  # ESC
                if self._phase == "goals":
                    # Go back to corners
                    self._phase = "corners"
                    # Rebuild corners from all points minus goals
                    self._points = self._points[:]
                    self._refresh()
                    self._draw_instructions()
                    print("\n=== BACK TO CORNERS ===")
                else:
                    print("Calibration cancelled.")
                    cv2.destroyWindow(window)
                    return False

        cv2.destroyWindow(window)
        return self.calibrated

    # ------------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))
            self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        self._overlay = self._image.copy()
        if not self._points:
            return

        color = self.CORNER_COLOR if self._phase == "corners" else self.GOAL_COLOR
        for i, pt in enumerate(self._points):
            cv2.circle(self._overlay, pt, 6, color, -1)
            label = str(i + 1) + ("G" if self._phase == "goals" else "")
            cv2.putText(self._overlay, label, (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show polygon outline once we have ≥3 pts
        pts = np.array(self._points, dtype=np.int32)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            cv2.polylines(self._overlay, [hull], True, self.POLYGON_COLOR, 2)

        self._draw_instructions()

    def _draw_instructions(self):
        """Draw status bar at the bottom."""
        h, w = self._overlay.shape[:2]
        bar = np.zeros((40, w, 3), dtype=np.uint8)
        if self._phase == "corners":
            msg = f"Click {max(4 - len(self._points), 0)} more corner(s) | ENTER=done  C=undo  ESC=cancel"
        else:
            msg = f"Click 2 goals (or S to skip) | ENTER=done  ESC=back"
        cv2.putText(bar, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.TEXT_COLOR, 1)
        self._overlay[h - 40:h] = bar

    # ------------------------------------------------------------------
    def _finalise_corners(self):
        pts = np.array(self._points, dtype=np.int32)
        self.polygon = cv2.convexHull(pts).squeeze()
        if self.polygon.ndim == 1:
            self.polygon = self.polygon.reshape(-1, 2)

    def _start_goals(self):
        self._phase = "goals"
        self._points = []  # fresh points for goals
        self._refresh()
        self._draw_instructions()
        print("\n=== PHASE 2: GOALS (optional) ===")
        print("Click the centre of each goal. Press S to skip, ENTER to confirm.")

    def _finalise_goals(self):
        self.goals = list(self._points)
        self.calibrated = True
        n = len(self.goals)
        print(f"Calibration complete: {len(self.polygon)}-point pitch polygon, {n} goal(s).")
