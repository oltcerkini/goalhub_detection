#!/usr/bin/env python3
"""
GoalHub Detection — Football Player Detection & Pitch Calibration.

Detects players in images or videos using rf-detr, calibrated to the pitch.

Usage:
    # Image mode
    python main.py match.jpg
    python main.py match.jpg --model nano --threshold 0.4

    # Video mode
    python main.py match.mp4
    python main.py match.mp4 --output annotated_match.mp4 --skip 2

Hotkeys (image mode):
    1 / 2     — pick my team
    Click     — select a player
    D         — delete selected player
    N         — name selected player
    S         — save results JSON
    Q / ESC   — quit

Video mode processes all frames and writes an annotated output file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detector import PlayerDetector
from pitch_calibrator import PitchCalibrator
from team_classifier import TeamClassifier
from ball_detector import BallDetector

# ---------------------------------------------------------------------------
WINDOW = "GoalHub Detection"

# BGR colours
COL_PITCH = (0, 200, 200)       # yellow-ish
COL_GOAL = (0, 255, 255)        # yellow
COL_SELECT = (255, 255, 0)      # cyan
COL_DELETED = (50, 50, 50)

# ---------------------------------------------------------------------------
# States (image mode)
# ---------------------------------------------------------------------------
S_CALIBRATE = 0
S_DETECTING = 1
S_CLASSIFYING = 2
S_SELECT_TEAM = 3
S_EDITING = 4

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}


# ===================================================================
# GoalHub Application
# ===================================================================
class GoalHubApp:
    def __init__(self, input_path, model_size="medium", threshold=0.5,
                 model_type="detection", output=None, skip_frames=1,
                 resize_long_side=1280):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        self.input_path = input_path
        self.threshold = threshold
        self.resize_long_side = resize_long_side
        self.output_path = output
        self.skip_frames = skip_frames
        self.is_video = self._is_video_file(input_path)

        if self.is_video:
            self._init_video()
        else:
            self.image = cv2.imread(input_path)
            if self.image is None:
                raise ValueError(f"Cannot load image: {input_path}")
            self._h, self._w = self.image.shape[:2]

        # State
        self.state = S_CALIBRATE
        self.selected_idx = -1
        self.deleted = set()
        self.player_names = {}
        self.my_team = None

        # Data
        self.polygon = None
        self.goals = []
        self.detections = None       # inside-pitch
        self.all_detections = None
        self.team_labels = None
        self.tracker = None

        # Modules
        self.detector = PlayerDetector(model_size=model_size)
        self.calibrator = PitchCalibrator()
        self.classifier = TeamClassifier()
        self.ball_detector = None

        # Display
        self._overlay = None
        self._info_text = ""

    # ------------------------------------------------------------------
    # Video initialisation
    # ------------------------------------------------------------------
    def _is_video_file(self, path):
        return Path(path).suffix.lower() in VIDEO_EXTS

    def _init_video(self):
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.input_path}")

        self.vid_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._h, self._w = self.vid_h, self.vid_w

        # First frame for calibration
        ret, self.image = self.cap.read()
        if not ret:
            raise IOError("Could not read first frame of video")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

        self._setup_tracker()
        print(f"Video: {self.vid_w}x{self.vid_h} @ {self.vid_fps:.1f} fps, "
              f"{self.vid_total} frames")

    def _setup_tracker(self):
        """Try to initialise ByteTrack for player IDs across frames."""
        try:
            import supervision as sv
            self.tracker = sv.ByteTrack()
            print("  Tracking enabled (ByteTrack)")
        except Exception:
            self.tracker = None
            print("  Tracking unavailable — per-frame detections only")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        if self.is_video:
            self._run_video()
        else:
            self._run_image()

    # ================================================================
    # IMAGE MODE
    # ================================================================
    def _run_image(self):
        cv2.namedWindow(WINDOW)

        # ---- Step 1: Calibrate ----
        if not self.calibrator.calibrate(self.image, window=WINDOW):
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return
        self.polygon = self.calibrator.polygon
        self.goals = self.calibrator.goals

        # ---- Step 2: Detect ----
        self.state = S_DETECTING
        self._info_text = "Detecting players…"
        self._render()
        cv2.waitKey(1)

        self._detect_and_filter(self.image)

        if self.detections is None or len(self.detections) == 0:
            self._info_text = "No detections — press Q to quit"
            self._idle()
            cv2.destroyAllWindows()
            return

        # ---- Step 3: Classify teams ----
        self.state = S_CLASSIFYING
        self._info_text = "Classifying teams…"
        self._render()
        cv2.waitKey(1)

        self.classifier.classify(self.image, self.detections)

        # ---- Step 4: Team selection ----
        self.state = S_SELECT_TEAM
        self._info_text = "Press  1  or  2  to pick your team   |   A = show all"
        self._team_select_loop()

        # ---- Step 5: Edit ----
        self.state = S_EDITING
        cv2.setMouseCallback(WINDOW, self._on_click_edit)
        self._edit_loop()

        cv2.destroyAllWindows()

    # ================================================================
    # VIDEO MODE
    # ================================================================
    def _run_video(self):
        cv2.namedWindow(WINDOW)

        # ---- Step 1: Calibrate on first frame ----
        print("\n=== Calibrating pitch on first frame ===")
        if not self.calibrator.calibrate(self.image, window=WINDOW):
            print("Calibration cancelled.")
            self.cap.release()
            cv2.destroyAllWindows()
            return
        self.polygon = self.calibrator.polygon
        self.goals = self.calibrator.goals
        cv2.destroyWindow(WINDOW)

        # ---- Step 2: Team selection (once, before processing) ----
        self._info_text = "Press  1  or  2  for your team   |   A = all players"
        self._show_team_selection_overlay()
        cv2.namedWindow(WINDOW)
        self._select_team_window()
        cv2.destroyWindow(WINDOW)

        # ---- Step 3: Process video ----
        self._process_video()

    def _show_team_selection_overlay(self):
        """Show the first frame with team selection prompt."""
        overlay = self.image.copy()
        self._draw_pitch(overlay)
        self._draw_goals(overlay)
        h, w = overlay.shape[:2]
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(bar, self._info_text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        overlay[h - 36:h] = bar
        self._overlay = overlay

    def _select_team_window(self):
        cv2.namedWindow(WINDOW)
        cv2.imshow(WINDOW, self._overlay)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('1'):
                self.my_team = 0
                self.classifier.set_my_team(0)
                print(f"  My team → Team 1")
                break
            elif key == ord('2'):
                self.my_team = 1
                self.classifier.set_my_team(1)
                print(f"  My team → Team 2")
                break
            elif key == ord('a'):
                self.my_team = None
                print("  Showing all players")
                break
            elif key in (ord('q'), 27):
                self.cap.release()
                sys.exit(0)

    def _process_video(self):
        """Process all frames, write annotated output."""
        # Output video — same dir as input
        if self.output_path is None:
            inp = Path(self.input_path)
            self.output_path = str(inp.parent / f"{inp.stem}_annotated.mp4")

        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, codec, self.vid_fps,
                                 (self.vid_w, self.vid_h))

        self.ball_detector = BallDetector()

        # Player tracking across frames
        all_players = {}        # track_id -> {name, team, first_frame, last_frame}
        save_data = {"my_team": f"Team {self.my_team + 1}" if self.my_team is not None else "All"}

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        processed = 0
        t_start = time.time()

        print("\nProcessing video…  (press Ctrl+C in terminal to stop early)")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_idx % self.skip_frames != 0:
                    frame_idx += 1
                    continue

                # Detect (downscale 4K → 1280px for better small-object recall)
                _, inside = self.detector.detect_and_filter_by_polygon(
                    frame, self.polygon, threshold=self.threshold,
                    resize_long_side=self.resize_long_side,
                )
                ball_xy = self.ball_detector.detect(frame, polygon=self.polygon)

                # Annotate
                annotated = frame.copy()
                self._draw_pitch(annotated)
                self._draw_goals(annotated)

                if inside is not None and len(inside) > 0:
                    # Track
                    if self.tracker is not None:
                        try:
                            tracked = self.tracker.update_with_detections(inside)
                            dets = tracked
                        except Exception as e:
                            print(f"    [!] Tracker error: {e}")
                            dets = inside
                    else:
                        dets = inside

                    # Classify
                    self.classifier.classify(frame, dets)

                    # Draw
                    for i in range(len(dets)):
                        x1, y1, x2, y2 = map(int, dets.xyxy[i])
                        conf = dets.confidence[i]

                        if self.tracker is not None and hasattr(dets, 'tracker_id') and dets.tracker_id is not None:
                            tid = int(dets.tracker_id[i])
                            if tid not in all_players:
                                all_players[tid] = {
                                    "id": tid,
                                    "name": "",
                                    "team": self.classifier.get_team_name(i),
                                    "first_frame": frame_idx,
                                    "last_frame": frame_idx,
                                }
                            all_players[tid]["last_frame"] = frame_idx
                            label = f"#{tid}"
                        else:
                            # Debug tracking failure
                            if not hasattr(dets, 'tracker_id'):
                                print(f"    [!] TRACK: no tracker_id attr, type={type(dets).__name__}", flush=True)
                            elif dets.tracker_id is None:
                                print(f"    [!] TRACK: tracker_id is None, dets len={len(dets)}", flush=True)
                            label = ""

                        # Colour
                        if self.team_labels and i in self.team_labels:
                            colour = self.classifier.get_colour(i)
                        else:
                            colour = (200, 200, 200)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                        # Conf + track ID label
                        label = f"{label} {conf:.2f}".strip()
                        cv2.putText(annotated, label, (x1, max(y1 - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

                # Ball
                if ball_xy is not None:
                    cx, cy, conf = ball_xy
                    cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 200, 255), -1)
                    cv2.putText(annotated, f"ball {conf:.2f}",
                                (int(cx) + 10, int(cy) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
                    for tx, ty in self.ball_detector.trail:
                        cv2.circle(annotated, (int(tx), int(ty)), 3,
                                   (0, 200, 255), -1)

                # Overlay frame counter
                cv2.putText(annotated, f"Frame {frame_idx}/{self.vid_total}",
                            (12, self.vid_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                writer.write(annotated)
                processed += 1

                # Progress
                if frame_idx % max(self.vid_total // 20, 30) == 0:
                    elapsed = time.time() - t_start
                    pct = frame_idx / self.vid_total * 100
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  {frame_idx}/{self.vid_total} ({pct:.0f}%)  "
                          f"─  {rate:.1f} fps")

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n  Stopped early.")

        # Cleanup
        writer.release()
        self.cap.release()
        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0

        print(f"\nDone: {processed} frames processed in {elapsed:.0f}s ({rate:.1f} fps)")
        print(f"Annotated video → {self.output_path}")

        # Save results
        save_data["video"] = self.input_path
        save_data["output"] = self.output_path
        save_data["calibration"] = {
            "pitch_polygon": self.polygon.tolist() if self.polygon is not None else [],
            "goals": [list(g) for g in self.goals],
        }
        save_data["players"] = list(all_players.values())

        json_path = Path(self.output_path).with_suffix(".json")
        print(f"DEBUG: all_players has {len(all_players)} entries, saving to {json_path}")
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Player data → {json_path}")

    # ================================================================
    # Shared helpers
    # ================================================================
    def _detect_and_filter(self, image):
        all_ppl, inside_ppl = self.detector.detect_and_filter_by_polygon(
            image, self.polygon, threshold=self.threshold,
            resize_long_side=self.resize_long_side,
        )
        self.all_detections = all_ppl
        self.detections = inside_ppl

    # ----------------------------------------------------------------
    # Team selection loop (image mode)
    # ----------------------------------------------------------------
    def _team_select_loop(self):
        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF

            if key == ord('1'):
                self.my_team = 0
                self.classifier.set_my_team(0)
                break
            elif key == ord('2'):
                self.my_team = 1
                self.classifier.set_my_team(1)
                break
            elif key == ord('a'):
                self.my_team = None
                self._info_text = "Showing all players"
                break
            elif key in (ord('q'), 27):
                cv2.destroyAllWindows()
                sys.exit(0)

    # ----------------------------------------------------------------
    # Edit loop (image mode)
    # ----------------------------------------------------------------
    def _edit_loop(self):
        self._info_text = "Click a player  |  D=delete  N=name  S=save  R=reset  Q=quit"
        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF

            if key == ord('d') and self.selected_idx >= 0:
                self.deleted.add(self.selected_idx)
                self.selected_idx = -1
                self._info_text = f"Player deleted  ({len(self.deleted)} removed)"

            elif key == ord('n') and self.selected_idx >= 0:
                print(f"\n  Name for player #{self.selected_idx}: ", end="", flush=True)
                name = sys.stdin.readline().strip()
                if name:
                    self.player_names[self.selected_idx] = name
                    self._info_text = f"Named → {name}"
                else:
                    self._info_text = "Name cancelled"

            elif key == ord('r'):
                self.deleted.clear()
                self.player_names.clear()
                self._info_text = "All edits reset"

            elif key == ord('s'):
                self._save_results()
                self._info_text = "Saved!"

            elif key in (ord('q'), 27):
                if self.selected_idx >= 0:
                    self.selected_idx = -1
                else:
                    break

    # ----------------------------------------------------------------
    # Mouse
    # ----------------------------------------------------------------
    def _on_click_edit(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.detections is None:
            return
        for i in range(len(self.detections)):
            if i in self.deleted:
                continue
            x1, y1, x2, y2 = map(int, self.detections.xyxy[i])
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_idx = i
                name = self.player_names.get(i, "")
                tag = f"#{i}" + (f" — {name}" if name else "")
                self._info_text = f"Selected {tag}  |  D=delete  N=name"
                return
        self.selected_idx = -1
        self._info_text = "Click a player  |  D=delete  N=name  S=save  Q=quit"

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------
    def _render(self):
        base = self.image.copy()
        self._draw_pitch(base)
        self._draw_goals(base)
        self._draw_detections(base)
        self._draw_status_bar(base)
        self._overlay = base
        cv2.imshow(WINDOW, self._overlay)

    def _draw_pitch(self, img):
        if self.polygon is not None:
            pts = self.polygon.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, COL_PITCH, 2)

    def _draw_goals(self, img):
        for g in self.goals:
            x, y = int(g[0]), int(g[1])
            cv2.circle(img, (x, y), 8, COL_GOAL, 2)
            cv2.putText(img, "GOAL", (x + 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GOAL, 1)

    def _draw_detections(self, img):
        if self.detections is None:
            return
        for i in range(len(self.detections)):
            if i in self.deleted:
                continue
            x1, y1, x2, y2 = map(int, self.detections.xyxy[i])
            conf = self.detections.confidence[i]

            if i == self.selected_idx:
                colour, thick = COL_SELECT, 3
            elif self.team_labels and i in self.team_labels:
                colour = self.classifier.get_colour(i)
                thick = 2
            else:
                colour, thick = (200, 200, 200), 1

            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)

            label_parts = []
            if i in self.player_names:
                label_parts.append(self.player_names[i])
            elif self.team_labels and i in self.team_labels:
                label_parts.append(self.classifier.get_team_name(i))
            label_parts.append(f"{conf:.2f}")

            cv2.putText(img, " | ".join(label_parts),
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    def _draw_status_bar(self, img):
        h, w = img.shape[:2]
        bar = np.zeros((36, w, 3), dtype=np.uint8)
        cv2.putText(bar, self._info_text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        img[h - 36:h] = bar

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    def _save_results(self):
        if self.output_path is None:
            inp = Path(self.input_path)
            self.output_path = str(inp.parent / f"{inp.stem}_goalhub.json")

        players = []
        for i in range(len(self.detections) if self.detections is not None else 0):
            if i in self.deleted:
                continue
            x1, y1, x2, y2 = self.detections.xyxy[i].tolist()
            entry = {
                "id": i,
                "name": self.player_names.get(i, ""),
                "team": (self.classifier.get_team_name(i)
                         if self.team_labels and i in self.team_labels
                         else "Unknown"),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(self.detections.confidence[i]),
            }
            players.append(entry)

        data = {
            "image": self.input_path,
            "calibration": {
                "pitch_polygon": self.polygon.tolist() if self.polygon is not None else [],
                "goals": [list(g) for g in self.goals],
            },
            "my_team": f"Team {self.my_team + 1}" if self.my_team is not None else "All",
            "players": players,
        }

        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(players)} players → {self.output_path}")

    # ----------------------------------------------------------------
    # Idle (no detections)
    # ----------------------------------------------------------------
    def _idle(self):
        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break


# ===================================================================
# CLI
# ===================================================================
def main():
    ap = argparse.ArgumentParser(
        description="GoalHub Detection — football player detection & pitch calibration"
    )
    ap.add_argument("input", help="Path to image or video file")
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "large"],
                    help="rf-detr model size (default: medium)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Detection confidence threshold (default: 0.5)")
    ap.add_argument("--resize", type=int, default=1280,
                    help="Downscale so longest edge = N px before detection (default: 1280, "
                         "0 = no resize)")
    ap.add_argument("--output", help="Output path (JSON for images, video for videos)")
    ap.add_argument("--skip", type=int, default=1,
                    help="Process every Nth frame in video (default: 1 = all)")
    args = ap.parse_args()

    try:
        resize = args.resize if args.resize > 0 else None
        app = GoalHubApp(
            args.input,
            model_size=args.model,
            threshold=args.threshold,
            output=args.output,
            skip_frames=args.skip,
            resize_long_side=resize,
        )
        app.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
