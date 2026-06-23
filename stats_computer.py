"""Football stats computer — distance, goals, passes, assists.

Usage:
    from stats_computer import PitchMapper, StatsComputer

    mapper = PitchMapper(pitch_polygon, goals)
    computer = StatsComputer(mapper)
    stats = computer.compute(ball_trail, player_detections, total_frames, fps, frame_skip)
"""

from collections import defaultdict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pitch → world mapping via homography
# ---------------------------------------------------------------------------

class PitchMapper:
    """Maps pixel coordinates to real-world meters using homography."""

    _PITCH_WIDTH = 105.0   # meters (standard)
    _PITCH_HEIGHT = 68.0

    def __init__(self, pitch_polygon):
        """pitch_polygon: 4-point list [(x,y), ...] from calibration."""
        pixel_pts = np.array(pitch_polygon, dtype=np.float32)

        # Reorder to [tl, tr, br, bl] by sorting y then x
        sorted_pts = sorted(pixel_pts, key=lambda p: (p[1], p[0]))
        top = sorted(sorted_pts[:2], key=lambda p: p[0])   # tl, tr
        bot = sorted(sorted_pts[2:], key=lambda p: p[0])   # bl, br
        src = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

        dst = np.array([
            [0, 0],
            [self._PITCH_WIDTH, 0],
            [self._PITCH_WIDTH, self._PITCH_HEIGHT],
            [0, self._PITCH_HEIGHT],
        ], dtype=np.float32)

        self._H = cv2.getPerspectiveTransform(src, dst)

    def pixel_to_world(self, x, y):
        """Return (meters_x, meters_y)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._H)
        return world[0, 0]

    def distance_m(self, x1, y1, x2, y2):
        """Euclidean distance in metres between two pixel points."""
        wx1, wy1 = self.pixel_to_world(x1, y1)
        wx2, wy2 = self.pixel_to_world(x2, y2)
        return float(np.sqrt((wx1 - wx2) ** 2 + (wy1 - wy2) ** 2))


# ---------------------------------------------------------------------------
# Stats computer
# ---------------------------------------------------------------------------

class StatsComputer:
    """Computes per-track distance, goals, passes, assists from detections."""

    def __init__(self, pitch_mapper, goal_threshold_px=45, pass_ball_min_m=5.0,
                 max_pass_distance_m=60.0, max_ball_speed_m_per_s=40.0,
                 min_pixel_movement=5, min_distance_m=0.3,
                 assist_window_frames=15):
        self.mapper = pitch_mapper
        self.goal_threshold_px = goal_threshold_px
        self.pass_ball_min_m = pass_ball_min_m
        self.max_pass_distance_m = max_pass_distance_m
        self.max_ball_speed_m_per_s = max_ball_speed_m_per_s
        self.min_px_move = min_pixel_movement  # squared threshold below
        self.min_dist_m = min_distance_m
        self.assist_window = assist_window_frames

    # ------------------------------------------------------------------
    # Covered distance
    # ------------------------------------------------------------------

    def per_track_distance(self, player_detections):
        """Return {track_id: total_distance_meters}.

        player_detections: list of dicts with track_id, frame, bbox.
        """
        # Group by track, sorted by frame
        tracks = defaultdict(list)
        for d in player_detections:
            tid = d["track_id"]
            bbox = d["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            tracks[tid].append((d["frame"], cx, cy))

        distances = {}
        min_px2 = self.min_px_move ** 2
        for tid, positions in tracks.items():
            positions.sort(key=lambda x: x[0])  # sort by frame
            total = 0.0
            for i in range(1, len(positions)):
                _, x1, y1 = positions[i - 1]
                _, x2, y2 = positions[i]
                dx, dy = x2 - x1, y2 - y1
                if dx * dx + dy * dy >= min_px2:
                    d = self.mapper.distance_m(x1, y1, x2, y2)
                    if d >= self.min_dist_m:
                        total += d
            if total > 0:
                distances[tid] = round(total, 1)

        return distances

    # ------------------------------------------------------------------
    # Goal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _dist_to_segment(px, py, x1, y1, x2, y2):
        """Perpendicular distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
        sx, sy = x2 - x1, y2 - y1
        dx, dy = px - x1, py - y1
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq == 0:
            return np.sqrt(dx * dx + dy * dy)
        t = max(0.0, min(1.0, (dx * sx + dy * sy) / seg_len_sq))
        proj_x = x1 + t * sx
        proj_y = y1 + t * sy
        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    @staticmethod
    def _is_crossing_segment(prev_x, prev_y, cur_x, cur_y, x1, y1, x2, y2):
        """Check if the line from (prev_x,prev_y) to (cur_x,cur_y) crosses
        the segment (x1,y1)-(x2,y2). Uses orientation-based intersection test."""
        def orient(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        o1 = orient(x1, y1, x2, y2, prev_x, prev_y)
        o2 = orient(x1, y1, x2, y2, cur_x, cur_y)
        o3 = orient(prev_x, prev_y, cur_x, cur_y, x1, y1)
        o4 = orient(prev_x, prev_y, cur_x, cur_y, x2, y2)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        # Collinear cases — ball is exactly on the line
        if o1 == 0 and min(x1, x2) <= prev_x <= max(x1, x2) and min(y1, y2) <= prev_y <= max(y1, y2):
            return True
        if o2 == 0 and min(x1, x2) <= cur_x <= max(x1, x2) and min(y1, y2) <= cur_y <= max(y1, y2):
            return True
        return False

    def detect_goals(self, ball_trail, goals_px):
        """Detect when ball crosses a goal line between two posts.

        ball_trail: list of (x, y, frame, conf)
        goals_px: list of 8 (gx, gy) — left goal 4pts (BL, BR, TL, TR), right goal 4pts (BL, BR, TL, TR)

        Returns list of dicts: [{frame, goal_index, goal_x, goal_y}, …]
        """
        if not ball_trail or not goals_px or len(goals_px) < 8:
            return []

        # Build goal segments from top bar (TL→TR) of each 4-point goal
        segments = []
        for g_idx in range(2):
            i = g_idx * 4 + 2  # TL index within each goal's 4 points
            segments.append((goals_px[i][0], goals_px[i][1],
                             goals_px[i + 1][0], goals_px[i + 1][1]))

        goals_found = []
        prev_near = [False] * 2  # one per goal
        just_scored = [False] * 2  # cooldown per goal
        recent = []

        for x, y, frame, _conf in ball_trail:
            recent.append((x, y))
            if len(recent) > 5:
                recent.pop(0)

            for g_idx, (x1, y1, x2, y2) in enumerate(segments):
                dist = self._dist_to_segment(x, y, x1, y1, x2, y2)
                near = dist < self.goal_threshold_px

                if near and not prev_near[g_idx] and not just_scored[g_idx]:
                    # Ball just arrived near goal line — check direction
                    # Ball should be moving toward the goal, not away
                    if len(recent) >= 3:
                        bx, by = recent[-3]
                        # Vector from recent past to current
                        mv_dx, mv_dy = x - bx, y - by
                        # Vector from past to goal midpoint
                        gx_mid = (x1 + x2) / 2
                        gy_mid = (y1 + y2) / 2
                        to_goal_dx = gx_mid - bx
                        to_goal_dy = gy_mid - by
                        dot = mv_dx * to_goal_dx + mv_dy * to_goal_dy
                        if dot < 0:
                            # Moving away from goal — likely false positive
                            prev_near[g_idx] = near
                            continue

                    goals_found.append({
                        "frame": frame,
                        "goal_index": g_idx,
                        "goal_x": float((x1 + x2) / 2),
                        "goal_y": float((y1 + y2) / 2),
                        "confidence": round(float(_conf), 3),
                    })
                    just_scored[g_idx] = True
                elif not near:
                    just_scored[g_idx] = False

                prev_near[g_idx] = near

        return goals_found

    # ------------------------------------------------------------------
    # Pass detection
    # ------------------------------------------------------------------

    def detect_passes(self, ball_trail, player_detections):
        """Detect passes from nearest-player-to-ball changes.

        ball_trail: list of (x, y, frame, conf)
        player_detections: list of dicts with track_id, frame, bbox

        Returns list of dicts:
            [{frame, passer_track_id, receiver_track_id,
              ball_distance_m, ball_speed_m_per_s,
              passer_x, passer_y, receiver_x, receiver_y}, …]
        """
        if not ball_trail or not player_detections:
            return []

        # Group players by frame with bbox centres
        players_by_frame = defaultdict(list)
        for d in player_detections:
            tid = d["track_id"]
            if tid <= 0:
                continue
            bbox = d["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            players_by_frame[d["frame"]].append({
                "tid": tid,
                "cx": cx,
                "cy": cy,
            })

        # Build ball frame lookup
        ball_by_frame = {f: (x, y, c) for x, y, f, c in ball_trail}

        # Find nearest player to ball per frame
        frame_closest = {}  # frame -> {tid, cx, cy, dist_px}
        for bx, by, frame, _conf in ball_trail:
            players = players_by_frame.get(frame, [])
            if not players:
                continue
            nearest = min(players, key=lambda p: (p["cx"] - bx) ** 2 + (p["cy"] - by) ** 2)
            dist_px = np.sqrt((nearest["cx"] - bx) ** 2 + (nearest["cy"] - by) ** 2)
            frame_closest[frame] = {
                "tid": nearest["tid"],
                "cx": nearest["cx"],
                "cy": nearest["cy"],
                "dist_px": dist_px,
            }

        passes = []
        prev_tid = None
        prev_frame = None
        prev_bx = None
        prev_by = None

        for frame in sorted(frame_closest.keys()):
            cur = frame_closest[frame]

            if prev_tid is not None and cur["tid"] != prev_tid:
                # Skip if too many frames between — likely a detection gap
                if prev_frame is not None and frame - prev_frame > 3:
                    prev_tid = cur["tid"]
                    prev_frame = frame
                    ball = ball_by_frame.get(frame)
                    if ball:
                        prev_bx, prev_by = ball[0], ball[1]
                    continue

                # Nearest player changed — check ball moved enough
                cur_ball = ball_by_frame.get(frame)
                prev_ball = ball_by_frame.get(prev_frame)

                if cur_ball and prev_ball:
                    ball_dist = self.mapper.distance_m(
                        prev_ball[0], prev_ball[1], cur_ball[0], cur_ball[1],
                    )
                    if ball_dist >= self.pass_ball_min_m and ball_dist <= self.max_pass_distance_m:
                        passes.append({
                            "frame": frame,
                            "passer_track_id": prev_tid,
                            "receiver_track_id": cur["tid"],
                            "ball_distance_m": round(ball_dist, 1),
                            "passer_x": round(float(prev_bx), 1) if prev_bx else None,
                            "passer_y": round(float(prev_by), 1) if prev_by else None,
                            "receiver_x": round(float(cur["cx"]), 1),
                            "receiver_y": round(float(cur["cy"]), 1),
                        })

            prev_tid = cur["tid"]
            prev_frame = frame
            ball = ball_by_frame.get(frame)
            if ball:
                prev_bx, prev_by = ball[0], ball[1]

        return passes

    # ------------------------------------------------------------------
    # Possession stats
    # ------------------------------------------------------------------

    def compute_possession(self, ball_trail, player_detections, team_labels=None):
        """Compute ball possession per team.

        Args:
            ball_trail: list of (x, y, frame, conf)
            player_detections: list of dicts with track_id, frame, bbox
            team_labels: dict track_id -> team name ('My Team' / 'Team 2' / etc.)

        Returns:
            dict with frames_per_team, percentage_per_team, total_frames
        """
        if not ball_trail or not player_detections:
            return {"frames_per_team": {}, "percentage_per_team": {}, "total_frames": 0}

        # Group players by frame with positions
        players_by_frame = defaultdict(list)
        track_teams = {}
        for d in player_detections:
            tid = d["track_id"]
            if tid <= 0:
                continue
            bbox = d["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            players_by_frame[d["frame"]].append({"tid": tid, "cx": cx, "cy": cy})
            if team_labels:
                track_teams[tid] = team_labels.get(tid, "Unknown")

        # Per frame: nearest player to ball → their team
        team_frames = defaultdict(int)
        total_with_ball = 0

        for bx, by, frame, conf in ball_trail:
            if conf < 0.3:  # skip low-confidence ball detections
                continue
            players = players_by_frame.get(frame, [])
            if not players:
                continue

            nearest = min(players, key=lambda p: (p["cx"] - bx) ** 2 + (p["cy"] - by) ** 2)
            team = track_teams.get(nearest["tid"], "Unknown")

            team_frames[team] += 1
            total_with_ball += 1

        percentages = {}
        if total_with_ball > 0:
            for team, frames in team_frames.items():
                percentages[team] = round(frames / total_with_ball * 100, 1)

        return {
            "frames_per_team": dict(team_frames),
            "percentage_per_team": percentages,
            "total_frames": total_with_ball,
        }

    # ------------------------------------------------------------------
    # Assists
    # ------------------------------------------------------------------

    def compute_assists(self, passes, goals):
        """Link passes to goals that happen within assist_window_frames after.

        passes: list from detect_passes()
        goals:  list from detect_goals()

        Returns list of dicts:
            [{pass_event, goal_event, frames_between}, …]
        """
        assists = []
        for goal in goals:
            goal_frame = goal["frame"]
            # Find the most recent pass before this goal
            best = None
            for p in passes:
                gap = goal_frame - p["frame"]
                if 0 < gap <= self.assist_window:
                    if best is None or gap < (goal_frame - best["frame"]):
                        best = {**p, "frames_before_goal": gap}
            if best:
                assists.append({
                    "pass_frame": best["frame"],
                    "passer": best["passer_track_id"],
                    "receiver": best["receiver_track_id"],
                    "goal_frame": goal_frame,
                    "frames_between": best["frames_before_goal"],
                })

        return assists

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def compute(self, ball_trail, player_detections, goals_px=None,
                total_frames=None, fps=None, frame_skip=None,
                team_labels=None):
        """Convenience: run all stats and return a single dict.

        Args:
            team_labels: dict track_id -> team name for possession tracking.
        """
        distances = self.per_track_distance(player_detections)
        goals = self.detect_goals(ball_trail, goals_px or [])
        passes = self.detect_passes(ball_trail, player_detections)
        assists = self.compute_assists(passes, goals)
        possession = self.compute_possession(ball_trail, player_detections, team_labels)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        return {
            "distances": {
                "per_track_meters": dict(sorted_distances),
                "total_distance_all_players_m": round(sum(distances.values()), 1),
            },
            "goals": goals,
            "passes": passes,
            "assists": assists,
            "possession": possession,
        }
