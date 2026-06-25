"""Post-processing for detection results.

Runs after all frames are processed:
  1. Merge fragmented tracks (same player lost and re-detected with new ID)
  2. Smooth team assignments (majority vote across all frames)
  3. Filter short/noise tracks
"""

import numpy as np
from collections import defaultdict


class PostProcessor:
    """Post-hoc fixes for detection and tracking."""

    def __init__(self, max_merge_gap=60, merge_proximity_px=200,
                 min_track_frames=3, min_track_conf=0.10):
        self.max_merge_gap = max_merge_gap
        self.merge_proximity_px = merge_proximity_px
        self.min_track_frames = min_track_frames
        self.min_track_conf = min_track_conf

    def process(self, all_players):
        """Run all post-processing steps. Modifies all_players in place.

        Args:
            all_players: dict of frame_key -> {track_id, frame, bbox, confidence, team, ...}

        Returns:
            dict of statistics about what was changed.
        """
        stats = {}

        # Step 1: Merge fragmented tracks
        n_merged = self._merge_fragmented_tracks(all_players)
        stats["tracks_merged"] = n_merged

        # Step 2: Smooth team assignments
        n_team_fixed = self._smooth_teams(all_players)
        stats["team_fixes"] = n_team_fixed

        # Step 3: Filter noise tracks
        n_removed = self._filter_noise(all_players)
        stats["noise_tracks_removed"] = n_removed

        return stats

    # ------------------------------------------------------------------
    # Track merging
    # ------------------------------------------------------------------

    def _build_track_summary(self, all_players):
        """Build per-track summary from detection data."""
        summary = {}
        for key, det in all_players.items():
            tid = det["track_id"]
            if tid <= 0:
                continue
            if tid not in summary:
                summary[tid] = {
                    "frame": det["frame"],
                    "bbox": det["bbox"],
                    "xs": [], "ys": [],
                    "first": det["frame"],
                    "last": det["frame"],
                }
            s = summary[tid]
            s["xs"].append((det["bbox"][0] + det["bbox"][2]) / 2)
            s["ys"].append((det["bbox"][1] + det["bbox"][3]) / 2)
            s["first"] = min(s["first"], det["frame"])
            s["last"] = max(s["last"], det["frame"])
        return summary

    def _merge_fragmented_tracks(self, all_players):
        """Merge tracks that don't overlap temporally and are spatially close."""
        summary = self._build_track_summary(all_players)
        tids = sorted(summary.keys(), key=lambda t: (summary[t]["first"], t))
        if len(tids) < 2:
            return 0

        # Union-find for tracking merge chains
        parent = {tid: tid for tid in tids}
        def find(t):
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t

        # Temporarily extended summaries for root tracks
        # (merged roots accumulate child data)
        ext = {tid: {"xs": list(summary[tid]["xs"]),
                      "ys": list(summary[tid]["ys"]),
                      "first": summary[tid]["first"],
                      "last": summary[tid]["last"]}
               for tid in tids}

        merge_count = 0
        for i, tid_a in enumerate(tids):
            ra = find(tid_a)
            sa = ext[ra]
            for j in range(i + 1, len(tids)):
                tid_b = tids[j]
                rb = find(tid_b)
                if ra == rb:
                    continue
                sb = ext[rb]

                # Must not overlap in time
                if not (sa["last"] < sb["first"] or sb["last"] < sa["first"]):
                    continue

                # Gap must be small
                gap = (sb["first"] - sa["last"]) if sb["first"] > sa["last"] else (sa["first"] - sb["last"])
                if gap < 0 or gap > self.max_merge_gap:
                    continue

                # Positions must be close
                avg_a = (np.mean(sa["xs"]), np.mean(sa["ys"]))
                avg_b = (np.mean(sb["xs"]), np.mean(sb["ys"]))
                dist = np.sqrt((avg_a[0] - avg_b[0]) ** 2 + (avg_a[1] - avg_b[1]) ** 2)
                if dist > self.merge_proximity_px:
                    continue

                # Merge rb into ra
                parent[rb] = ra
                sa["xs"].extend(sb["xs"])
                sa["ys"].extend(sb["ys"])
                sa["first"] = min(sa["first"], sb["first"])
                sa["last"] = max(sa["last"], sb["last"])
                merge_count += 1

        if merge_count == 0:
            return 0

        # Build final remapping
        remap = {}
        for tid in tids:
            root = find(tid)
            if root != tid:
                remap[tid] = root

        # Apply to all_players
        applied = set()
        for det in all_players.values():
            old = det["track_id"]
            if old in remap:
                det["track_id"] = remap[old]
                applied.add(old)

        n_unique_merged = len(applied)
        print(f"  PostProc: merged {n_unique_merged} fragmented tracks into {merge_count} groups")
        return n_unique_merged

    # ------------------------------------------------------------------
    # Team smoothing
    # ------------------------------------------------------------------

    def _smooth_teams(self, all_players):
        """Assign each track the majority team across all frames."""
        # Collect team votes per track
        team_votes = defaultdict(lambda: {"My Team": 0, "Team 2": 0, "Team 1": 0,
                                          "GK My Team": 0, "GK Team 2": 0, "GK Team 1": 0,
                                          "Referee": 0, "Unknown": 0})

        for det in all_players.values():
            tid = det["track_id"]
            team = det.get("team", "Unknown")
            if tid > 0 and team in team_votes[tid]:
                team_votes[tid][team] += 1

        # For each track, find the majority non-Unknown team
        fixes = 0
        for tid, votes in team_votes.items():
            # Skip if not enough data
            total = sum(votes.values())
            if total < 3:
                continue

            # Count outfield teams (non-GK) and GK variants
            outfield = {"My Team": votes["My Team"], "Team 1": votes["Team 1"], "Team 2": votes["Team 2"]}
            gk = {"GK My Team": votes["GK My Team"], "GK Team 1": votes["GK Team 1"], "GK Team 2": votes["GK Team 2"]}

            # Map GK variants to their base team
            gk_to_team = {"GK My Team": "My Team", "GK Team 1": "Team 1", "GK Team 2": "Team 2"}

            # Count total per base team (outfield + GK)
            team_totals = {}
            for team_name in ["My Team", "Team 1", "Team 2"]:
                team_totals[team_name] = outfield[team_name]
            for gk_name, base_name in gk_to_team.items():
                team_totals[base_name] = team_totals.get(base_name, 0) + gk[gk_name]

            # Also count referee
            team_totals["Referee"] = votes["Referee"]

            if not team_totals:
                continue

            # Winner = team with most votes (excluding Unknown)
            winner = max(team_totals, key=team_totals.get)
            winner_votes = team_totals[winner]
            total_meaningful = sum(team_totals.values())

            if total_meaningful == 0 or winner_votes / total_meaningful < 0.5:
                continue  # no clear majority

            # Fix all detections for this track
            for det in all_players.values():
                if det["track_id"] == tid and det.get("team", "") not in ("", "Unknown"):
                    old_team = det["team"]
                    # Map to base team if GK variant
                    if old_team in gk_to_team:
                        old_team = gk_to_team[old_team]
                    if old_team != winner:
                        # Preserve GK prefix if applicable
                        if "GK" in det.get("team", ""):
                            det["team"] = f"GK {winner}" if winner != "Referee" else "Referee"
                        else:
                            det["team"] = winner
                        fixes += 1

        if fixes:
            print(f"  PostProc: fixed {fixes} team assignments by majority vote")
        return fixes

    # ------------------------------------------------------------------
    # Noise filtering
    # ------------------------------------------------------------------

    def _filter_noise(self, all_players):
        """Remove very short, low-confidence tracks (likely false positives)."""
        # Count frames and avg confidence per track
        track_info = defaultdict(lambda: {"frames": 0, "conf_sum": 0.0, "frame_nums": set()})

        for det in all_players.values():
            tid = det["track_id"]
            if tid <= 0:
                continue
            track_info[tid]["frames"] += 1
            track_info[tid]["conf_sum"] += det.get("confidence", 0)
            track_info[tid]["frame_nums"].add(det["frame"])

        to_remove = set()
        for tid, info in track_info.items():
            avg_conf = info["conf_sum"] / info["frames"] if info["frames"] > 0 else 0
            unique_frames = len(info["frame_nums"])

            # Remove if very few unique frames OR very low avg confidence
            if unique_frames < self.min_track_frames:
                to_remove.add(tid)
            elif avg_conf < self.min_track_conf and unique_frames < 5:
                to_remove.add(tid)

        if not to_remove:
            return 0

        # Remove matching entries
        keys_to_delete = []
        for key, det in all_players.items():
            if det["track_id"] in to_remove:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del all_players[key]

        print(f"  PostProc: removed {len(to_remove)} noise tracks ({len(keys_to_delete)} detections)")
        return len(to_remove)
