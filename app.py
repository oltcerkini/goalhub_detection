#!/usr/bin/env python3
"""GoalHub Web App — upload, calibrate, detect, and view football stats."""

import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
APP_DATA = BASE_DIR / "app_data"
UPLOAD_DIR = APP_DATA / "uploads"
OUTPUT_DIR = APP_DATA / "output"
CALIB_DIR = APP_DATA / "calibrations"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR, CALIB_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="GoalHub", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static (JS, CSS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory task store
tasks: dict = {}  # task_id -> {status, video_name, results_path, ...}

# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_frame(video_path: str, frame_idx: int = 0):
    """Grab a frame from the video as a numpy array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _frame_as_jpg_bytes(frame) -> bytes:
    """Encode a numpy frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Template not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Upload ─────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video, save it, extract the first frame for calibration."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    video_name = f"{video_id}{ext}"
    video_path = UPLOAD_DIR / video_name

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    # Extract first frame
    frame = _extract_frame(str(video_path), 30)
    if frame is None:
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read video file")

    frame_bytes = _frame_as_jpg_bytes(frame)
    frame_jpg_name = video_name + "_frame.jpg"
    frame_jpg_path = UPLOAD_DIR / frame_jpg_name
    with open(frame_jpg_path, "wb") as f:
        f.write(frame_bytes)

    return {
        "video_id": video_id,
        "video_name": video_name,
        "frame_url": f"/api/media/{frame_jpg_name}",
        "width": frame.shape[1],
        "height": frame.shape[0],
    }


# ── Calibration ────────────────────────────────────────────────────────────

@app.post("/api/calibrate")
async def save_calibration(data: dict):
    """Save calibration JSON (pitch polygon + goals + team preference)."""
    video_id = data.get("video_id")
    if not video_id:
        raise HTTPException(400, "Missing video_id")

    polygon = data.get("pitch_polygon")
    goals = data.get("goals", [])
    my_team = data.get("my_team", "All")

    if not polygon or len(polygon) < 4:
        raise HTTPException(400, "Need at least 4 pitch-corner points")

    cal = {
        "pitch_polygon": polygon,
        "goals": goals,
        "my_team": my_team,
    }
    cal_path = CALIB_DIR / f"{video_id}.json"
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)

    return {"status": "ok", "calibration_path": str(cal_path)}


# ── Process ────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def start_processing(data: dict):
    """Start processing a video in the background."""
    video_id = data.get("video_id")
    video_name = data.get("video_name")
    threshold = data.get("threshold", 0.2)
    skip = data.get("skip", 1)
    model = data.get("model", None)
    team_tracks = data.get("team_tracks", "")

    if not video_id or not video_name:
        raise HTTPException(400, "Missing video_id or video_name")

    cal_path = CALIB_DIR / f"{video_id}.json"
    if not cal_path.exists():
        raise HTTPException(400, "No calibration found for this video. Calibrate first.")

    video_path = UPLOAD_DIR / video_name
    if not video_path.exists():
        raise HTTPException(400, "Video file not found")

    task_id = str(uuid.uuid4())[:8]
    video_stem = Path(video_name).stem
    output_path = OUTPUT_DIR / f"{video_stem}_processed.json"

    tasks[task_id] = {
        "status": "queued",
        "video_id": video_id,
        "video_name": video_name,
        "results_path": str(output_path),
        "started_at": None,
    }

    # Launch processing in background
    _launch_processing(task_id, str(video_path), str(cal_path), str(output_path),
                       threshold, skip, model, team_tracks)

    return {"task_id": task_id}


def _launch_processing(task_id, video_path, cal_path, output_path,
                       threshold, skip, model, team_tracks):
    """Run process.py as a subprocess with the given arguments."""
    cmd = [
        sys.executable, str(BASE_DIR / "process.py"),
        video_path,
        "--calibration", cal_path,
        "--threshold", str(threshold),
        "--skip", str(skip),
        "--gamma", "0.85",
        "--output-dir", str(OUTPUT_DIR),
    ]
    if model:
        cmd.extend(["--model", model])
    if team_tracks:
        cmd.extend(["--team-tracks", team_tracks])

    tasks[task_id]["status"] = "processing"
    tasks[task_id]["started_at"] = time.time()

    def _run():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            tasks[task_id]["stdout"] = result.stdout
            tasks[task_id]["stderr"] = result.stderr

            if result.returncode != 0:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = result.stderr[-2000:] if result.stderr else "Unknown error"
                return

            if not Path(output_path).exists():
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = "Output JSON not found at expected path"
                return

            tasks[task_id]["status"] = "completed"

        except subprocess.TimeoutExpired:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = "Processing timed out (>10 minutes)"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    import threading
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


# ── Status & Results ───────────────────────────────────────────────────────

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Poll processing status."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    resp = {
        "status": task["status"],
        "stdout": task.get("stdout", ""),
        "stderr": task.get("stderr", ""),
    }
    if task["status"] == "error":
        resp["error"] = task.get("error", "")
    if task["status"] == "completed":
        resp["results_url"] = f"/api/results/{task_id}"
        resp["video_url"] = _find_output_video(task)

    return resp


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Return the full results JSON."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed yet")

    results_path = task.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(404, "Results file not found")

    with open(results_path) as f:
        data = json.load(f)

    # Load player names if available
    names_path = Path(results_path).with_suffix(".names.json")
    if names_path.exists():
        with open(names_path) as f:
            data["player_names"] = json.load(f)
    else:
        data["player_names"] = {}

    # Convert heatmap path to URL
    hmap = data.get("heatmap")
    if hmap and Path(hmap).exists():
        data["heatmap_url"] = f"/api/media/{Path(hmap).name}"
    else:
        data["heatmap_url"] = None

    return data


@app.put("/api/player-name")
async def update_player_name(data: dict):
    """Update a player's name mapping."""
    task_id = data.get("task_id")
    track_id = data.get("track_id")
    name = data.get("name", "")

    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    results_path = task.get("results_path")
    if not results_path:
        raise HTTPException(400, "No results path")

    names_path = Path(results_path).with_suffix(".names.json")
    names = {}
    if names_path.exists():
        with open(names_path) as f:
            names = json.load(f)

    if name.strip():
        names[str(track_id)] = name.strip()
    else:
        names.pop(str(track_id), None)

    with open(names_path, "w") as f:
        json.dump(names, f, indent=2)

    return {"status": "ok"}


# ── Media Serving ──────────────────────────────────────────────────────────

@app.get("/api/media/{filename}")
async def serve_media(filename: str):
    """Serve uploaded / output files."""
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        path = d / filename
        if path.exists():
            return FileResponse(str(path), media_type=_media_type(filename))
    raise HTTPException(404, "File not found")


def _media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")


def _find_output_video(task) -> str | None:
    """Find the processed MP4 from process.py output."""
    video_name = task.get("video_name", "")
    if not video_name:
        return None
    stem = Path(video_name).stem
    candidates = [
        OUTPUT_DIR / f"{stem}_processed.mp4",
        OUTPUT_DIR / f"{video_name}",
    ]
    for c in candidates:
        if c.exists():
            return f"/api/media/{c.name}"
    return None


# ── Re-render (team filter) ─────────────────────────────────────────────────

@app.post("/api/re-render")
async def re_render(data: dict):
    """Re-render the processed video filtered to one team."""
    task_id = data.get("task_id")
    team = data.get("team")  # "My Team" or "Team 2"
    attacking_goal = data.get("attacking_goal")  # "left" or "right" (optional)

    if not task_id or not team:
        raise HTTPException(400, "Missing task_id or team")

    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed yet")

    results_path = task.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(404, "Results file not found")

    # Build output path (attacking_goal in filename to avoid cache collisions)
    video_name = task.get("video_name", "")
    stem = Path(video_name).stem
    team_slug = team.lower().replace(" ", "_")
    attack_slug = f"_{attacking_goal}" if attacking_goal else ""
    filtered_path = OUTPUT_DIR / f"{stem}_filtered_{team_slug}{attack_slug}.mp4"

    # Skip if already rendered (but only if we have no active exclusions, otherwise
    # re-render to ensure the excluded tracks are also filtered out)
    excluded = task.get("excluded_tracks", [])
    if filtered_path.exists() and not excluded:
        return {"video_url": f"/api/media/{filtered_path.name}", "team": team}

    # Run render_filtered.py
    cmd = [
        sys.executable, str(BASE_DIR / "render_filtered.py"),
        "--results", results_path,
        "--team", team,
        "--output-dir", str(OUTPUT_DIR),
    ]
    if attacking_goal:
        cmd.extend(["--attacking-goal", attacking_goal])

    # Also pass any previously excluded tracks from clean-render
    if excluded:
        cmd.extend(["--exclude-tracks", ",".join(str(t) for t in excluded)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(500, f"Re-render failed: {result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Re-render timed out (>10 minutes)")

    if not filtered_path.exists():
        raise HTTPException(500, "Re-rendered video not found at expected path")

    return {"video_url": f"/api/media/{filtered_path.name}", "team": team}


# ── Review frame ────────────────────────────────────────────────────────────

@app.get("/api/review-frame/{task_id}")
async def get_review_frame(task_id: str):
    """Generate a review frame showing all detected player tracks.

    Returns the frame as a JPEG URL and a JSON array of detections
    (track_id, bbox, team, confidence) so the frontend can render
    clickable overlays.
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed yet")

    results_path = task.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(404, "Results file not found")

    with open(results_path) as f:
        data = json.load(f)

    video_path = data.get("video")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, "Source video not found in results")

    # Find the frame with the most unique track_ids (best review frame)
    frame_tracks = defaultdict(set)
    for det in data.get("detections", []):
        tid = det.get("track_id", -1)
        if tid > 0:
            frame_tracks[det["frame"]].add(tid)

    if not frame_tracks:
        raise HTTPException(400, "No player detections found in results")

    best_frame = max(frame_tracks, key=lambda f: len(frame_tracks[f]))

    # Extract frame from video
    frame = _extract_frame(video_path, best_frame)
    if frame is None:
        raise HTTPException(500, "Failed to extract frame from video")

    # Get player detections for this frame only
    players_in_frame = [
        det for det in data.get("detections", [])
        if det.get("frame") == best_frame and det.get("track_id", -1) > 0
    ]

    # Build detection data for the frontend
    detections_json = []
    for p in players_in_frame:
        detections_json.append({
            "track_id": p["track_id"],
            "bbox": p["bbox"],
            "team": p.get("team", "Unknown"),
            "confidence": p.get("confidence", 0),
        })

    # Save the review frame as JPEG
    frame_jpg_name = f"review_{task_id}.jpg"
    frame_jpg_path = OUTPUT_DIR / frame_jpg_name
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(frame_jpg_path, "wb") as f:
        f.write(buf.tobytes())

    # Count unique track_ids across the whole video
    all_track_ids = set()
    for det in data.get("detections", []):
        tid = det.get("track_id", -1)
        if tid > 0:
            all_track_ids.add(tid)

    return {
        "frame_url": f"/api/media/{frame_jpg_name}",
        "detections": detections_json,
        "frame_idx": best_frame,
        "img_width": frame.shape[1],
        "img_height": frame.shape[0],
        "total_player_tracks": len(all_track_ids),
    }


# ── Clean render (exclude tracks) ───────────────────────────────────────────

@app.post("/api/clean-render")
async def clean_render(data: dict):
    """Re-render the video excluding specified player track IDs.

    Runs a lightweight re-render from the existing results JSON — no
    re-detection, just skipping the excluded tracks during rendering.
    Stores the exclusion list in task state so subsequent team-filter
    re-renders also respect the exclusions.
    """
    task_id = data.get("task_id")
    exclude_tracks = data.get("exclude_tracks", [])

    if not task_id:
        raise HTTPException(400, "Missing task_id")
    if not exclude_tracks:
        raise HTTPException(400, "No tracks to exclude")

    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed yet")

    results_path = task.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(404, "Results file not found")

    video_name = task.get("video_name", "")
    stem = Path(video_name).stem
    clean_path = OUTPUT_DIR / f"{stem}_clean.mp4"

    # Skip if already rendered
    if clean_path.exists():
        return {"video_url": f"/api/media/{clean_path.name}", "excluded_count": len(exclude_tracks)}

    # Run render_filtered.py with --exclude-tracks
    cmd = [
        sys.executable, str(BASE_DIR / "render_filtered.py"),
        "--results", results_path,
        "--exclude-tracks", ",".join(str(t) for t in exclude_tracks),
        "--output-dir", str(OUTPUT_DIR),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(500, f"Clean render failed: {result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Clean render timed out (>10 minutes)")

    # Find output from the script's OUTPUT_PATH: marker
    output_path = None
    for line in result.stdout.split("\n"):
        if line.startswith("OUTPUT_PATH:"):
            output_path = line.split(":", 1)[1].strip()
            break

    if not output_path or not Path(output_path).exists():
        raise HTTPException(500, "Clean rendered video not found at expected path")

    # Store excluded tracks in task state
    tasks[task_id]["excluded_tracks"] = list(exclude_tracks)

    return {
        "video_url": f"/api/media/{Path(output_path).name}",
        "excluded_count": len(exclude_tracks),
    }


# ── List tasks ─────────────────────────────────────────────────────────────

@app.get("/api/tasks")
async def list_tasks():
    """Return all known tasks and their status."""
    return {
        tid: {"status": t["status"], "video": t.get("video_name", "")}
        for tid, t in tasks.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
