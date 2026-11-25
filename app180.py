#!/usr/bin/env python3
"""GoalHub 180 — upload a VR180 video, trace/calibrate it, detect players.

Same shape as the drone app (app.py) but for equirectangular 180 footage:
    upload  ->  trace the pitch curves on a frame (calib/calibrate.html)
            ->  solve the camera pose (calib/calibrate.py)
            ->  detect players (vr180_pipeline.py)
Nothing here touches the flat/drone pipeline.

    python app180.py            # http://127.0.0.1:8081
"""
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from equirect import normalize_frame  # noqa: E402

DATA = BASE_DIR / "app_data180"
UPLOAD_DIR = DATA / "uploads"
OUTPUT_DIR = DATA / "output"
CALIB_DIR = DATA / "calibrations"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
TOOL_HTML = BASE_DIR / "calib" / "calibrate.html"

for d in (UPLOAD_DIR, OUTPUT_DIR, CALIB_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GoalHub 180", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

tasks: dict = {}


def _grab_normalized_frame(video_path: str, frame_idx: int):
    """Read a frame, crop to the equirect content region (the pipeline's canonical
    space) so the traced pixels line up with what vr180_pipeline.py sees."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None
    content, meta = normalize_frame(frame)
    return content, meta


@app.get("/", response_class=HTMLResponse)
async def index():
    p = TEMPLATES_DIR / "index180.html"
    if not p.exists():
        return HTMLResponse("<h1>index180.html not found</h1>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/calib/tool")
async def calib_tool():
    """The tracing tool, served same-origin so the page can embed it and read its
    trace (no file downloads, no CORS)."""
    return FileResponse(str(TOOL_HTML), media_type="text/html",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/media/{name}")
async def media(name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = UPLOAD_DIR / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), frame: int = Form(240)):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    video_name = f"{video_id}{ext}"
    video_path = UPLOAD_DIR / video_name
    with open(video_path, "wb") as f:            # stream: clips are multi-GB
        while True:
            chunk = await file.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    content, meta = _grab_normalized_frame(str(video_path), int(frame))
    if content is None:
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read a frame from the video")
    frame_name = f"{video_id}_frame.jpg"
    cv2.imwrite(str(UPLOAD_DIR / frame_name), content,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    # pin the crop so every processed frame is normalized exactly like this one
    (UPLOAD_DIR / f"{video_id}.meta.json").write_text(json.dumps(
        {"region": meta["region"], "span": meta["span"], "layout": meta["layout"]}))
    return {"video_id": video_id, "video_name": video_name, "frame_name": frame_name,
            "frame_url": f"/api/media/{frame_name}",
            "width": int(content.shape[1]), "height": int(content.shape[0]),
            "content_region": meta["region"], "layout": meta["layout"]}


@app.post("/api/calibrate")
async def calibrate(data: dict):
    """Solve the pose from the traced curves (same tool output as calibrate.html)."""
    video_id = data.get("video_id")
    trace = data.get("trace")
    if not video_id or not isinstance(trace, dict) or not trace.get("lines"):
        raise HTTPException(400, "Need video_id and a trace with lines")
    trace_path = CALIB_DIR / f"{video_id}_trace.json"
    trace_path.write_text(json.dumps(trace))
    cal_path = CALIB_DIR / f"{video_id}.json"

    r = subprocess.run([sys.executable, str(BASE_DIR / "calib" / "calibrate.py"),
                        str(trace_path), "-o", str(cal_path)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not cal_path.exists():
        raise HTTPException(500, (r.stderr or "calibration failed")[-1500:])
    cfg = json.loads(cal_path.read_text())
    return {"status": "ok", "quality": cfg.get("quality", {}),
            "calibration_path": str(cal_path)}


@app.post("/api/process")
async def process(data: dict):
    video_id = data.get("video_id")
    video_name = data.get("video_name")
    if not video_id or not video_name:
        raise HTTPException(400, "Missing video_id or video_name")
    cal_path = CALIB_DIR / f"{video_id}.json"
    if not cal_path.exists():
        raise HTTPException(400, "Calibrate first — no calibration for this video.")
    video_path = UPLOAD_DIR / video_name
    if not video_path.exists():
        raise HTTPException(400, "Video file not found")

    start = float(data.get("start", 0.0))
    frames = data.get("frames")            # None = whole clip
    vspan = data.get("vspan")              # None = calibration vspan / 180
    out_path = OUTPUT_DIR / f"{video_id}_track.json"
    out_video = OUTPUT_DIR / f"{video_id}_players.mp4"
    out_follow = OUTPUT_DIR / f"{video_id}_follow.mp4"
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "queued", "video_id": video_id,
                      "results_path": str(out_path), "video_path": str(out_video),
                      "follow_path": str(out_follow), "started_at": None}

    def _run():
        cmd = [sys.executable, str(BASE_DIR / "vr180_pipeline.py"),
               "--video", str(video_path), "--calibration", str(cal_path),
               "--start", str(start), "--render", str(out_video),
               "--render-follow", str(out_follow),
               "--enhance", "auto", "--out", str(out_path)]
        meta_p = UPLOAD_DIR / f"{video_id}.meta.json"
        if meta_p.exists():
            cmd += ["--region", *[str(v) for v in json.loads(meta_p.read_text())["region"]]]
        trace_p = CALIB_DIR / f"{video_id}_trace.json"
        if trace_p.exists():
            cmd += ["--trace", str(trace_p)]
        if frames:
            cmd.extend(["--frames", str(int(frames))])
        if vspan:
            cmd.extend(["--vspan", str(float(vspan))])
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
            tasks[task_id]["stderr"] = (r.stderr or "")[-2000:]
            if r.returncode != 0 or not out_path.exists():
                tasks[task_id]["status"] = "error"
                tasks[task_id]["error"] = (r.stderr or "pipeline failed")[-2000:]
            else:
                tasks[task_id]["status"] = "completed"
        except Exception as e:                       # noqa: BLE001
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    tasks[task_id]["status"] = "processing"
    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
async def status(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "unknown task")
    return {"status": t["status"], "error": t.get("error")}


@app.get("/api/video/{task_id}")
async def video(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "unknown task")
    p = Path(t.get("video_path") or "")
    if not p.exists():
        raise HTTPException(404, "no video yet")
    return FileResponse(str(p), media_type="video/mp4")


@app.get("/api/follow/{task_id}")
async def follow(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "unknown task")
    p = Path(t.get("follow_path") or "")
    if not p.exists():
        raise HTTPException(404, "no follow video yet")
    return FileResponse(str(p), media_type="video/mp4")


@app.get("/api/results/{task_id}")
async def results(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "unknown task")
    p = Path(t["results_path"])
    if not p.exists():
        raise HTTPException(404, "no results yet")
    return json.loads(p.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
