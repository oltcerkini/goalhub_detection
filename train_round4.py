"""Round 4: YouTube matches + Roboflow + all previous data.

Pipeline:
  1. Extract 2fps from all YouTube videos
  2. Download more Roboflow datasets
  3. Pseudo-label with round 3 model
  4. Merge everything: Soccana + 1fps + 3fps + Keremberke + Roboflow + YouTube
  5. Train round 4
"""
import shutil
import subprocess
import multiprocessing
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
YOUTUBE_DIR = Path("youtube_matches")
FRAMES_DIR = Path("semi_supervised_data/youtube_frames")
PSEUDO_DIR = Path("semi_supervised_data/youtube_pseudo")
COMBINED_DIR = Path("semi_supervised_data/combined_round4")

SOCCANA_DIR = Path("soccana_dataset/V1")
PSEUDO_1FPS = Path("semi_supervised_data/pseudo_labels")
PSEUDO_3FPS = Path("semi_supervised_data/pseudo_labels_3fps")
KEREMBERKE_DIR = Path("datasets/keremberke/yolo")
ROBOFLOW_DIR = Path("datasets/roboflow_football_yolo")
DJI_DIR = Path("semi_supervised_data/dji_frames")
PSEUDO_DJI = Path("semi_supervised_data/dji_pseudo")

# Use whichever is best after round 3 completes
CKPT = Path("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")
FALLBACK_CKPT = Path("runs/detect/goalhub_finetune/semi_supervised_r2/weights/best.pt")

CONF_THRESH = 0.7
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PATIENCE = 20
EXTRACT_FPS = 2


def step0_check_model():
    """Pick the best available checkpoint."""
    if CKPT.exists():
        print(f"Using round 3 checkpoint: {CKPT}")
        return CKPT
    if FALLBACK_CKPT.exists():
        print(f"Round 3 not found, using round 2: {FALLBACK_CKPT}")
        return FALLBACK_CKPT
    print("No checkpoint found!")
    return None


def step1_extract_frames():
    """YouTube frames already extracted previously. Just verify."""
    print("=" * 60)
    print("Step 1: Verifying YouTube frames")
    print("=" * 60)
    if FRAMES_DIR.exists():
        dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
        # Quick count using first dir as sample
        sample = sum(1 for _ in dirs[0].glob("*.jpg")) if dirs else 0
        print(f"  Found {len(dirs)} video dirs, ~{len(dirs) * max(sample, 1)} estimated frames")
        return len(dirs) > 0
    print("  No frames found.")
    return False


def step2_pseudo_label(model_path):
    """Pseudo-label YouTube frames with the trained model."""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Step 2: Pseudo-labeling YouTube frames")
    print("=" * 60)

    model = YOLO(str(model_path))
    frame_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())

    total = 0
    labeled = 0

    for frame_dir in frame_dirs:
        video_name = frame_dir.name
        safe_name = video_name.encode("ascii", "replace").decode("ascii").replace("?", "_")
        out_dir = PSEUDO_DIR / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(frame_dir.glob("*.jpg"))
        for img_path in images:
            results = model(str(img_path), conf=CONF_THRESH, iou=0.5,
                           device=0, verbose=False)[0]
            stem = img_path.stem
            label_path = out_dir / f"{stem}.txt"
            img_out = out_dir / f"{stem}.jpg"

            lines = []
            if results.boxes is not None:
                for box, cls_id in zip(results.boxes.xywhn, results.boxes.cls):
                    x, y, w, h = box.tolist()
                    lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            if lines:
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))
                shutil.copy2(img_path, img_out)
                labeled += 1
            total += 1

        safe = video_name.encode("utf-8", "replace").decode("utf-8", "replace")
        print(f"  {safe}: {labeled} labeled")

    print(f"  Total: {total} frames, {labeled} with pseudo-labels")

    # Also pseudo-label DJI frames
    if DJI_DIR.exists():
        print("\n  Pseudo-labeling DJI frames...")
        dji_dirs = sorted(d for d in DJI_DIR.iterdir() if d.is_dir())
        for dji_dir in dji_dirs:
            vname = dji_dir.name
            out_dir = PSEUDO_DJI / vname
            out_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(dji_dir.glob("*.jpg"))
            for img_path in images:
                results = model(str(img_path), conf=CONF_THRESH, iou=0.5,
                               device=0, verbose=False)[0]
                stem = img_path.stem
                lines = []
                if results.boxes is not None:
                    for box, cls_id in zip(results.boxes.xywhn, results.boxes.cls):
                        x, y, w, h = box.tolist()
                        lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                if lines:
                    with open(out_dir / f"{stem}.txt", "w") as f:
                        f.write("\n".join(lines))
                    shutil.copy2(img_path, out_dir / f"{stem}.jpg")
            safe_v = vname.encode("utf-8", "replace").decode("utf-8", "replace")
            print(f"    {safe_v}: done")
        dj_total = sum(1 for _ in PSEUDO_DJI.rglob("*.txt"))
        print(f"    DJI total: {dj_total} pseudo-labels")

    return labeled > 0


def step3_merge():
    """Merge ALL data sources into one dataset."""
    print("\n" + "=" * 60)
    print("Step 3: Merging all datasets")
    print("=" * 60)

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)

    sources = []

    # 1. Soccana
    for split in ["train", "test"]:
        for sub in ["images", "labels"]:
            src = SOCCANA_DIR / sub / split
            dst = COMBINED_DIR / sub / ("train" if split == "train" else "test")
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
    sources.append("Soccana")

    # 2. 1fps pseudo-labels
    pseudo_img_dir = COMBINED_DIR / "images" / "train"
    pseudo_lbl_dir = COMBINED_DIR / "labels" / "train"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    for label_file in PSEUDO_1FPS.rglob("*.txt"):
        img_file = label_file.with_suffix(".jpg")
        if img_file.exists():
            stem = f"ps1_{label_file.parent.name}_{label_file.stem}"
            shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
            shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
    sources.append("Pseudo-1fps")

    # 3. 3fps pseudo-labels
    for label_file in PSEUDO_3FPS.rglob("*.txt"):
        img_file = label_file.with_suffix(".jpg")
        if img_file.exists():
            stem = f"ps3_{label_file.parent.name}_{label_file.stem}"
            shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
            shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
    sources.append("Pseudo-3fps")

    # 4. Keremberke
    for split in ["train", "valid"]:
        dst_s = "train" if split == "train" else "test"
        img_src = KEREMBERKE_DIR / split / "images"
        lbl_src = KEREMBERKE_DIR / split / "labels"
        if img_src.exists():
            for img_path in img_src.glob("*"):
                stem = f"ker_{img_path.stem}"
                lbl_path = lbl_src / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(img_path, COMBINED_DIR / "images" / dst_s / f"{stem}.jpg")
                    shutil.copy2(lbl_path, COMBINED_DIR / "labels" / dst_s / f"{stem}.txt")
    sources.append("Keremberke")

    # 5. Roboflow
    for split in ["train", "valid"]:
        dst_s = "train" if split == "train" else "test"
        img_src = ROBOFLOW_DIR / split / "images"
        lbl_src = ROBOFLOW_DIR / split / "labels"
        if img_src.exists():
            for img_path in img_src.glob("*"):
                stem = f"rf_{img_path.stem}"
                lbl_path = lbl_src / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(img_path, COMBINED_DIR / "images" / dst_s / f"{stem}.jpg")
                    shutil.copy2(lbl_path, COMBINED_DIR / "labels" / dst_s / f"{stem}.txt")
    sources.append("Roboflow")

    # 6. DJI pseudo-labels (if available)
    if PSEUDO_DJI.exists():
        dji_count = 0
        for label_file in PSEUDO_DJI.rglob("*.txt"):
            img_file = label_file.with_suffix(".jpg")
            if img_file.exists():
                stem = f"dji_{label_file.parent.name}_{label_file.stem}"
                shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
                shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
                dji_count += 1
        if dji_count > 0:
            print(f"  DJI pseudo-labels: {dji_count}")
            sources.append(f"DJI ({dji_count})")

    # 7. YouTube pseudo-labels (if available)
    if PSEUDO_DIR.exists():
        yt_count = 0
        for label_file in PSEUDO_DIR.rglob("*.txt"):
            img_file = label_file.with_suffix(".jpg")
            if img_file.exists():
                stem = f"yt_{label_file.parent.name}_{label_file.stem}"
                shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
                shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
                yt_count += 1
        if yt_count > 0:
            sources.append(f"YouTube ({yt_count})")

    total_train = len(list((COMBINED_DIR / "images/train").glob("*")))
    total_test = len(list((COMBINED_DIR / "images/test").glob("*")))

    print(f"  Sources: {', '.join(sources)}")
    print(f"  Train images: {total_train}")
    print(f"  Test images: {total_test}")

    yaml_path = COMBINED_DIR / "data.yaml"
    yaml_path.write_text(f"""path: {COMBINED_DIR.resolve().as_posix()}
train: images/train
val: images/test
nc: 3
names: ["Player", "Ball", "Referee"]
""")
    print(f"  data.yaml written")
    return True


def step4_train():
    """Train round 4 on the mega-dataset."""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Step 4: Training round 4")
    print("=" * 60)

    yaml_path = COMBINED_DIR / "data.yaml"
    model_path = step0_check_model()
    if not model_path:
        return False

    model = YOLO(str(model_path))
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        project="goalhub_finetune",
        name="semi_supervised_r4",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
    )
    print(f"\nRound 4 complete! Best mAP50: {results.results_dict}")
    return True


if __name__ == "__main__":
    multiprocessing.freeze_support()

    model_path = step0_check_model()
    if not model_path:
        exit(1)

    step1_extract_frames()
    step2_pseudo_label(model_path)
    step3_merge()
    step4_train()

    print("\n" + "=" * 60)
    print("Round 4 complete!")
    print("=" * 60)
