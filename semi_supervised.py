"""Semi-supervised pipeline: pseudo-label custom video frames + train round 2.

Run AFTER supervised training on Soccana completes:
  python semi_supervised.py

Steps:
  1. Run fine-tuned model on extracted frames → YOLO-format pseudo-labels
  2. Keep only high-confidence detections (conf > 0.7)
  3. Copy frames + labels into a merged dataset with Soccana
  4. Fine-tune round 2 on combined data
"""
import shutil
import multiprocessing
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRAMES_DIR = Path("semi_supervised_data/raw_frames")          # from extract_frames.py
PSEUDO_DIR = Path("semi_supervised_data/pseudo_labels")       # temp output
COMBINED_DIR = Path("semi_supervised_data/combined_dataset")  # merged dataset
SOCCANA_DIR = Path("soccana_dataset/V1")

# Which fine-tuned checkpoint to use for pseudo-labeling
CKPT = Path("soccana_dataset/runs/detect/goalhub_finetune/yolo26m_soccana/weights/best.pt")

# Pseudo-label confidence threshold
CONF_THRESH = 0.7

# Round 2 training params
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PATIENCE = 20


def step1_pseudo_label():
    """Run inference on all extracted frames, generate YOLO-format labels."""
    from ultralytics import YOLO

    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}")
        print("Have you completed the supervised training yet?")
        return False

    print("=" * 60)
    print("Step 1: Pseudo-labeling extracted frames")
    print("=" * 60)

    model = YOLO(str(CKPT))
    frame_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())

    total_frames = 0
    labeled_frames = 0

    for frame_dir in frame_dirs:
        video_name = frame_dir.name
        out_dir = PSEUDO_DIR / video_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(frame_dir.glob("*.jpg"))
        print(f"  Processing {video_name}: {len(images)} frames...")

        for img_path in images:
            results = model(str(img_path), conf=CONF_THRESH, iou=0.5,
                            device=0, verbose=False)[0]

            frame_stem = img_path.stem
            label_path = out_dir / f"{frame_stem}.txt"
            img_out_path = PSEUDO_DIR / video_name / f"{frame_stem}.jpg"

            lines = []
            if results.boxes is not None:
                for box, cls_id, conf in zip(results.boxes.xywhn,
                                              results.boxes.cls,
                                              results.boxes.conf):
                    x, y, w, h = box.tolist()
                    lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            if lines:
                # Save label file
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))
                # Copy image to pseudo dir
                shutil.copy2(img_path, img_out_path)
                labeled_frames += 1
            total_frames += 1

        print(f"    -> {labeled_frames} frames with detections above {CONF_THRESH}")

    total_with_dets = sum(1 for _ in PSEUDO_DIR.rglob("*.txt"))
    print(f"\n  Total frames: {total_frames}, with pseudo-labels: {total_with_dets}")
    return True


def step2_merge_datasets():
    """Merge pseudo-labeled frames with Soccana dataset into combined dataset."""
    print("\n" + "=" * 60)
    print("Step 2: Merging pseudo-labels with Soccana dataset")
    print("=" * 60)

    # Remove old combined dataset if exists
    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)

    # Copy Soccana data
    for split in ["train", "test"]:
        src_img = SOCCANA_DIR / "images" / split
        src_lbl = SOCCANA_DIR / "labels" / split
        dst_img = COMBINED_DIR / "images" / split
        dst_lbl = COMBINED_DIR / "labels" / split

        if src_img.exists():
            shutil.copytree(src_img, dst_img)
        if src_lbl.exists():
            shutil.copytree(src_lbl, dst_lbl)

    # Count Soccana images
    soccana_count = len(list((COMBINED_DIR / "images/train").glob("*")))
    print(f"  Soccana train images: {soccana_count}")

    # Copy pseudo-labeled frames into the train split
    pseudo_img_dir = COMBINED_DIR / "images" / "train"
    pseudo_lbl_dir = COMBINED_DIR / "labels" / "train"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    pseudo_count = 0
    for label_file in PSEUDO_DIR.rglob("*.txt"):
        # Corresponding image
        img_file = label_file.with_suffix(".jpg")
        if not img_file.exists():
            continue
        # Use unique name to avoid collisions
        stem = f"pseudo_{label_file.parent.name}_{label_file.stem}"
        shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
        shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
        pseudo_count += 1

    print(f"  Pseudo-labeled frames added: {pseudo_count}")
    print(f"  Total train images: {soccana_count + pseudo_count}")

    # Create data.yaml
    yaml_path = COMBINED_DIR / "data.yaml"
    yaml_content = f"""path: {COMBINED_DIR.resolve().as_posix()}
train: images/train
val: images/test
nc: 3
names: ["Player", "Ball", "Referee"]
"""
    yaml_path.write_text(yaml_content)
    print(f"  data.yaml written to {yaml_path}")
    return True


def step3_train_round2():
    """Fine-tune on combined dataset."""
    import torch
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Step 3: Training round 2 on combined dataset")
    print("=" * 60)

    yaml_path = COMBINED_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"  Dataset not found at {yaml_path}. Run step 2 first.")
        return False

    # Start from the Soccana fine-tuned weights
    model = YOLO(str(CKPT))

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
        name="semi_supervised_r2",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
    )
    print("\nRound 2 training complete!")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return True


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not step1_pseudo_label():
        print("\nPseudo-labeling skipped or failed. Exiting.")
        exit(1)

    step2_merge_datasets()
    step3_train_round2()

    print("\n" + "=" * 60)
    print("Semi-supervised pipeline complete!")
    print("=" * 60)
