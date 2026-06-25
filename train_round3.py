"""Round 3: pseudo-label 3fps frames + merge extra datasets + train.

Steps:
  1. Run round 2 model on the 840 3fps frames → pseudo-labels
  2. Merge: Soccana (8677) + 3fps pseudo-labels (840) + keremberke (1098) = ~10615
  3. Train round 3 for 50 epochs
"""
import shutil
import multiprocessing
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
FRAMES_3FPS = Path("semi_supervised_data/raw_frames_3fps")
PSEUDO_DIR = Path("semi_supervised_data/pseudo_labels_3fps")
COMBINED_DIR = Path("semi_supervised_data/combined_round3")
SOCCANA_DIR = Path("soccana_dataset/V1")
KEREMBERKE_DIR = Path("datasets/keremberke/yolo")

CKPT = Path("runs/detect/goalhub_finetune/semi_supervised_r2/weights/best.pt")
CONF_THRESH = 0.7
EPOCHS = 50
BATCH = 16
IMGSZ = 640
PATIENCE = 20


def step1_pseudo_label_3fps():
    """Run round 2 model on 3fps frames → YOLO labels."""
    from ultralytics import YOLO

    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}")
        return False

    print("=" * 60)
    print("Step 1: Pseudo-labeling 3fps frames with round 2 model")
    print("=" * 60)

    model = YOLO(str(CKPT))
    frame_dirs = sorted(d for d in FRAMES_3FPS.iterdir() if d.is_dir())

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
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))
                shutil.copy2(img_path, img_out_path)
                labeled_frames += 1
            total_frames += 1

        print(f"    -> {labeled_frames} frames with detections")

    print(f"\n  Total: {total_frames} frames, {labeled_frames} with pseudo-labels")
    return True


def step2_merge():
    """Merge Soccana + pseudo-labels + keremberke into one dataset."""
    print("\n" + "=" * 60)
    print("Step 2: Merging datasets")
    print("=" * 60)

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)

    counts = {}

    # 1. Copy Soccana (train + test)
    for split in ["train", "test"]:
        src_img = SOCCANA_DIR / "images" / split
        src_lbl = SOCCANA_DIR / "labels" / split
        dst_img = COMBINED_DIR / "images" / split
        dst_lbl = COMBINED_DIR / "labels" / split
        if src_img.exists():
            shutil.copytree(src_img, dst_img)
        if src_lbl.exists():
            shutil.copytree(src_lbl, dst_lbl)

    counts["soccana_train"] = len(list((COMBINED_DIR / "images/train").glob("*")))

    # 2. Copy pseudo-labeled 3fps frames → train
    pseudo_img_dir = COMBINED_DIR / "images" / "train"
    pseudo_lbl_dir = COMBINED_DIR / "labels" / "train"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    pseudo_count = 0
    for label_file in PSEUDO_DIR.rglob("*.txt"):
        img_file = label_file.with_suffix(".jpg")
        if not img_file.exists():
            continue
        stem = f"r3_{label_file.parent.name}_{label_file.stem}"
        shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
        shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
        pseudo_count += 1
    counts["pseudo_3fps"] = pseudo_count

    # 3. Copy keremberke dataset → train+valid
    for split in ["train", "valid"]:
        src_img = KEREMBERKE_DIR / split / "images"
        src_lbl = KEREMBERKE_DIR / split / "labels"
        if not src_img.exists():
            continue
        dst_split = "train" if split == "train" else "test"
        dst_img = COMBINED_DIR / "images" / dst_split
        dst_lbl = COMBINED_DIR / "labels" / dst_split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in src_img.glob("*"):
            stem = f"keremberke_{img_path.stem}"
            lbl_path = src_lbl / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(img_path, dst_img / f"{stem}.jpg")
                shutil.copy2(lbl_path, dst_lbl / f"{stem}.txt")
                counts[f"keremberke_{split}"] = counts.get(f"keremberke_{split}", 0) + 1

    total_train = len(list((COMBINED_DIR / "images/train").glob("*")))
    print(f"  Soccana train: {counts.get('soccana_train', 0)}")
    print(f"  Pseudo 3fps:  {pseudo_count}")
    print(f"  Keremberke:    {counts.get('keremberke_train', 0)} train, {counts.get('keremberke_valid', 0)} test")
    print(f"  Total train:   {total_train}")

    yaml_path = COMBINED_DIR / "data.yaml"
    yaml_path.write_text(f"""path: {COMBINED_DIR.resolve().as_posix()}
train: images/train
val: images/test
nc: 3
names: ["Player", "Ball", "Referee"]
""")
    print(f"  data.yaml written")
    return True


def step3_train():
    """Train round 3 on the combined dataset."""
    import torch
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Step 3: Training round 3")
    print("=" * 60)

    yaml_path = COMBINED_DIR / "data.yaml"
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
        name="semi_supervised_r3",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
    )
    print("\nRound 3 complete!")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return True


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if step1_pseudo_label_3fps() and step2_merge() and step3_train():
        print("\n" + "=" * 60)
        print("Round 3 complete!")
        print("=" * 60)
    else:
        print("Round 3 failed.")
