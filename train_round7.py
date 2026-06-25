"""Round 7: Self-training loop — re-pseudo-label with best model, then retrain.

After round 6 (YOLO26l at 1280), this round:
1. Re-pseudo-labels ALL frames (YouTube + DJI) with the 0.92 model
2. Filters low-confidence detections more aggressively
3. Retrains from scratch on cleaner labels

This typically gives +1-2% mAP since the labels are better than round 4's.
"""
import multiprocessing
from pathlib import Path
import shutil
import sys


# ── Config ────────────────────────────────────────────────────────────────
# Sources to re-pseudo-label with the improved model
FRAME_SOURCES = [
    Path("semi_supervised_data/youtube_frames"),
    Path("semi_supervised_data/dji_frames"),
]
PSEUDO_OUT = Path("semi_supervised_data/self_train_pseudo")

# Combined output
COMBINED_DIR = Path("semi_supervised_data/combined_round7")
SOCCANA_DIR = Path("soccana_dataset/V1")
KEREMBERKE_DIR = Path("datasets/keremberke/yolo")
ROBOFLOW_DIR = Path("datasets/roboflow_football_yolo")

# Use best available checkpoint (prefer round 6, then round 5)
CKPT_R6 = Path("runs/detect/goalhub_finetune/yolo26l_r6_1280/weights/best.pt")
CKPT_R5 = Path("runs/detect/goalhub_finetune/yolo26l_r5/weights/best.pt")

CONF_THRESH = 0.75  # stricter than round 4's 0.7 — cleaner labels
EPOCHS = 50
BATCH = 4
IMGSZ = 1280
PATIENCE = 30


def get_checkpoint():
    for ckpt in [CKPT_R6, CKPT_R5]:
        if ckpt.exists():
            return ckpt
    print("No round 5 or 6 checkpoint found! Run those first.")
    sys.exit(1)


def step1_relabel():
    """Re-pseudo-label all YouTube + DJI frames with the latest model."""
    from ultralytics import YOLO

    ckpt = get_checkpoint()
    print("=" * 60)
    print("Step 1: Self-training re-label")
    print(f"  Model: {ckpt}")
    print(f"  Confidence threshold: {CONF_THRESH}")
    print("=" * 60)

    model = YOLO(str(ckpt))
    total_labeled = 0

    for src_dir in FRAME_SOURCES:
        if not src_dir.exists():
            continue
        video_dirs = sorted(d for d in src_dir.iterdir() if d.is_dir())
        for vdir in video_dirs:
            out_dir = PSEUDO_OUT / vdir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(vdir.glob("*.jpg"))
            labeled = 0

            for img_path in images:
                results = model(str(img_path), conf=CONF_THRESH, iou=0.5,
                               device=0, verbose=False, imgsz=IMGSZ)[0]
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
                    labeled += 1

            total_labeled += labeled
            safe = vdir.name.encode("utf-8", "replace").decode("utf-8", "replace")[:50]
            print(f"  {safe}: {labeled}/{len(images)} labeled")

    print(f"\nTotal: {total_labeled} frames re-labeled")
    return total_labeled > 0


def step2_merge():
    """Merge all data sources + new cleaner pseudo-labels."""
    print("\n" + "=" * 60)
    print("Step 2: Merging datasets")
    print("=" * 60)

    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)

    sources = []

    # Soccana
    for split in ["train", "test"]:
        for sub in ["images", "labels"]:
            src = SOCCANA_DIR / sub / split
            dst = COMBINED_DIR / sub / ("train" if split == "train" else "test")
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
    sources.append("Soccana")

    # Keremberke
    for split in ["train", "valid"]:
        dst_s = "train" if split == "train" else "test"
        img_src = KEREMBERKE_DIR / split / "images"
        lbl_src = KEREMBERKE_DIR / split / "labels"
        if img_src.exists():
            for img_path in img_src.glob("*"):
                stem = f"ker_{img_path.stem}"
                lbl = lbl_src / f"{img_path.stem}.txt"
                if lbl.exists():
                    shutil.copy2(img_path, COMBINED_DIR / "images" / dst_s / f"{stem}.jpg")
                    shutil.copy2(lbl, COMBINED_DIR / "labels" / dst_s / f"{stem}.txt")
    sources.append("Keremberke")

    # Roboflow
    for split in ["train", "valid"]:
        dst_s = "train" if split == "train" else "test"
        img_src = ROBOFLOW_DIR / split / "images"
        lbl_src = ROBOFLOW_DIR / split / "labels"
        if img_src.exists():
            for img_path in img_src.glob("*"):
                stem = f"rf_{img_path.stem}"
                lbl = lbl_src / f"{img_path.stem}.txt"
                if lbl.exists():
                    shutil.copy2(img_path, COMBINED_DIR / "images" / dst_s / f"{stem}.jpg")
                    shutil.copy2(lbl, COMBINED_DIR / "labels" / dst_s / f"{stem}.txt")
    sources.append("Roboflow")

    # Self-trained pseudo-labels (higher quality than round 4's)
    pseudo_img_dir = COMBINED_DIR / "images" / "train"
    pseudo_lbl_dir = COMBINED_DIR / "labels" / "train"
    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_lbl_dir.mkdir(parents=True, exist_ok=True)

    pseudo_count = 0
    for label_file in PSEUDO_OUT.rglob("*.txt"):
        img_file = label_file.with_suffix(".jpg")
        if img_file.exists():
            stem = f"st_{label_file.parent.name}_{label_file.stem}"
            shutil.copy2(img_file, pseudo_img_dir / f"{stem}.jpg")
            shutil.copy2(label_file, pseudo_lbl_dir / f"{stem}.txt")
            pseudo_count += 1
    sources.append(f"Self-train ({pseudo_count})")

    total_train = len(list((COMBINED_DIR / "images/train").glob("*")))
    print(f"  Sources: {', '.join(sources)}")
    print(f"  Total train: {total_train}")

    (COMBINED_DIR / "data.yaml").write_text(
        f"path: {COMBINED_DIR.resolve().as_posix()}\n"
        "train: images/train\nval: images/test\nnc: 3\n"
        'names: ["Player", "Ball", "Referee"]\n'
    )
    return True


def step3_train():
    """Train round 7 on cleaner labels."""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Step 3: Training round 7")
    print("=" * 60)

    yaml_path = COMBINED_DIR / "data.yaml"
    model = YOLO(str(get_checkpoint()))

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
        name="yolo26l_r7_self_train",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
        lr0=0.003,
        warmup_epochs=3,
        cos_lr=True,
    )

    print("\n" + "=" * 60)
    print("Round 7 complete!")
    best = results.results_dict
    print(f"  Best mAP50: {best.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP50-95: {best.get('metrics/mAP50-95(B)', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if step1_relabel() and step2_merge() and step3_train():
        print("\nAll done!")
