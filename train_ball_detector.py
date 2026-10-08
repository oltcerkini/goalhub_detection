#!/usr/bin/env python3
"""Fine-tune YOLO26l on ball-only data for Phase 3 ball detection.

Usage:
    python train_ball_detector.py --data ball_dataset/data.yaml

Trains a single-class ball detector from COCO-pretrained YOLO26l at 1280px
resolution with aggressive copy-paste augmentation for small-ball detection.

Output:
    ball_detector_yolo26l.pt  (copied to project root)
"""

import argparse
import multiprocessing
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Train ball-specific YOLO detector")
    ap.add_argument("--data", default="ball_dataset/data.yaml",
                    help="Path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Training resolution (longest edge)")
    ap.add_argument("--batch", type=int, default=4,
                    help="Batch size (depends on GPU memory; reduce if OOM)")
    ap.add_argument("--model", default="yolo26l.pt",
                    help="Base model (COCO-pretrained YOLO26l or custom checkpoint)")
    ap.add_argument("--name", default="ball_yolo26l_1280",
                    help="Run name for the training output")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run extract_ball_data.py first to create the dataset.")
        return

    # Verify dataset
    train_imgs = list(data_path.parent.glob("images/train/*.jpg"))
    val_imgs = list(data_path.parent.glob("images/val/*.jpg"))
    print(f"Dataset: {len(train_imgs)} train, {len(val_imgs)} val")

    if len(train_imgs) == 0:
        print("No training images found. Check the dataset structure.")
        return

    from ultralytics import YOLO

    print("=" * 60)
    print(f"Training ball-specific detector")
    print(f"  Base model: {args.model}")
    print(f"  Dataset: {args.data} ({len(train_imgs)} train, {len(val_imgs)} val)")
    print(f"  Image size: {args.imgsz}x{args.imgsz}")
    print(f"  Batch: {args.batch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Classes: 1 (ball only)")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        cache=False,
        fraction=0.3,
        amp=True,
        project="runs/detect/ball_detector",
        name=args.name,
        exist_ok=True,
        patience=30,
        verbose=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=15,
        deterministic=False,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.05,
        scale=0.5,          # aggressive scaling for small-ball simulation
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.5,     # copy-paste ball onto different backgrounds
    )

    # Copy best model to project root with a recognizable name
    best_path = Path(f"runs/detect/ball_detector/{args.name}/weights/best.pt")
    if best_path.exists():
        dest = Path("ball_detector_yolo26l.pt")
        shutil.copy2(str(best_path), str(dest))
        print(f"\nModel copied to {dest}")

    print("\n" + "=" * 60)
    print("Training complete!")
    best = results.results_dict
    print(f"  Best mAP50: {best.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP50-95: {best.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Model: {best_path}")
    print(f"  Ready for ball_detector.py integration")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
