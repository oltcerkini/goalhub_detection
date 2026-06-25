"""Round 6: YOLO26m at 1024px — fine-tuned from R3, 15 epochs.

Best bang-for-buck: 2.56x resolution of R3's 640px, fits 12GB VRAM at batch=4.
"""
import multiprocessing
from pathlib import Path

DATA_YAML = Path("semi_supervised_data/combined_round4_small/data.yaml")
CKPT_R3 = Path("runs/detect/goalhub_finetune/semi_supervised_r3/weights/best.pt")

IMGSZ = 1024
BATCH = 4
EPOCHS = 15
PATIENCE = 10
WORKERS = 2


def main():
    from ultralytics import YOLO

    yaml_path = DATA_YAML
    if not yaml_path.exists():
        print(f"Dataset not found: {yaml_path}")
        return

    ckpt = CKPT_R3 if CKPT_R3.exists() else "yolo26m.pt"
    print("=" * 60)
    print("Round 6: YOLO26m at 1024x1024 (fine-tuned from R3)")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt}")
    print(f"  Dataset: {yaml_path} (53K images)")
    print(f"  Image size: {IMGSZ}x{IMGSZ}")
    print(f"  Batch: {BATCH}")
    print(f"  Epochs: {EPOCHS}")
    print()

    model = YOLO(str(ckpt))

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        amp=True,
        project="goalhub_finetune",
        name="semi_supervised_r6",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=5,
        deterministic=False,
    )

    print("\n" + "=" * 60)
    print("Round 6 complete!")
    best = results.results_dict
    print(f"  Best mAP50: {best.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP50-95: {best.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Model: goalhub_finetune/semi_supervised_r6/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
