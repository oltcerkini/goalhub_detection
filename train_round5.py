"""Round 5: Train YOLO26l (large) on merged dataset for maximum accuracy.

Uses the combined dataset from round 4 and a larger YOLO26l backbone.
Batch size reduced to 8 to fit 12GB VRAM on RTX 5070.
"""
import multiprocessing
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
# Use the round 4 merged dataset
DATA_YAML = Path("semi_supervised_data/combined_round4/data.yaml")
FALLBACK_YAML = Path("semi_supervised_data/combined_round3/data.yaml")

# Start from COCO-pretrained YOLO26l (largest 26-series model)
BASE_MODEL = "yolo26l.pt"

EPOCHS = 50
BATCH = 8          # reduced from 16 to fit 12GB VRAM with the larger model
IMGSZ = 640
PATIENCE = 25      # more patience since larger model converges slower
WORKERS = 2


def main():
    from ultralytics import YOLO

    # Pick dataset
    yaml_path = DATA_YAML if DATA_YAML.exists() else FALLBACK_YAML
    if not yaml_path.exists():
        print(f"Dataset not found at {yaml_path}")
        return

    print("=" * 60)
    print("Round 5: YOLO26l training")
    print("=" * 60)
    print(f"  Model: {BASE_MODEL}")
    print(f"  Dataset: {yaml_path}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH}")
    print(f"  Image size: {IMGSZ}")
    print()

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        cache=False,
        amp=True,
        project="goalhub_finetune",
        name="yolo26l_r5",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
        # YOLO26l specific: lower learning rate for stable convergence
        lr0=0.005,           # half of default 0.01
        lrf=0.005,           # don't decay as aggressively
        warmup_epochs=5,     # longer warmup for larger model
        cos_lr=True,         # cosine schedule for smoother convergence
    )

    print("\n" + "=" * 60)
    print("Round 5 complete!")
    best = results.results_dict
    print(f"  Best mAP50: {best.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Best mAP50-95: {best.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Model: goalhub_finetune/yolo26l_r5/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
