"""Convert COCO JSON annotations to YOLO format."""
import json
import os
from pathlib import Path
import shutil


def convert_coco_to_yolo(coco_json, img_dir, out_dir, class_map):
    """Convert COCO JSON annotations to YOLO .txt files.

    Args:
        coco_json: path to COCO JSON file
        img_dir: directory containing source images
        out_dir: output directory for YOLO dataset (images + labels)
        class_map: dict mapping COCO category_id -> YOLO class_id
    """
    with open(coco_json) as f:
        coco = json.load(f)

    # Build image lookup: id -> filename, (width, height)
    images = {}
    for img in coco["images"]:
        images[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Build annotation lookup: image_id -> list of annotations
    anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns.setdefault(img_id, []).append(ann)

    img_out = Path(out_dir) / "images"
    lbl_out = Path(out_dir) / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, img_info in images.items():
        src_path = Path(img_dir) / img_info["file_name"]
        if not src_path.exists():
            continue

        stem = Path(img_info["file_name"]).stem

        # Copy image
        shutil.copy2(src_path, img_out / f"{stem}.jpg")

        # Write YOLO labels
        w, h = img_info["width"], img_info["height"]
        lines = []
        for ann in anns.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in class_map:
                continue
            cls_id = class_map[cat_id]
            # COCO: [x, y, width, height] (top-left)
            # YOLO: [x_center, y_center, width, height] (normalized)
            x, y, bw, bh = ann["bbox"]
            x_c = (x + bw / 2) / w
            y_c = (y + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}")

        if lines:
            with open(lbl_out / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines))
            converted += 1

    return converted


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python coco_to_yolo.py <coco.json> <img_dir> <out_dir> [class_map_json]")
        sys.exit(1)

    coco_json = sys.argv[1]
    img_dir = sys.argv[2]
    out_dir = sys.argv[3]

    # Default: player=0, football=1
    class_map = {1: 0, 2: 1}
    if len(sys.argv) >= 5:
        with open(sys.argv[4]) as f:
            class_map = {int(k): int(v) for k, v in json.load(f).items()}

    n = convert_coco_to_yolo(coco_json, img_dir, out_dir, class_map)
    print(f"Converted {n} images with annotations to {out_dir}")
