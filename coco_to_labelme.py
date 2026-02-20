#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


COCO_PATH = Path("roboflow/coco_json/_annotations.coco.json")
IMAGES_DIR = Path("roboflow/images")
OUT_DIR = Path("roboflow/json")
LABELME_VERSION = "5.5.0"


def polygons_from_segmentation(segmentation):
    polygons = []

    if isinstance(segmentation, dict):
        return polygons  # skip RLE

    if isinstance(segmentation, list):
        if not segmentation:
            return polygons

        if isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]

        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6:
                continue

            pts = []
            for i in range(0, len(poly) - 1, 2):
                pts.append([float(poly[i]), float(poly[i + 1])])

            if len(pts) >= 3:
                polygons.append(pts)

    return polygons


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with COCO_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    converted = 0
    skipped_rle = 0

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]  # keep full Roboflow name as-is

        image_file_path = IMAGES_DIR / file_name
        if not image_file_path.exists():
            print(f"[WARN] Missing image: {image_file_path}")
            continue

        with Image.open(image_file_path) as im:
            width, height = im.size

        shapes = []
        for ann in anns_by_image.get(image_id, []):
            label = categories.get(ann.get("category_id"), str(ann.get("category_id")))
            seg = ann.get("segmentation", [])

            if isinstance(seg, dict):
                skipped_rle += 1
                continue

            for pts in polygons_from_segmentation(seg):
                shapes.append(
                    {
                        "label": label,
                        "points": pts,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )

        labelme_obj = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": f"..\\images\\{file_name}",
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }

        out_json_path = OUT_DIR / f"{Path(file_name).stem}.json"
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(labelme_obj, f, indent=4, ensure_ascii=False)

        converted += 1

    print(f"Done. Converted {converted} images to LabelMe JSON in: {OUT_DIR}")
    if skipped_rle:
        print(f"Note: skipped {skipped_rle} RLE annotations (only polygon segmentations are converted).")


if __name__ == "__main__":
    main()
