from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wood_data.defects import mask_to_uncompressed_rle, save_coco


def main() -> None:
    repaired = 0
    by_file: dict[str, int] = {}
    for path in sorted((PROJECT_ROOT / "data_processed" / "defect_coco").glob("instances_*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in document["images"]}
        grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in document["annotations"]:
            if annotation.get("source_dataset") == "vsb" and not annotation.get("segmentation"):
                grouped[int(annotation["image_id"])].append(annotation)
        file_repairs = 0
        for image_id, annotations in grouped.items():
            image = images[image_id]
            mask_path = PROJECT_ROOT / str(image["unified_semantic_mask"])
            unified = np.asarray(Image.open(mask_path))
            for annotation in annotations:
                binary = (unified == int(annotation["category_id"])).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                x, y, width, height = map(float, annotation["bbox"])
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2 = min(binary.shape[1], int(np.ceil(x + width)))
                y2 = min(binary.shape[0], int(np.ceil(y + height)))
                best = max(
                    range(1, count),
                    key=lambda component: int(np.count_nonzero(labels[y1:y2, x1:x2] == component)),
                    default=0,
                )
                if best == 0 or not np.any(labels[y1:y2, x1:x2] == best):
                    raise ValueError(f"no category component for annotation {annotation['id']} in {path.name}")
                component_mask = labels == best
                annotation["segmentation"] = mask_to_uncompressed_rle(component_mask)
                annotation["area"] = int(stats[best, cv2.CC_STAT_AREA])
                annotation["segmentation_encoding"] = "uncompressed_rle_fallback"
                repaired += 1
                file_repairs += 1
        if file_repairs:
            save_coco(path, document)
            by_file[path.name] = file_repairs
    summary_path = PROJECT_ROOT / "data_processed" / "defect_coco" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["vsb_rle_fallbacks"] = repaired
    summary["vsb_rle_fallbacks_by_file"] = by_file
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"repaired": repaired, "by_file": by_file}, indent=2))


if __name__ == "__main__":
    main()
