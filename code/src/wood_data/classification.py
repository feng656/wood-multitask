from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


def crop_box_with_context(
    image: Image.Image,
    bbox: list[float],
    context: float = 0.20,
    output_size: int = 320,
) -> Image.Image:
    x, y, width, height = bbox
    side = max(width, height) * (1.0 + 2.0 * context)
    side = max(side, 2.0)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    crop = image.crop((center_x - side / 2, center_y - side / 2, center_x + side / 2, center_y + side / 2))
    return crop.resize((output_size, output_size), Image.Resampling.LANCZOS)


def normal_center_crop(image: Image.Image, output_size: int = 320) -> Image.Image:
    return ImageOps.fit(image, (output_size, output_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def generate_classification_crops(
    coco_documents: list[dict[str, object]],
    workspace_root: Path,
    output_root: Path,
    index_path: Path,
    context: float = 0.20,
    output_size: int = 320,
    exclude_datasets: set[str] | None = None,
    initial_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = list(initial_rows or [])
    class_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        class_counts[str(row["category"])] += 1
    exclude_datasets = exclude_datasets or set()
    for document in coco_documents:
        images = {int(item["id"]): item for item in document["images"]}
        annotations: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in document["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for image_id, item in images.items():
            source = workspace_root / str(item["file_name"])
            selected = annotations.get(image_id, [])
            is_verified_normal = item.get("defect_label_state") == "verified_negative"
            source_dataset = str(item["source_dataset"])
            if source_dataset in exclude_datasets:
                continue
            if not selected and not (
                is_verified_normal and source_dataset in {"vsb", "vnwoodknot", "mokume", "oulu"}
            ):
                continue
            with Image.open(source) as raw_image:
                image = raw_image.convert("RGB")
                if selected:
                    for annotation in selected:
                        category = str(annotation["category"])
                        crop = crop_box_with_context(image, list(annotation["bbox"]), context, output_size)
                        name = f"{source_dataset}_{image_id:06d}_{int(annotation['id']):07d}.jpg"
                        destination = output_root / str(item["split"]) / category / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists():
                            crop.save(destination, quality=92, subsampling=0)
                        class_counts[category] += 1
                        rows.append({
                            "crop_path": destination.relative_to(workspace_root).as_posix(),
                            "source_image": item["file_name"], "source_dataset": source_dataset,
                            "group_id": item["group_id"], "split": item["split"], "category": category,
                            "subtype": annotation.get("subtype", ""),
                            "attributes": ";".join(annotation.get("attributes", [])),
                            "source_class": annotation.get("source_class", ""),
                            "annotation_geometry": annotation.get("geometry_type", "bbox"),
                            "bbox": ",".join(str(value) for value in annotation["bbox"]),
                        })
                else:
                    crop = normal_center_crop(image, output_size)
                    name = f"{source_dataset}_{image_id:06d}_normal.jpg"
                    destination = output_root / str(item["split"]) / "normal" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        crop.save(destination, quality=92, subsampling=0)
                    class_counts["normal"] += 1
                    rows.append({
                        "crop_path": destination.relative_to(workspace_root).as_posix(),
                        "source_image": item["file_name"], "source_dataset": source_dataset,
                        "group_id": item["group_id"], "split": item["split"], "category": "normal",
                        "subtype": "", "attributes": "verified_negative", "bbox": "",
                        "source_class": "verified_negative", "annotation_geometry": "image_level",
                    })
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "crop_path", "source_image", "source_dataset", "group_id", "split", "category",
            "subtype", "attributes", "source_class", "annotation_geometry", "bbox",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"total": len(rows), "class_counts": dict(sorted(class_counts.items()))}
