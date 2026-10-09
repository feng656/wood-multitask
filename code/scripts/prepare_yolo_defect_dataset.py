"""Materialize a YOLO segmentation view from the locked extended COCO files.

The source COCO files and raw images are read-only inputs.  Images in the
training view are symlinks to the existing workspace files; only normalized
YOLO label text files and a data YAML are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test", "external_test")
LABEL_FORMATS = ("segment", "box")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--source-datasets", nargs="+", default=None, help="Optional dataset_id filter, e.g. vsb vnwoodknot")
    parser.add_argument("--label-format", choices=LABEL_FORMATS, default="box")
    return parser.parse_args()


def resolve_workspace_path(value: str, workspace_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()


def decode_uncompressed_rle(segmentation: dict[str, Any]) -> np.ndarray:
    height, width = map(int, segmentation["size"])
    counts = segmentation["counts"]
    if isinstance(counts, str):
        raise ValueError("compressed COCO RLE is not supported without pycocotools")
    values = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    current = 0
    for count in counts:
        count = int(count)
        if count < 0 or cursor + count > values.size:
            raise ValueError("invalid uncompressed COCO RLE")
        if current:
            values[cursor : cursor + count] = 1
        cursor += count
        current = 1 - current
    return values.reshape((height, width), order="F")


def rle_to_polygons(segmentation: dict[str, Any]) -> list[list[float]]:
    import cv2

    mask = decode_uncompressed_rle(segmentation)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2).astype(float).ravel().tolist() for contour in contours if len(contour) >= 3]


def segmentation_polygons(segmentation: Any) -> list[list[float]]:
    if isinstance(segmentation, list):
        polygons: list[list[float]] = []
        for polygon in segmentation:
            if isinstance(polygon, list) and len(polygon) >= 6:
                polygons.append([float(value) for value in polygon])
        return polygons
    if isinstance(segmentation, dict):
        return rle_to_polygons(segmentation)
    return []


def ensure_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"refusing to replace existing dataset file: {destination}")
    try:
        destination.symlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def yolo_box_line(category_id: int, bbox: list[float], width: float, height: float) -> str:
    x, y, box_width, box_height = bbox
    center_x = (x + box_width / 2.0) / width
    center_y = (y + box_height / 2.0) / height
    return f"{category_id} {center_x:.6f} {center_y:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def convert_split(
    coco_path: Path,
    split: str,
    output: Path,
    workspace_root: Path,
    source_datasets: set[str] | None,
    label_format: str,
) -> dict[str, int]:
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): index for index, item in enumerate(sorted(payload["categories"], key=lambda x: int(x["id"]))) }
    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    annotation_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        annotation_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    image_count = 0
    annotation_count = 0
    empty_count = 0
    skipped_images = 0
    skipped_ignore_images = 0
    skipped_missing_segmentation = 0
    for image in payload.get("images", []):
        if source_datasets is not None and str(image.get("source_dataset")) not in source_datasets:
            skipped_images += 1
            continue
        image_id = int(image["id"])
        image_annotations = annotation_by_image.get(image_id, [])
        if image.get("ignore_bboxes"):
            skipped_ignore_images += 1
            continue
        if label_format == "segment" and any(
            not segmentation_polygons(annotation.get("segmentation"))
            for annotation in image_annotations
        ):
            skipped_missing_segmentation += 1
            continue
        image_count += 1
        source = resolve_workspace_path(str(image["file_name"]), workspace_root)
        if not source.exists():
            raise FileNotFoundError(f"COCO image does not exist: {source}")
        suffix = source.suffix.lower() or ".png"
        name = f"{image_id:08d}{suffix}"
        ensure_link(source, images_dir / name)
        lines: list[str] = []
        width = float(image["width"])
        height = float(image["height"])
        for annotation in image_annotations:
            category_id = int(annotation["category_id"])
            if category_id not in categories:
                raise ValueError(f"unknown category id {category_id} in {coco_path}")
            if label_format == "box":
                bbox = annotation.get("bbox") or []
                if len(bbox) != 4:
                    continue
                lines.append(yolo_box_line(categories[category_id], [float(value) for value in bbox], width, height))
                annotation_count += 1
                continue
            polygons = segmentation_polygons(annotation.get("segmentation"))
            if not polygons:
                continue
            for polygon in polygons:
                if len(polygon) < 6:
                    continue
                normalized: list[str] = []
                for index, value in enumerate(polygon):
                    normalized.append(f"{value / (width if index % 2 == 0 else height):.6f}")
                lines.append(f"{categories[category_id]} " + " ".join(normalized))
                annotation_count += 1
        if not lines:
            empty_count += 1
        write_label(labels_dir / f"{image_id:08d}.txt", lines)
    return {
        "images": image_count,
        "annotations": annotation_count,
        "empty_images": empty_count,
        "skipped_images": skipped_images,
        "skipped_ignore_images": skipped_ignore_images,
        "skipped_missing_segmentation": skipped_missing_segmentation,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    workspace_root = (args.workspace_root or project_root.parent).resolve()
    output = (args.output or project_root / "outputs" / "defect_baseline" / "yolo_dataset").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"source": "locked extended COCO", "splits": {}, "classes": []}
    source_datasets = set(args.source_datasets) if args.source_datasets else None
    for split in args.splits:
        coco_path = project_root / "data_processed" / "defect_coco" / f"instances_{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(coco_path)
        summary["splits"][split] = convert_split(coco_path, split, output, workspace_root, source_datasets, args.label_format)

    train = "images/train"
    val = "images/val"
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: " + output.as_posix() + "\n"
        + f"train: {train}\nval: {val}\n"
        + ("test: images/test\n" if "test" in args.splits else "")
        + ("external_test: images/external_test\n" if "external_test" in args.splits else "")
        + "names:\n"
        + "  0: knot\n  1: crack\n",
        encoding="utf-8",
    )
    summary["classes"] = ["knot", "crack"]
    summary["label_format"] = args.label_format
    summary["source_datasets"] = sorted(source_datasets) if source_datasets is not None else []
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
