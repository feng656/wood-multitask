"""Materialize a YOLO segmentation view for bridge annotations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wood_data.defects import CATEGORY_IDS, parse_bridge_polygons_and_ignores
from wood_data.labelme import load_labelme_without_image_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def ensure_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
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


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_bridge_map(path: Path, project_root: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Path] = {}
    for item in payload.get("audits", []):
        source_relative = str(item["source_relative"])
        normalized_relative = str(item["normalized_relative"])
        result[source_relative] = project_root / normalized_relative
    return result


def normalize_polygons(annotation: dict[str, Any], width: float, height: float) -> str | None:
    segmentation = annotation.get("segmentation") or []
    if not segmentation:
        return None
    polygon = segmentation[0]
    normalized: list[str] = []
    for index, value in enumerate(polygon):
        normalized.append(f"{float(value) / (width if index % 2 == 0 else height):.6f}")
    # COCO reserves 0 for background; YOLO class indices are zero-based.
    return f"{CATEGORY_IDS[str(annotation['category'])] - 1} " + " ".join(normalized)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    workspace_root = (args.workspace_root or project_root.parent).resolve()
    output = (args.output or project_root / "outputs" / "defect_baseline" / "bridge_yolo_dataset").resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(project_root / "data_processed" / "manifests" / "manifest.csv")
    bridge_map = load_bridge_map(project_root / "data_processed" / "bridge_annotations" / "normalization_audit.json", project_root)
    selected = [row for row in manifest if row["dataset_role"] == "bridge" and row["split"] in set(args.splits)]

    summary: dict[str, Any] = {"splits": {}, "classes": ["knot", "crack"]}
    for split in args.splits:
        split_rows = [row for row in selected if row["split"] == split]
        image_count = 0
        annotation_count = 0
        empty_count = 0
        skipped_ignore_images = 0
        for row in split_rows:
            source_image = (workspace_root / row["image_path"]).resolve()
            if not source_image.exists():
                raise FileNotFoundError(source_image)

            annotation_reference = bridge_map.get(row["defect_annotation_path"], workspace_root / row["defect_annotation_path"])
            payload = load_labelme_without_image_data(annotation_reference)
            annotations, ignored = parse_bridge_polygons_and_ignores(annotation_reference)
            if ignored:
                skipped_ignore_images += 1
                continue
            image_count += 1
            destination_image = output / "images" / split / f"{row['sample_id'].replace(':', '_')}{source_image.suffix.lower() or '.png'}"
            ensure_link(source_image, destination_image)
            width = float(row["image_width"])
            height = float(row["image_height"])
            lines: list[str] = []
            for annotation in annotations:
                line = normalize_polygons(annotation, width, height)
                if line is not None:
                    lines.append(line)
                    annotation_count += 1
            if not payload.get("shapes"):
                empty_count += 1
            elif not lines:
                empty_count += 1
            write_label(output / "labels" / split / f"{row['sample_id'].replace(':', '_')}.txt", lines)

        summary["splits"][split] = {
            "images": image_count,
            "annotations": annotation_count,
            "empty_images": empty_count,
            "skipped_ignore_images": skipped_ignore_images,
        }

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: " + output.as_posix() + "\n"
        + "train: images/train\nval: images/val\n"
        + ("test: images/test\n" if "test" in args.splits else "")
        + "names:\n"
        + "  0: knot\n  1: crack\n",
        encoding="utf-8",
    )
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
