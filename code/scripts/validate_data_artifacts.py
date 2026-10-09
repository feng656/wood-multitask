from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PROCESSED = PROJECT_ROOT / "data_processed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate() -> dict[str, object]:
    manifest = read_csv(PROCESSED / "manifests" / "manifest.csv")
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    manifest_by_path = {row["image_path"]: row for row in manifest}
    errors: list[str] = []
    warnings: list[str] = []

    ring_rows = read_csv(PROCESSED / "ring_targets" / "ring_targets.csv")
    expected_ring = {row["sample_id"] for row in manifest if row["has_ring"] == "1"}
    actual_ring = {row["sample_id"] for row in ring_rows if row["status"] == "ok"}
    if actual_ring != expected_ring:
        errors.append(f"ring target sample mismatch: missing={len(expected_ring - actual_ring)}, extra={len(actual_ring - expected_ring)}")
    ring_missing_paths = 0
    ring_dimension_errors = 0
    for row in ring_rows:
        expected_size = (int(row["target_width"]), int(row["target_height"]))
        model_image = WORKSPACE_ROOT / row["model_image_path"]
        if not model_image.exists():
            ring_missing_paths += 1
        else:
            with Image.open(model_image) as image:
                if image.size != expected_size:
                    ring_dimension_errors += 1
        for field in ("skeleton_path", "boundary_path", "distance_path", "instance_path", "valid_path"):
            path = PROJECT_ROOT / row[field]
            if not path.exists():
                ring_missing_paths += 1
                continue
            with Image.open(path) as image:
                if image.size != expected_size:
                    ring_dimension_errors += 1
    if ring_missing_paths:
        errors.append(f"missing ring target files: {ring_missing_paths}")
    if ring_dimension_errors:
        errors.append(f"ring target dimension mismatches: {ring_dimension_errors}")

    coco_files = sorted((PROCESSED / "defect_coco").glob("instances_*.json"))
    coco_images = 0
    coco_annotations = 0
    coco_segmented = 0
    coco_bbox_errors = 0
    coco_reference_errors = 0
    coco_group_split_errors = 0
    coco_category_errors = 0
    coco_ignore_bbox_errors = 0
    coco_ignore_regions = 0
    coco_dataset_counts: Counter[str] = Counter()
    coco_category_counts: Counter[str] = Counter()
    for path in coco_files:
        document = json.loads(path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in document["images"]}
        valid_categories = {int(item["id"]) for item in document["categories"]}
        category_names = {str(item["name"]) for item in document["categories"]}
        if category_names != {"knot", "crack"}:
            coco_category_errors += 1
        coco_images += len(images)
        if len(images) != len(document["images"]):
            errors.append(f"duplicate COCO image IDs in {path.name}")
        annotation_ids = [int(item["id"]) for item in document["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            errors.append(f"duplicate COCO annotation IDs in {path.name}")
        for image in images.values():
            coco_dataset_counts[str(image["source_dataset"])] += 1
            source = WORKSPACE_ROOT / str(image["file_name"])
            if not source.exists():
                coco_reference_errors += 1
            manifest_row = manifest_by_sample.get(str(image["sample_id"]))
            if not manifest_row or manifest_row["split"] != image["split"] or manifest_row["group_id"] != image["group_id"]:
                coco_group_split_errors += 1
            for ignored in image.get("ignore_bboxes", []):
                coco_ignore_regions += 1
                x, y, width, height = map(float, ignored["bbox"])
                if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > float(image["width"]) + 1 or y + height > float(image["height"]) + 1:
                    coco_ignore_bbox_errors += 1
        for annotation in document["annotations"]:
            coco_annotations += 1
            coco_category_counts[str(annotation.get("category", ""))] += 1
            image = images.get(int(annotation["image_id"]))
            if image is None or int(annotation["category_id"]) not in valid_categories:
                coco_reference_errors += 1
                continue
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 0 or height <= 0 or x < -1 or y < -1 or x + width > float(image["width"]) + 1 or y + height > float(image["height"]) + 1:
                coco_bbox_errors += 1
            segmentation = annotation.get("segmentation", [])
            if segmentation:
                coco_segmented += 1
                if isinstance(segmentation, dict):
                    if segmentation.get("size") != [int(image["height"]), int(image["width"])] or not segmentation.get("counts"):
                        coco_reference_errors += 1
                elif any(len(polygon) < 6 or len(polygon) % 2 for polygon in segmentation):
                    coco_reference_errors += 1
    if coco_bbox_errors:
        errors.append(f"invalid or out-of-bounds COCO boxes: {coco_bbox_errors}")
    if coco_reference_errors:
        errors.append(f"invalid COCO references/segmentations: {coco_reference_errors}")
    if coco_group_split_errors:
        errors.append(f"COCO group/split inheritance errors: {coco_group_split_errors}")
    if coco_category_errors:
        errors.append(f"COCO category contract errors: {coco_category_errors}")
    if coco_ignore_bbox_errors:
        errors.append(f"invalid COCO ignore boxes: {coco_ignore_bbox_errors}")
    expected_coco_images = sum(
        row["supervision_scope"] == "primary" and row["defect_label_state"] != "missing"
        for row in manifest
    )
    expected_coco_annotations = sum(
        int(row.get("defect_instance_count") or 0)
        for row in manifest
        if row["supervision_scope"] == "primary" and row["defect_label_state"] != "missing"
    )
    expected_ignore_regions = sum(
        int(row.get("ignored_defect_instance_count") or 0)
        for row in manifest
        if row["dataset_id"] != "vsb" and row["supervision_scope"] == "primary"
    )
    if coco_images != expected_coco_images:
        errors.append(f"COCO image coverage mismatch: expected={expected_coco_images}, actual={coco_images}")
    if coco_annotations != expected_coco_annotations:
        errors.append(
            f"COCO target instance mismatch: expected={expected_coco_annotations}, actual={coco_annotations}"
        )
    if coco_ignore_regions != expected_ignore_regions:
        errors.append(
            f"COCO ignore-region mismatch: expected={expected_ignore_regions}, actual={coco_ignore_regions}"
        )

    crop_index = PROCESSED / "classification_crops" / "classification_crops.csv"
    crop_rows = read_csv(crop_index) if crop_index.exists() else []
    crop_missing = 0
    crop_inheritance_errors = 0
    for row in crop_rows:
        crop_path = Path(row["crop_path"])
        if not crop_path.is_absolute():
            crop_path = WORKSPACE_ROOT / crop_path
        if not crop_path.exists():
            crop_missing += 1
        source = manifest_by_path.get(row["source_image"])
        if not source or source["split"] != row["split"] or source["group_id"] != row["group_id"]:
            crop_inheritance_errors += 1
    actual_crop_files = sum(1 for path in (PROCESSED / "classification_crops").rglob("*.jpg") if path.is_file())
    project_crop_rows = sum(row["crop_path"].startswith("wood-multitask-final/") for row in crop_rows)
    if crop_missing:
        errors.append(f"missing indexed classification crops: {crop_missing}")
    if crop_inheritance_errors:
        errors.append(f"classification crop group/split inheritance errors: {crop_inheritance_errors}")
    if crop_rows and actual_crop_files != project_crop_rows:
        errors.append(f"materialized classification crop file/index mismatch: files={actual_crop_files}, local_rows={project_crop_rows}")
    crop_categories = {row["category"] for row in crop_rows}
    if not crop_categories.issubset({"normal", "knot", "crack"}):
        errors.append(f"unexpected classification categories: {sorted(crop_categories - {'normal', 'knot', 'crack'})}")
    expected_normal_crops = sum(
        row["defect_label_state"] == "verified_negative"
        and row["dataset_id"] in {"vsb", "vnwoodknot", "mokume", "oulu"}
        for row in manifest
    )
    if len(crop_rows) != expected_coco_annotations + expected_normal_crops:
        errors.append(
            "classification crop coverage mismatch: "
            f"expected={expected_coco_annotations + expected_normal_crops}, actual={len(crop_rows)}"
        )

    bridge_path = PROCESSED / "bridge_annotations" / "normalization_audit.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8")) if bridge_path.exists() else {}
    duplicate_path = PROCESSED / "duplicate_audit" / "summary.json"
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8")) if duplicate_path.exists() else {}
    visual_review_path = PROCESSED / "duplicate_audit" / "visual_review.json"
    visual_review = json.loads(visual_review_path.read_text(encoding="utf-8")) if visual_review_path.exists() else {}
    provenance_path = PROCESSED / "provenance" / "source_verification.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    non_nested = sum(int(row.get("non_nested_pairs") or 0) for row in ring_rows)
    if non_nested:
        warnings.append(f"{non_nested} adjacent closed-ring pairs contain pixels outside the next larger polygon; review listed source annotations before topology-sensitive experiments.")
    if duplicate.get("cross_split_candidates"):
        warnings.append("pHash produced cross-split candidates; these are audit candidates, not automatically confirmed leaks.")
        if visual_review.get("candidates_reviewed") != duplicate.get("cross_split_candidates"):
            errors.append("pHash visual-review coverage does not match the candidate count")
    if provenance and provenance.get("status") != "ok":
        errors.append("source provenance verification failed")

    report = {
        "status": "ok" if not errors else "error",
        "manifest_images": len(manifest),
        "ring_targets": {
            "expected": len(expected_ring), "indexed_ok": len(actual_ring),
            "files": len(ring_rows) * 5, "missing_paths": ring_missing_paths,
            "dimension_errors": ring_dimension_errors, "non_nested_pairs": non_nested,
        },
        "defect_coco": {
            "files": len(coco_files), "images": coco_images, "annotations": coco_annotations,
            "segmented_annotations": coco_segmented, "bbox_errors": coco_bbox_errors,
            "reference_errors": coco_reference_errors, "group_split_errors": coco_group_split_errors,
            "category_contract_errors": coco_category_errors,
            "ignore_bbox_errors": coco_ignore_bbox_errors,
            "ignore_regions": coco_ignore_regions,
            "dataset_counts": dict(sorted(coco_dataset_counts.items())),
            "category_counts": dict(sorted(coco_category_counts.items())),
        },
        "classification_crops": {
            "index_rows": len(crop_rows), "files": actual_crop_files, "missing": crop_missing,
            "inheritance_errors": crop_inheritance_errors,
            "class_counts": dict(sorted(Counter(row["category"] for row in crop_rows).items())),
            "reused_external_files": len(crop_rows) - project_crop_rows,
            "materialized_local_files": actual_crop_files,
            "expected_normal_crops": expected_normal_crops,
        },
        "bridge_normalization": {
            "files": bridge.get("files", 0), "input_shapes": bridge.get("input_shapes", 0),
            "output_shapes": bridge.get("output_shapes", 0), "actions": bridge.get("actions", {}),
        },
        "duplicate_audit": duplicate,
        "duplicate_visual_review": visual_review,
        "source_provenance": {
            "status": provenance.get("status", "not_run"),
            "source_unchanged": provenance.get("source_unchanged"),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> int:
    report = validate()
    destination = PROCESSED / "integrity_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
