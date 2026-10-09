from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wood_data.bridge_normalize import normalize_bridge_file
from wood_data.classification import generate_classification_crops
from wood_data.defects import (
    attach_vsb_segmentations,
    coco_document,
    decode_vsb_rgb,
    parse_bridge_polygons_and_ignores,
    parse_oulu_boxes_and_ignores,
    parse_vn_boxes,
    parse_vsb_boxes,
    process_vsb_task,
    save_coco,
)
from wood_data.phash import audit_cross_split_hashes
from wood_data.ring_targets import (
    IndianaAnnotations,
    indiana_targets,
    labelme_closed_targets,
    mokume_targets,
    save_target,
    write_target_index,
)


PROCESSED = PROJECT_ROOT / "data_processed"
MANIFEST = PROCESSED / "manifests" / "manifest.csv"


def load_records() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config() -> tuple[dict[str, object], dict[str, object]]:
    config = json.loads((PROJECT_ROOT / "configs" / "data.json").read_text(encoding="utf-8"))
    taxonomy = json.loads((PROJECT_ROOT / "configs" / "defect_taxonomy.json").read_text(encoding="utf-8"))
    return config, taxonomy


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def aligned_ring_image(record: dict[str, str], target_size: tuple[int, int]) -> str:
    source_size = (int(record["image_width"]), int(record["image_height"]))
    if source_size == target_size:
        return record["image_path"]
    destination = PROCESSED / "ring_images" / record["dataset_id"] / f"{safe_name(record['sample_id'])}.jpg"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(WORKSPACE_ROOT / record["image_path"]) as image:
            resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            resized.save(destination, quality=95, subsampling=0)
    return destination.relative_to(WORKSPACE_ROOT).as_posix()


def run_rings(records: list[dict[str, str]], boundary_width: int, distance_tau: float) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    indiana = IndianaAnnotations()
    previous: dict[str, dict[str, str]] = {}
    previous_index = PROCESSED / "ring_targets" / "ring_targets.csv"
    if previous_index.exists():
        with previous_index.open("r", encoding="utf-8-sig", newline="") as handle:
            previous = {row["sample_id"]: row for row in csv.DictReader(handle)}
    for record in records:
        if record["has_ring"] != "1":
            continue
        base = {
            "sample_id": record["sample_id"], "dataset_id": record["dataset_id"],
            "split": record["split"], "source_annotation": record["ring_annotation_path"],
            "width": record["image_width"], "height": record["image_height"],
        }
        old = previous.get(record["sample_id"])
        if old and old.get("status") == "ok" and all(
            old.get(field) and (PROJECT_ROOT / old[field]).exists()
            for field in ("skeleton_path", "boundary_path", "distance_path", "instance_path", "valid_path")
        ):
            with Image.open(PROJECT_ROOT / old["skeleton_path"]) as target_image:
                target_size = target_image.size
            old.update({
                "model_image_path": aligned_ring_image(record, target_size),
                "target_width": target_size[0], "target_height": target_size[1],
            })
            rows.append(old)
            continue
        try:
            annotation = WORKSPACE_ROOT / record["ring_annotation_path"]
            image_shape = (int(record["image_height"]), int(record["image_width"]))
            if record["dataset_id"] == "mokume":
                target = mokume_targets(annotation, boundary_width, distance_tau)
            elif record["dataset_id"] == "indiana":
                ignore = WORKSPACE_ROOT / record["ignore_mask_path"] if record["ignore_mask_path"] else None
                target = indiana_targets(
                    annotation, Path(record["image_path"]).name, image_shape, ignore, indiana,
                    boundary_width, distance_tau,
                )
            else:
                target = labelme_closed_targets(annotation, image_shape, boundary_width, distance_tau)
            paths = save_target(PROCESSED, record["dataset_id"], safe_name(record["sample_id"]), target)
            paths = {key: Path(value).relative_to(PROJECT_ROOT).as_posix() for key, value in paths.items()}
            target_size = (target.skeleton.shape[1], target.skeleton.shape[0])
            rows.append({
                **base, **paths, "model_image_path": aligned_ring_image(record, target_size),
                "target_width": target_size[0], "target_height": target_size[1],
                "skeleton_pixels": int(target.skeleton.sum()),
                "valid_pixels": int(target.valid.sum()), "closed_ring_count": target.closed_ring_count,
                "instance_count": int(target.instance.max()), "non_nested_pairs": target.non_nested_pairs,
                "status": "ok", "error": "",
            })
        except Exception as error:
            rows.append({**base, "status": "error", "error": f"{type(error).__name__}: {error}"})
    index_path = PROCESSED / "ring_targets" / "ring_targets.csv"
    write_target_index(index_path, rows)
    summary = {
        "requested": len(rows), "generated": sum(row["status"] == "ok" for row in rows),
        "errors": sum(row["status"] != "ok" for row in rows),
        "closed_rings": sum(int(row.get("closed_ring_count", 0) or 0) for row in rows),
        "non_nested_pairs": sum(int(row.get("non_nested_pairs", 0) or 0) for row in rows),
        "by_dataset": dict(Counter(row["dataset_id"] for row in rows if row["status"] == "ok")),
        "aligned_derived_images": sum(
            row.get("model_image_path", "").startswith("wood-multitask-final/") for row in rows
        ),
        "boundary_width": boundary_width, "distance_tau": distance_tau,
        "index": index_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    (PROCESSED / "ring_targets" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_bridge_normalization(records: list[dict[str, str]]) -> dict[str, object]:
    output_root = PROCESSED / "bridge_annotations"
    audits: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        source_reference = record["defect_annotation_path"]
        if record["dataset_role"] != "bridge" or not source_reference or source_reference in seen:
            continue
        seen.add(source_reference)
        source = WORKSPACE_ROOT / source_reference
        destination = output_root / record["dataset_id"] / f"{safe_name(record['sample_id'])}.json"
        audit = normalize_bridge_file(
            source, destination, int(record["image_width"]), int(record["image_height"])
        )
        audit["sample_id"] = record["sample_id"]
        audit["source_relative"] = source_reference
        audit["normalized_relative"] = destination.relative_to(PROJECT_ROOT).as_posix()
        audits.append(audit)
    actions = Counter(change["action"] for audit in audits for change in audit["changes"])
    summary = {
        "files": len(audits), "input_shapes": sum(int(item["input_shapes"]) for item in audits),
        "output_shapes": sum(int(item["output_shapes"]) for item in audits),
        "primary_instances": sum(int(item["primary_instances"]) for item in audits),
        "ignored_instances": sum(int(item["ignored_instances"]) for item in audits),
        "actions": dict(sorted(actions.items())), "audits": audits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "normalization_audit.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def normalized_bridge_map() -> dict[str, Path]:
    path = PROCESSED / "bridge_annotations" / "normalization_audit.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["source_relative"]: PROJECT_ROOT / item["normalized_relative"] for item in payload["audits"]}


def run_coco(
    records: list[dict[str, str]], taxonomy: dict[str, object], vsb_segmentation_limit: int
) -> dict[str, object]:
    by_split_images: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    by_split_annotations: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    image_counters: Counter[str] = Counter()
    annotation_counters: Counter[str] = Counter()
    normalized = normalized_bridge_map()
    segmented_vsb = 0
    unmatched_components = 0
    unknown_color_pixels = 0
    ignored_regions = 0
    defect_mappings = taxonomy["datasets"]

    # VSB is already fully materialized and audited. Merge its COCO files while
    # preserving the fixed split and referencing files in place.
    vsb_root = WORKSPACE_ROOT / "vsb-final"
    for split in ("train", "val", "test"):
        source_coco = vsb_root / "annotations" / "coco" / f"instances_{split}.json"
        payload = json.loads(source_coco.read_text(encoding="utf-8"))
        id_map: dict[int, int] = {}
        for source_image in payload["images"]:
            image_counters[split] += 1
            image_id = image_counters[split]
            id_map[int(source_image["id"])] = image_id
            image_entry = dict(source_image)
            image_entry["id"] = image_id
            image_entry["file_name"] = f"vsb-final/{source_image['file_name']}"
            if image_entry.get("unified_semantic_mask"):
                image_entry["unified_semantic_mask"] = f"vsb-final/{image_entry['unified_semantic_mask']}"
            image_entry["source_coco"] = source_coco.relative_to(WORKSPACE_ROOT).as_posix()
            by_split_images[split].append(image_entry)
        for source_annotation in payload["annotations"]:
            annotation_counters[split] += 1
            annotation = dict(source_annotation)
            annotation["id"] = annotation_counters[split]
            annotation["image_id"] = id_map[int(source_annotation["image_id"])]
            by_split_annotations[split].append(annotation)
        segmented_vsb += len(payload["images"])

    for record in records:
        if record["dataset_id"] == "vsb" or record["defect_label_state"] == "missing":
            continue
        split = record["split"]
        image_counters[split] += 1
        image_id = image_counters[split]
        image_entry = {
            "id": image_id, "file_name": record["image_path"],
            "width": int(record["image_width"]), "height": int(record["image_height"]),
            "source_dataset": record["dataset_id"], "group_id": record["group_id"],
            "sample_id": record["sample_id"], "split": split,
            "defect_label_state": record["defect_label_state"],
            "ignored_instance_count": int(record.get("ignored_defect_instance_count") or 0),
        }
        if record["defect_mask_path"]:
            image_entry["source_semantic_mask"] = record["defect_mask_path"]
        by_split_images[split].append(image_entry)

        path = WORKSPACE_ROOT / record["defect_annotation_path"] if record["defect_annotation_path"] else None
        if record["dataset_id"] == "vnwoodknot" and path:
            mapped = defect_mappings["vnwoodknot"][record["raw_defect_classes"]]
            annotations = parse_vn_boxes(path, int(record["image_width"]), int(record["image_height"]), mapped)
        elif record["dataset_id"] == "oulu" and path:
            annotations, ignored = parse_oulu_boxes_and_ignores(path, defect_mappings["oulu"])
            if ignored:
                image_entry["ignore_bboxes"] = ignored
                ignored_regions += len(ignored)
        elif record["dataset_role"] == "bridge" and path:
            annotations, ignored = parse_bridge_polygons_and_ignores(
                normalized.get(record["defect_annotation_path"], path)
            )
            if ignored:
                image_entry["ignore_bboxes"] = ignored
                ignored_regions += len(ignored)
        else:
            annotations = []

        for annotation in annotations:
            annotation_counters[split] += 1
            annotation.update({
                "id": annotation_counters[split], "image_id": image_id,
                "source_dataset": record["dataset_id"], "group_id": record["group_id"],
            })
            by_split_annotations[split].append(annotation)

    output_root = PROCESSED / "defect_coco"
    files: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in sorted(by_split_images):
        document = coco_document(by_split_images[split], by_split_annotations[split])
        destination = output_root / f"instances_{split}.json"
        save_coco(destination, document)
        files[split] = destination.relative_to(PROJECT_ROOT).as_posix()
        counts[split] = {"images": len(by_split_images[split]), "annotations": len(by_split_annotations[split])}

    mask_contract = output_root / "vsb_semantic_mask_contract.csv"
    with mask_contract.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "split", "source_semantic_mask", "decoder", "unified_category_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record["dataset_id"] == "vsb":
                writer.writerow({
                    "sample_id": record["sample_id"], "split": record["split"],
                    "source_semantic_mask": record["defect_mask_path"],
                    "decoder": "prebuilt_vsb_final_two_class_id_mask",
                    "unified_category_ids": "0=background;1=knot;2=crack",
                })
    summary = {
        "files": files, "counts": counts, "vsb_masks_materialized": segmented_vsb,
        "vsb_unmatched_box_components": unmatched_components,
        "vsb_unknown_color_pixels": unknown_color_pixels,
        "ignored_non_target_regions": ignored_regions,
        "vsb_mask_contract_rows": sum(record["dataset_id"] == "vsb" for record in records),
        "vsb_mask_strategy": "Reuse lossless two-class masks from immutable vsb-final; no duplicate image or mask materialization.",
        "vsb_segmentation_limit_argument": vsb_segmentation_limit,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_coco_documents() -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((PROCESSED / "defect_coco").glob("instances_*.json"))]


def run_crops() -> dict[str, object]:
    vsb_rows: list[dict[str, object]] = []
    source_index = WORKSPACE_ROOT / "vsb-final" / "manifests" / "classification_crops.csv"
    with source_index.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            vsb_rows.append({
                "crop_path": f"vsb-final/{row['crop_path']}",
                "source_image": f"vsb-final/{row['source_image']}",
                "source_dataset": row["source_dataset"],
                "group_id": row["group_id"],
                "split": row["split"],
                "category": row["category"],
                "subtype": row.get("subtype", ""),
                "attributes": row.get("attributes", ""),
                "source_class": row.get("source_class", ""),
                "annotation_geometry": "bbox+semantic_instance_mask",
                "bbox": row.get("bbox", ""),
            })
    summary = generate_classification_crops(
        load_coco_documents(), WORKSPACE_ROOT, PROCESSED / "classification_crops",
        PROCESSED / "classification_crops" / "classification_crops.csv",
        exclude_datasets={"vsb"}, initial_rows=vsb_rows,
    )
    summary["reused_vsb_crops"] = len(vsb_rows)
    (PROCESSED / "classification_crops" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_duplicates(records: list[dict[str, str]], threshold: int) -> dict[str, object]:
    vsb_hashes: dict[str, int] = {}
    with (WORKSPACE_ROOT / "vsb-final" / "manifests" / "perceptual_hashes.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("selected") == "1" and row.get("phash_hex"):
                vsb_hashes[row["sample_id"]] = int(row["phash_hex"], 16)

    primary = [record for record in records if record["supervision_scope"] == "primary"]
    summary = audit_cross_split_hashes(
        primary,
        WORKSPACE_ROOT,
        PROCESSED / "duplicate_audit" / "primary" / "cross_split_phash_candidates.csv",
        threshold,
        precomputed_hashes=vsb_hashes,
    )
    summary["candidate_csv"] = Path(summary["candidate_csv"]).relative_to(PROJECT_ROOT).as_posix()
    summary["hash_index"] = Path(summary["hash_index"]).relative_to(PROJECT_ROOT).as_posix()
    summary["scope"] = "primary supervision only"
    summary["excluded_auxiliary_images"] = len(records) - len(primary)

    wvtec = [record for record in records if record["dataset_id"] == "wvtec_wood"]
    wvtec_summary = audit_cross_split_hashes(
        wvtec,
        WORKSPACE_ROOT,
        PROCESSED / "duplicate_audit" / "wvtec_auxiliary" / "cross_split_phash_candidates.csv",
        threshold,
    )
    for field in ("candidate_csv", "hash_index"):
        wvtec_summary[field] = Path(wvtec_summary[field]).relative_to(PROJECT_ROOT).as_posix()
    summary["auxiliary_audits"] = {"wvtec_wood": wvtec_summary}
    coating_groups = Counter(
        record["group_id"] for record in records if record["dataset_id"] == "coating"
    )
    summary["coating_group_audit"] = {
        "images": sum(coating_groups.values()),
        "physical_groups": len(coating_groups),
        "variants_per_group": dict(sorted(Counter(coating_groups.values()).items())),
        "cross_split_audit": "not applicable; every derivative is auxiliary_stress and excluded from primary training",
    }
    (PROCESSED / "duplicate_audit" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable annual-ring and wood-defect training artifacts")
    parser.add_argument("--stages", nargs="+", choices=["bridge", "rings", "coco", "crops", "duplicates"], default=["bridge", "rings", "coco", "crops", "duplicates"])
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument("--distance-tau", type=float, default=16.0)
    parser.add_argument("--vsb-segmentation-limit", type=int, default=64, help="0 skips materialization, -1 processes all VSB masks")
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()
    records = load_records()
    _, taxonomy = load_config()
    summaries: dict[str, object] = {}
    if "bridge" in args.stages:
        summaries["bridge"] = run_bridge_normalization(records)
    if "rings" in args.stages:
        summaries["rings"] = run_rings(records, args.boundary_width, args.distance_tau)
    if "coco" in args.stages:
        summaries["coco"] = run_coco(records, taxonomy, args.vsb_segmentation_limit)
    if "crops" in args.stages:
        summaries["crops"] = run_crops()
    if "duplicates" in args.stages:
        summaries["duplicates"] = run_duplicates(records, args.phash_threshold)
    print(json.dumps({name: {key: value for key, value in summary.items() if key != "audits"} for name, summary in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
