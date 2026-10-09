from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from .model import DataRecord
from .splits import split_leaks


def _path_exists(workspace_root: Path, value: str) -> bool:
    if not value:
        return True
    if value.startswith("zip://"):
        archive = value[len("zip://") :].split("::", 1)[0]
        return (workspace_root / archive).exists()
    return (workspace_root / value).exists()


def build_quality_report(records: list[DataRecord], workspace_root: Path) -> dict[str, object]:
    duplicates = [
        path for path, count in Counter(r.image_path for r in records).items() if count > 1
    ]
    missing_files: list[dict[str, str]] = []
    path_fields = (
        "image_path",
        "ring_annotation_path",
        "defect_annotation_path",
        "defect_mask_path",
        "pith_annotation_path",
        "ignore_mask_path",
        "auxiliary_annotation_path",
    )
    for record in records:
        for field in path_fields:
            value = getattr(record, field)
            if value and not _path_exists(workspace_root, value):
                missing_files.append({"sample_id": record.sample_id, "field": field, "path": value})

    invalid_images = [
        {
            "sample_id": record.sample_id,
            "image_path": record.image_path,
            "width": record.image_width,
            "height": record.image_height,
            "bytes": record.image_bytes,
        }
        for record in records
        if record.image_width <= 0 or record.image_height <= 0 or record.image_bytes <= 0
    ]
    inventory = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.image_path):
        inventory.update(
            f"{record.image_path}\t{record.image_width}\t{record.image_height}\t{record.image_bytes}\n".encode("utf-8")
        )

    by_dataset: dict[str, dict[str, object]] = {}
    for dataset_id in sorted({r.dataset_id for r in records}):
        selected = [r for r in records if r.dataset_id == dataset_id]
        by_dataset[dataset_id] = {
            "images": len(selected),
            "groups": len({r.group_id for r in selected}),
            "splits": dict(sorted(Counter(r.split for r in selected).items())),
            "ring_labels": sum(r.has_ring for r in selected),
            "pith_labels": sum(r.has_pith for r in selected),
            "defect_box_labels": sum(r.has_defect_box for r in selected),
            "defect_mask_labels": sum(r.has_defect_mask for r in selected),
            "defect_class_labels": sum(r.has_defect_class for r in selected),
            "verified_negative": sum(r.defect_label_state == "verified_negative" for r in selected),
            "ignored_defect_instances": sum(r.ignored_defect_instance_count for r in selected),
            "missing_ring_labels": sum(r.ring_label_state == "missing" for r in selected),
            "missing_defect_labels": sum(r.defect_label_state == "missing" for r in selected),
            "supervision_scopes": dict(sorted(Counter(r.supervision_scope for r in selected).items())),
        }

    fold_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.dataset_id == "urudendro4":
            fold_groups[record.ring_r2_fold].add(record.group_id)

    warnings: list[str] = []
    if not any(r.dataset_role == "bridge" for r in records):
        warnings.append("No dual-label bridge set exists yet; unlabeled tasks must remain masked during training.")
    missing_main_ring = [
        r.sample_id
        for r in records
        if r.dataset_role in {"ring_main", "ring_multiface"} and not r.has_ring
    ]
    if missing_main_ring:
        preview = ", ".join(missing_main_ring[:5])
        warnings.append(
            f"Main ring data contains {len(missing_main_ring)} source images without ring labels: {preview}."
        )
    vsb_groups = len({r.group_id for r in records if r.dataset_id == "vsb"})
    if vsb_groups < 20:
        warnings.append(
            f"VSB has only {vsb_groups} conservative acquisition groups. The fixed vsb-final split prevents frame leakage but does not create additional independent wood sources."
        )
    warnings.append(
        "VNWoodKnot numeric class mapping is centralized in configs/defect_taxonomy.json and must be confirmed against the dataset publication before final experiments."
    )

    return {
        "status": "ok" if not duplicates and not missing_files and not invalid_images and not split_leaks(records) else "error",
        "total_images": len(records),
        "inventory_sha256": inventory.hexdigest(),
        "datasets": by_dataset,
        "duplicate_image_paths": duplicates,
        "missing_files": missing_files,
        "invalid_images": invalid_images,
        "split_leaks": split_leaks(records),
        "ring_r2_fold_groups": {key: len(value) for key, value in sorted(fold_groups.items())},
        "warnings": warnings,
    }


def report_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Data quality report",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Indexed images: **{report['total_images']}**",
        "",
        f"Inventory SHA-256: `{report['inventory_sha256']}`",
        "",
        "| Dataset | Images | Groups | Split counts | Ring | Pith | Defect boxes | Defect masks | Defect classes | Ignored instances |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, stats in report["datasets"].items():
        split_text = ", ".join(f"{k}={v}" for k, v in stats["splits"].items())
        lines.append(
            f"| {dataset_id} | {stats['images']} | {stats['groups']} | {split_text} | "
            f"{stats['ring_labels']} | {stats['pith_labels']} | {stats['defect_box_labels']} | "
            f"{stats['defect_mask_labels']} | {stats['defect_class_labels']} | {stats['ignored_defect_instances']} |"
        )
    lines.extend(["", "## Integrity checks", ""])
    lines.append(f"- Duplicate image paths: {len(report['duplicate_image_paths'])}")
    lines.append(f"- Missing referenced files: {len(report['missing_files'])}")
    lines.append(f"- Invalid or unreadable images: {len(report['invalid_images'])}")
    lines.append(f"- Cross-split physical-group leaks: {len(report['split_leaks'])}")
    lines.append(
        "- UruDendro4 R2 fold tree counts: "
        + ", ".join(f"fold {k}={v}" for k, v in report["ring_r2_fold_groups"].items())
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)
