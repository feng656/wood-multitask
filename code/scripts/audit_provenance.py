from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PROCESSED = PROJECT_ROOT / "data_processed"
PROVENANCE = PROCESSED / "provenance"
MANIFEST = PROCESSED / "manifests" / "manifest.csv"
PATH_FIELDS = (
    "image_path",
    "ring_annotation_path",
    "defect_annotation_path",
    "defect_mask_path",
    "pith_annotation_path",
    "ignore_mask_path",
    "auxiliary_annotation_path",
)
ANNOTATION_FIELDS = PATH_FIELDS[1:]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_path(value: str) -> tuple[str, Path] | None:
    if not value:
        return None
    if value.startswith("zip://"):
        value = value[len("zip://") :].split("::", 1)[0]
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        return relative, resolved
    return path.as_posix(), (WORKSPACE_ROOT / path).resolve()


def unique_references(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...]
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in rows:
        for field in fields:
            reference = referenced_path(row.get(field, ""))
            if reference is not None:
                result[reference[0]] = reference[1]
    return result


def metadata_fingerprint(rows: list[dict[str, str]]) -> dict[str, object]:
    by_dataset: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_id"]].append(row)
    datasets: dict[str, object] = {}
    overall = hashlib.sha256()
    total_files = 0
    total_bytes = 0
    missing: list[str] = []
    for dataset_id in sorted(by_dataset):
        references = unique_references(by_dataset[dataset_id], PATH_FIELDS)
        digest = hashlib.sha256()
        dataset_bytes = 0
        dataset_missing: list[str] = []
        for relative, path in sorted(references.items()):
            if not path.exists():
                dataset_missing.append(relative)
                missing.append(relative)
                line = f"{relative}\tMISSING\n".encode("utf-8")
            else:
                stat = path.stat()
                dataset_bytes += stat.st_size
                line = f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
            digest.update(line)
            overall.update(dataset_id.encode("utf-8") + b"\t" + line)
        datasets[dataset_id] = {
            "files": len(references),
            "bytes": dataset_bytes,
            "metadata_sha256": digest.hexdigest(),
            "missing": dataset_missing,
        }
        total_files += len(references)
        total_bytes += dataset_bytes
    return {
        "file_count": total_files,
        "total_bytes": total_bytes,
        "metadata_sha256": overall.hexdigest(),
        "datasets": datasets,
        "missing": missing,
    }


def annotation_hashes(rows: list[dict[str, str]]) -> dict[str, object]:
    by_dataset: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["dataset_id"] != "vsb":
            by_dataset[row["dataset_id"]].append(row)
    result: dict[str, object] = {}
    for dataset_id in sorted(by_dataset):
        references = unique_references(by_dataset[dataset_id], ANNOTATION_FIELDS)
        aggregate = hashlib.sha256()
        total_bytes = 0
        missing: list[str] = []
        for relative, path in sorted(references.items()):
            if not path.exists():
                missing.append(relative)
                file_hash = "MISSING"
                size = 0
            else:
                size = path.stat().st_size
                total_bytes += size
                file_hash = sha256_file(path)
            aggregate.update(f"{relative}\t{size}\t{file_hash}\n".encode("utf-8"))
        result[dataset_id] = {
            "files": len(references),
            "bytes": total_bytes,
            "aggregate_content_sha256": aggregate.hexdigest(),
            "missing": missing,
        }

    vsb_summary_path = WORKSPACE_ROOT / "vsb-final" / "metadata" / "selection_summary.json"
    vsb_summary = json.loads(vsb_summary_path.read_text(encoding="utf-8"))
    expected = vsb_summary.get("critical_sha256", {})
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative, expected_hash in sorted(expected.items()):
        path = WORKSPACE_ROOT / "vsb-final" / relative
        actual_hash = sha256_file(path) if path.exists() else "MISSING"
        actual[relative] = actual_hash
        if actual_hash != expected_hash:
            mismatches.append(relative)
    result["vsb"] = {
        "files": len(actual),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "mismatches": mismatches,
        "source_unchanged_flag": bool(vsb_summary.get("source", {}).get("source_unchanged")),
    }
    return result


def build_registry(rows: list[dict[str, str]]) -> dict[str, object]:
    registry = json.loads(
        (PROJECT_ROOT / "configs" / "dataset_registry.json").read_text(encoding="utf-8")
    )
    data_config = json.loads(
        (PROJECT_ROOT / "configs" / "data.json").read_text(encoding="utf-8")
    )
    by_dataset: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_id"]].append(row)
    for dataset_id, metadata in registry["datasets"].items():
        selected = by_dataset.get(dataset_id, [])
        metadata["configured_path"] = data_config.get("datasets", {}).get(dataset_id, "")
        metadata["manifest_images"] = len(selected)
        metadata["physical_groups"] = len({row["group_id"] for row in selected})
        metadata["splits"] = dict(sorted(Counter(row["split"] for row in selected).items()))
        metadata["ring_images"] = sum(row["has_ring"] == "1" for row in selected)
        metadata["defect_labeled_images"] = sum(
            row["defect_label_state"] != "missing" for row in selected
        )
        metadata["target_instances_manifest"] = sum(
            int(row.get("defect_instance_count") or 0) for row in selected
        )
        metadata["ignored_instances_manifest"] = sum(
            int(row.get("ignored_defect_instance_count") or 0) for row in selected
        )
    return registry


def write_registry(registry: dict[str, object]) -> None:
    (PROVENANCE / "dataset_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "dataset_id", "title", "status", "protocol_role", "unit_of_independence", "configured_path",
        "manifest_images", "physical_groups", "splits", "ring_images",
        "defect_labeled_images", "target_instances_manifest",
        "ignored_instances_manifest", "license_status",
    ]
    with (PROVENANCE / "dataset_registry.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_id, metadata in registry["datasets"].items():
            writer.writerow({
                field: (
                    dataset_id if field == "dataset_id" else
                    json.dumps(metadata.get(field, ""), ensure_ascii=False, sort_keys=True)
                    if field == "splits" else metadata.get(field, "")
                )
                for field in fields
            })


def snapshot(rows: list[dict[str, str]]) -> dict[str, object]:
    document_snapshot = PROVENANCE / "多任务项目数据.source-snapshot.docx"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(MANIFEST),
        "source_document_sha256": sha256_file(document_snapshot),
        "source_metadata": metadata_fingerprint(rows),
        "critical_annotations": annotation_hashes(rows),
    }


def compare(baseline: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    before = baseline["source_metadata"]
    after = current["source_metadata"]
    changed_datasets: list[str] = []
    for dataset_id, stats in before["datasets"].items():
        current_stats = after["datasets"].get(dataset_id, {})
        if stats.get("metadata_sha256") != current_stats.get("metadata_sha256"):
            changed_datasets.append(dataset_id)
    annotation_changes: list[str] = []
    for dataset_id, stats in baseline["critical_annotations"].items():
        current_stats = current["critical_annotations"].get(dataset_id, {})
        if dataset_id == "vsb":
            if current_stats.get("mismatches"):
                annotation_changes.append(dataset_id)
        elif stats.get("aggregate_content_sha256") != current_stats.get("aggregate_content_sha256"):
            annotation_changes.append(dataset_id)
    unchanged = (
        before.get("metadata_sha256") == after.get("metadata_sha256")
        and not changed_datasets
        and not annotation_changes
        and not after.get("missing")
    )
    return {
        "status": "ok" if unchanged else "error",
        "source_unchanged": unchanged,
        "changed_datasets": changed_datasets,
        "annotation_changes": annotation_changes,
        "missing_files": after.get("missing", []),
        "baseline_metadata_sha256": before.get("metadata_sha256"),
        "current_metadata_sha256": after.get("metadata_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify read-only source provenance.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    PROVENANCE.mkdir(parents=True, exist_ok=True)
    rows = read_rows(MANIFEST)
    registry = build_registry(rows)
    write_registry(registry)
    current = snapshot(rows)
    if args.baseline:
        destination = PROVENANCE / "source_baseline.json"
        destination.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": "ok",
            "mode": "baseline",
            "files": current["source_metadata"]["file_count"],
            "metadata_sha256": current["source_metadata"]["metadata_sha256"],
        }, ensure_ascii=False, indent=2))
        return 0

    baseline_path = PROVENANCE / "source_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = compare(baseline, current)
    result["current_snapshot"] = current
    (PROVENANCE / "source_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "current_snapshot"}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
