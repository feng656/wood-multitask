from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .model import DataRecord
from .quality import build_quality_report, report_markdown
from .scanners import ScanContext, enrich_image_metadata, scan_all
from .splits import assign_primary_splits


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _metadata_cache(manifest_path: Path) -> dict[str, tuple[int, int, int]]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["image_path"]: (
                int(row.get("image_width") or 0),
                int(row.get("image_height") or 0),
                int(row.get("image_bytes") or 0),
            )
            for row in csv.DictReader(handle)
            if row.get("image_path")
        }


def build(config_path: Path, refresh_image_metadata: bool = False) -> int:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    project_root = config_path.parent.parent
    taxonomy_path = _resolve(project_root, config["defect_taxonomy"])
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        config["_defect_taxonomy"] = json.load(handle)
    workspace_root = _resolve(project_root, config["workspace_root"])
    dataset_root = _resolve(project_root, config["dataset_root"])
    output_dir = _resolve(project_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    context = ScanContext(workspace_root, dataset_root)
    records = scan_all(context, config)
    cache = {} if refresh_image_metadata else _metadata_cache(manifest_path)
    enrich_image_metadata(records, workspace_root, cache)
    assign_primary_splits(records, config["splits"], int(config["seed"]))
    records.sort(key=lambda item: (item.dataset_id, item.sample_id))

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DataRecord.field_names())
        writer.writeheader()
        writer.writerows(record.to_csv_row() for record in records)

    group_rows: dict[str, dict[str, object]] = {}
    for record in records:
        current = group_rows.setdefault(
            record.group_id,
            {
                "dataset_id": record.dataset_id,
                "group_id": record.group_id,
                "split": record.split,
                "ring_r2_fold": record.ring_r2_fold,
                "sample_count": 0,
            },
        )
        current["sample_count"] = int(current["sample_count"]) + 1
    with (output_dir / "group_splits.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        field_names = ["dataset_id", "group_id", "split", "ring_r2_fold", "sample_count"]
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(
            sorted(
                group_rows.values(),
                key=lambda row: (str(row["dataset_id"]), str(row["group_id"])),
            )
        )

    report = build_quality_report(records, workspace_root)
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "quality_report.md").write_text(
        report_markdown(report), encoding="utf-8"
    )

    print(f"manifest: {manifest_path}")
    print(f"records: {len(records)}")
    print(f"quality: {report['status']}")
    return 0 if report["status"] == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified wood dataset manifest")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[2] / "configs" / "data.json",
    )
    parser.add_argument(
        "--refresh-image-metadata",
        action="store_true",
        help="Open every source image header instead of reusing matching path/size metadata.",
    )
    args = parser.parse_args()
    return build(args.config, refresh_image_metadata=args.refresh_image_metadata)


if __name__ == "__main__":
    raise SystemExit(main())
