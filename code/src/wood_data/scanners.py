from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from .labelme import (
    bridge_ignored_count,
    bridge_summary,
    load_labelme_without_image_data,
    mokume_bridge_annotation,
    mokume_bridge_is_complete,
)
from .model import DataRecord


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}


class ScanContext:
    def __init__(self, workspace_root: Path, dataset_root: Path):
        self.workspace_root = workspace_root.resolve()
        self.dataset_root = dataset_root.resolve()

    def rel(self, path: Path | None) -> str:
        if path is None:
            return ""
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def root(self, relative_path: str) -> Path:
        return self.dataset_root / Path(relative_path)


def _read_key_set(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row[key].strip() for row in csv.DictReader(handle) if row.get(key)}


def _exact_annotation(directory: Path, stem: str, extension: str = ".json") -> Path | None:
    candidate = directory / f"{stem}{extension}"
    return candidate if candidate.exists() else None


def _json_has_point_label(path: Path | None, label: str) -> bool:
    if path is None or not path.exists():
        return False
    try:
        payload = load_labelme_without_image_data(path)
    except (OSError, UnicodeError, ValueError):
        return False
    return any(
        shape.get("shape_type") == "point" and shape.get("label") == label
        for shape in payload.get("shapes", [])
    )


def scan_urudendro(ctx: ScanContext, root: Path) -> list[DataRecord]:
    images = root / "images" / "images"
    annotations = root / "ring_annotations" / "ring_annotations"
    defect_annotations = root / "annotations" / "defects"
    pith_path = root / "pith_location.csv"
    pith_ids = _read_key_set(pith_path, "Image")
    records: list[DataRecord] = []
    for image in sorted(images.glob("*.png")):
        match = re.match(r"^([A-Za-z]+\d+)", image.stem)
        physical_id = match.group(1) if match else image.stem
        annotation = _exact_annotation(annotations, image.stem)
        defect_annotation = _exact_annotation(defect_annotations, image.stem)
        defect_count, category, subtype, attributes = bridge_summary(defect_annotation)
        ignored_count = bridge_ignored_count(defect_annotation)
        records.append(
            DataRecord(
                image_path=ctx.rel(image),
                dataset_id="urudendro",
                sample_id=f"urudendro:{image.stem}",
                group_id=f"urudendro:{physical_id}",
                tree_id=physical_id,
                surface_type="transverse",
                has_ring=annotation is not None,
                has_pith=image.stem in pith_ids,
                ring_annotation_path=ctx.rel(annotation),
                pith_annotation_path=ctx.rel(pith_path) if image.stem in pith_ids else "",
                defect_annotation_path=ctx.rel(defect_annotation),
                has_defect_box=defect_annotation is not None,
                has_defect_class=defect_annotation is not None,
                dataset_role="bridge" if defect_annotation is not None else "ring_main",
                ring_label_state="complete_closed" if annotation else "missing",
                defect_label_state=(
                    "positive" if defect_count else ("ignore_only" if ignored_count else "verified_negative")
                ) if defect_annotation is not None else "missing",
                defect_instance_count=defect_count,
                ignored_defect_instance_count=ignored_count,
                raw_defect_classes=category,
                unified_category=category if defect_count else ("normal" if defect_annotation is not None else ""),
                defect_subtype=subtype,
                attributes=attributes,
                annotation_geometry="polygon",
                notes=(
                    "Exact JSON selected; annotator-specific *-M/*-S/*-V files retained as source metadata only. "
                    "Bridge defects use LabelMe polygons in annotations/defects."
                ),
            )
        )
    return records


def scan_urudendro2(ctx: ScanContext, root: Path) -> list[DataRecord]:
    images = root / "images"
    annotations = root / "annotations" / "annual_rings"
    pith_path = root / "pith_location.txt"
    pith_ids = _read_key_set(pith_path, "Code")
    records: list[DataRecord] = []
    for image in sorted(images.glob("*.jpg")):
        physical_id = re.sub(r"(?:-2|C)$", "", image.stem)
        annotation = _exact_annotation(annotations, image.stem)
        has_pith_point = _json_has_point_label(annotation, "pith")
        has_pith = image.stem in pith_ids or has_pith_point
        records.append(
            DataRecord(
                image_path=ctx.rel(image),
                dataset_id="urudendro2",
                sample_id=f"urudendro2:{image.stem}",
                group_id=f"urudendro2:{physical_id}",
                tree_id=physical_id,
                surface_type="transverse",
                has_ring=annotation is not None,
                has_pith=has_pith,
                ring_annotation_path=ctx.rel(annotation),
                pith_annotation_path=(
                    ctx.rel(annotation)
                    if has_pith_point
                    else (ctx.rel(pith_path) if image.stem in pith_ids else "")
                ),
                dataset_role="ring_main",
                ring_label_state="complete_closed" if annotation else "missing",
                notes="Pith is read from JSON point label when present." if has_pith_point else "",
            )
        )
    return records


def scan_urudendro4(ctx: ScanContext, root: Path) -> list[DataRecord]:
    images = root / "images"
    annotations = root / "annotations" / "annual_rings"
    pith_path = root / "pith_location.txt"
    pith_ids = _read_key_set(pith_path, "Code")
    pattern = re.compile(r"^(T[^_]+)_(B[^_]+)_(N[^_]+)_(.+)$")
    records: list[DataRecord] = []
    for image in sorted(images.glob("*.png")):
        match = pattern.match(image.stem)
        if match:
            height, batch, number, view = match.groups()
            tree_id = f"{batch}_{number}"
            disc_id = f"{height}_{tree_id}"
        else:
            height = view = ""
            tree_id = disc_id = image.stem
        annotation = _exact_annotation(annotations, image.stem)
        records.append(
            DataRecord(
                image_path=ctx.rel(image),
                dataset_id="urudendro4",
                sample_id=f"urudendro4:{image.stem}",
                group_id=f"urudendro4:{tree_id}",
                tree_id=tree_id,
                face_id=view,
                surface_type="transverse",
                has_ring=annotation is not None,
                has_pith=image.stem in pith_ids,
                ring_annotation_path=ctx.rel(annotation),
                pith_annotation_path=ctx.rel(pith_path) if image.stem in pith_ids else "",
                dataset_role="ring_external_r1",
                ring_label_state="complete_closed" if annotation else "missing",
                notes=f"disc_id={disc_id}; height={height}; all heights share tree-level group_id",
            )
        )
    return records


def scan_mokume(ctx: ScanContext, root: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    for cube_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cube_id = cube_dir.name
        complete_bridge = mokume_bridge_is_complete(cube_dir)
        for face_id in "ABCDEF":
            image = cube_dir / f"{face_id}_col.png"
            if not image.exists():
                continue
            annotation = cube_dir / f"{face_id}_ann.png"
            defect_annotation = mokume_bridge_annotation(cube_dir, face_id) if complete_bridge else None
            defect_count, category, subtype, attributes = bridge_summary(defect_annotation)
            ignored_count = bridge_ignored_count(defect_annotation)
            surface = "transverse" if face_id in {"B", "D"} else "side"
            records.append(
                DataRecord(
                    image_path=ctx.rel(image),
                    dataset_id="mokume",
                    sample_id=f"mokume:{cube_id}:{face_id}",
                    group_id=f"mokume:{cube_id}",
                    cube_id=cube_id,
                    face_id=face_id,
                    surface_type=surface,
                    has_ring=annotation.exists(),
                    ring_annotation_path=ctx.rel(annotation) if annotation.exists() else "",
                    defect_annotation_path=ctx.rel(defect_annotation),
                    has_defect_box=defect_annotation is not None,
                    has_defect_class=defect_annotation is not None,
                    dataset_role="bridge" if defect_annotation is not None else "ring_multiface",
                    ring_label_state="local_1px" if annotation.exists() else "missing",
                    defect_label_state=(
                        "positive" if defect_count else ("ignore_only" if ignored_count else "verified_negative")
                    ) if defect_annotation is not None else "missing",
                    defect_instance_count=defect_count,
                    ignored_defect_instance_count=ignored_count,
                    raw_defect_classes=category,
                    unified_category=category if defect_count else ("normal" if defect_annotation is not None else ""),
                    defect_subtype=subtype,
                    attributes=attributes,
                    annotation_geometry="axis_aligned_rectangle_from_polygon",
                    notes=(
                        "Raw cube face used; derived ImagePairs are intentionally excluded before splitting. "
                        "Bridge defects use LabelMe polygons in cube-level defects."
                    ),
                )
            )
    return records


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _box_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                labels.append(parts[0])
    return labels


def _mapped_summary(raw_labels: list[str], mapping: dict[str, dict[str, object]]) -> tuple[str, str, str]:
    mapped = [mapping[label] for label in raw_labels if label in mapping]
    categories = sorted({str(item.get("category", "")) for item in mapped if item.get("category")})
    subtypes = sorted({str(item.get("subtype", "")) for item in mapped if item.get("subtype")})
    attributes = sorted(
        {
            str(attribute)
            for item in mapped
            for attribute in item.get("attributes", [])
            if attribute
        }
    )
    return ";".join(categories), ";".join(subtypes), ";".join(attributes)


def scan_vsb(
    ctx: ScanContext, root: Path, taxonomy: dict[str, dict[str, object]]
) -> list[DataRecord]:
    manifest_path = root / "manifests" / "manifest.csv"
    inventory_path = root / "manifests" / "source_inventory.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"vsb-final manifest not found: {manifest_path}")
    inventory: dict[str, dict[str, str]] = {}
    if inventory_path.exists():
        with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
            inventory = {row["sample_id"]: row for row in csv.DictReader(handle)}

    records: list[DataRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = inventory.get(row["sample_id"], {})
            excluded_count = int(source.get("excluded_instance_count") or 0)
            image = root / row["image_path"]
            annotation = root / row["defect_annotation_path"]
            mask = root / row["defect_mask_path"]
            records.append(
                DataRecord(
                    image_path=ctx.rel(image),
                    dataset_id="vsb",
                    sample_id=row["sample_id"],
                    group_id=row["group_id"],
                    split=row["split"],
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                    image_bytes=int(row["image_bytes"]),
                    surface_type="board_surface",
                    has_defect_box=True,
                    has_defect_mask=True,
                    has_defect_class=True,
                    defect_annotation_path=ctx.rel(annotation),
                    defect_mask_path=ctx.rel(mask),
                    dataset_role="defect_main",
                    defect_label_state=row["defect_label_state"],
                    defect_instance_count=int(row["defect_instance_count"]),
                    ignored_defect_instance_count=excluded_count,
                    raw_defect_classes=row["raw_defect_classes"],
                    unified_category=row["unified_category"],
                    annotation_geometry="bbox+semantic_instance_mask",
                    source_variant=row["selection_reason"],
                    base_sample_id=row["sample_id"],
                    notes=(
                        "Immutable vsb-final selection; fixed split inherited. "
                        f"source_raw_classes={source.get('raw_classes', '')}; "
                        f"excluded_source_instances={excluded_count}; phash={row.get('phash_hex', '')}"
                    ),
                )
            )
    return records


def _find_vn_root(root: Path) -> Path:
    for candidate in root.rglob("VNWoodKnot"):
        if candidate.is_dir() and all((candidate / name).is_dir() for name in ("train", "test", "validation")):
            return candidate
    raise FileNotFoundError(f"VNWoodKnot split root not found below {root}")


def scan_vnwoodknot(
    ctx: ScanContext, root: Path, class_map: dict[str, dict[str, object]]
) -> list[DataRecord]:
    data_root = _find_vn_root(root)
    records: list[DataRecord] = []
    for source_split in ("train", "validation", "test"):
        for class_dir in sorted((data_root / source_split).iterdir()):
            if not class_dir.is_dir() or class_dir.name not in class_map:
                continue
            mapped = class_map[class_dir.name]
            for image in sorted(class_dir.glob("*.jpg")):
                annotation = image.with_suffix(".txt")
                count = _line_count(annotation)
                state = "verified_negative" if class_dir.name == "0" else "positive"
                records.append(
                    DataRecord(
                        image_path=ctx.rel(image),
                        dataset_id="vnwoodknot",
                        sample_id=f"vnwoodknot:{source_split}:{class_dir.name}:{image.stem}",
                        group_id=f"vnwoodknot:{source_split}:{image.stem.lower()}",
                        surface_type="board_surface",
                        has_defect_box=annotation.exists(),
                        has_defect_class=True,
                        defect_annotation_path=ctx.rel(annotation) if annotation.exists() else "",
                        dataset_role="defect_external_then_joint",
                        source_split=source_split,
                        defect_label_state=state,
                        defect_instance_count=count,
                        raw_defect_classes=class_dir.name,
                        unified_category=str(mapped["category"]),
                        defect_subtype=str(mapped["subtype"]),
                        attributes=";".join(str(item) for item in mapped.get("attributes", [])),
                        notes=f"source_class_id={class_dir.name}; source split retained for audit",
                    )
                )
    return records


def scan_indiana(ctx: ScanContext, root: Path) -> list[DataRecord]:
    surfaces = {
        "train-data": "cleaned",
        "train-data-dry": "dry",
        "train-data-rough": "rough",
    }
    records: list[DataRecord] = []
    for directory, surface in surfaces.items():
        data_dir = root / directory
        annotation_xml = data_dir / "label" / "annotations.xml"
        annotated_names: set[str] = set()
        if annotation_xml.exists():
            tree = ET.parse(annotation_xml)
            annotated_names = {
                PurePosixPath(node.attrib.get("name", "")).name
                for node in tree.findall(".//image")
                if node.attrib.get("name")
                and any(polyline.attrib.get("label") == "Ring" for polyline in node.findall("polyline"))
            }
        for image in sorted((data_dir / "image").glob("*.jpg")):
            physical_id = image.stem.split("-", 1)[0]
            has_annotation = image.name in annotated_names
            ignore = data_dir / "ignore" / image.name
            records.append(
                DataRecord(
                    image_path=ctx.rel(image),
                    dataset_id="indiana",
                    sample_id=f"indiana:{surface}:{image.stem}",
                    group_id=f"indiana:{physical_id}",
                    tree_id=physical_id,
                    face_id=image.stem[len(physical_id) + 1 :],
                    surface_type=surface,
                    has_ring=has_annotation,
                    ring_annotation_path=ctx.rel(annotation_xml) if has_annotation else "",
                    ignore_mask_path=ctx.rel(ignore) if ignore.exists() else "",
                    dataset_role="ring_external",
                    ring_label_state="partial_with_ignore" if has_annotation else "missing",
                    notes=(
                        "CVAT XML contains at least one Ring polyline; the provided ignore mask defines valid pixels."
                        if has_annotation
                        else "No Ring polyline is present for this image; non-ring CVAT labels do not count as ring supervision."
                    ),
                )
            )
    return records


def scan_oulu(
    ctx: ScanContext, root: Path, taxonomy: dict[str, dict[str, object]]
) -> list[DataRecord]:
    images = root / "JPEGImages"
    annotations = root / "Annotations"
    records: list[DataRecord] = []
    for image in sorted(images.glob("*.jpg")):
        annotation = annotations / f"{image.stem}.xml"
        categories: list[str] = []
        if annotation.exists():
            tree = ET.parse(annotation)
            categories = [
                (node.text or "").strip()
                for node in tree.findall(".//object/name")
                if (node.text or "").strip()
            ]
        target_categories = [
            value for value in categories
            if value in taxonomy and taxonomy[value].get("train_action") == "keep"
        ]
        ignored_categories = [value for value in categories if value not in target_categories]
        category, subtype, attributes = _mapped_summary(target_categories, taxonomy)
        if target_categories:
            label_state = "positive"
        elif ignored_categories:
            label_state = "ignore_only"
        else:
            label_state = "verified_negative" if annotation.exists() else "missing"
        records.append(
            DataRecord(
                image_path=ctx.rel(image),
                dataset_id="oulu",
                sample_id=f"oulu:{image.stem}",
                group_id=f"oulu:{image.stem}",
                surface_type="board_surface",
                has_defect_box=annotation.exists(),
                has_defect_class=annotation.exists(),
                defect_annotation_path=ctx.rel(annotation) if annotation.exists() else "",
                dataset_role="defect_external",
                defect_label_state=label_state,
                defect_instance_count=len(target_categories),
                ignored_defect_instance_count=len(ignored_categories),
                raw_defect_classes=";".join(sorted(set(categories))),
                unified_category=category if target_categories else ("normal" if label_state == "verified_negative" else ""),
                defect_subtype=subtype,
                attributes=attributes,
                annotation_geometry="bbox",
                notes=(
                    "OULU small/edge labels are knot attributes. Non-knot/non-split boxes are retained as ignore regions; "
                    f"ignored_source_classes={';'.join(sorted(set(ignored_categories)))}"
                ),
            )
        )
    return records


def scan_wvtec_wood(ctx: ScanContext, root: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    for official_split in ("train", "test"):
        split_root = root / official_split
        if not split_root.exists():
            continue
        for category_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            category = category_dir.name
            for image in sorted(category_dir.glob("*.png")):
                mask = root / "ground_truth" / category / f"{image.stem}_mask.png"
                records.append(
                    DataRecord(
                        image_path=ctx.rel(image),
                        dataset_id="wvtec_wood",
                        sample_id=f"wvtec_wood:{official_split}:{category}:{image.stem}",
                        group_id=f"wvtec_wood:{official_split}:{category}:{image.stem}",
                        split="aux_train_reference" if official_split == "train" else "external_anomaly_test",
                        surface_type="wood_texture",
                        dataset_role="aux_anomaly_external",
                        source_split=official_split,
                        supervision_scope="auxiliary_only",
                        auxiliary_task="anomaly_segmentation",
                        auxiliary_label=category,
                        auxiliary_annotation_path=ctx.rel(mask) if mask.exists() else "",
                        source_variant=category,
                        base_sample_id=f"wvtec_wood:{official_split}:{category}:{image.stem}",
                        license_id="CC-BY-NC-SA-4.0",
                        notes="Official MVTec AD wood split retained; excluded from primary knot/crack training.",
                    )
                )
    return records


def _coating_base_and_variant(variant: str, stem: str) -> tuple[str, str]:
    if "DirectionChange" in variant:
        value = int(stem)
        base = (value - 1) // 8 + 1
        transform = (value - 1) % 8
        return f"{base:06d}", f"direction_{transform}"
    base = stem.split("_", 1)[0]
    suffix = stem[len(base):].lstrip("_") or "original"
    return f"{int(base):06d}", suffix


def scan_coating(ctx: ScanContext, root: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        source_root = variant_dir / variant_dir.name
        images = source_root / "Images"
        labels = source_root / "Labels"
        if not images.exists():
            continue
        for image in sorted(images.glob("*.jpg")):
            base_id, transform = _coating_base_and_variant(variant_dir.name, image.stem)
            annotation = labels / f"{image.stem}.txt"
            records.append(
                DataRecord(
                    image_path=ctx.rel(image),
                    dataset_id="coating",
                    sample_id=f"coating:{variant_dir.name}:{image.stem}",
                    group_id=f"coating:{base_id}",
                    split="auxiliary_stress",
                    surface_type="coated_wood_or_surface",
                    dataset_role="aux_synthetic_stress",
                    source_split="provided",
                    supervision_scope="auxiliary_only",
                    auxiliary_task="synthetic_detection_stress",
                    auxiliary_label="source_classes_0_1_2_3_unmapped",
                    auxiliary_annotation_path=ctx.rel(annotation) if annotation.exists() else "",
                    annotation_geometry="yolo_bbox",
                    source_variant=f"{variant_dir.name}:{transform}",
                    base_sample_id=f"coating:{base_id}",
                    notes="One of 45 synthetic derivatives; all derivatives inherit the same physical group and are excluded from primary training.",
                )
            )
    return records


def scan_all(ctx: ScanContext, config: dict[str, object]) -> list[DataRecord]:
    paths: dict[str, str] = config["datasets"]
    taxonomies: dict[str, dict[str, dict[str, object]]] = config["_defect_taxonomy"]["datasets"]
    records: list[DataRecord] = []
    records.extend(scan_urudendro(ctx, ctx.root(paths["urudendro"])))
    records.extend(scan_urudendro2(ctx, ctx.root(paths["urudendro2"])))
    records.extend(scan_urudendro4(ctx, ctx.root(paths["urudendro4"])))
    records.extend(scan_mokume(ctx, ctx.root(paths["mokume"])))
    records.extend(scan_vsb(ctx, ctx.root(paths["vsb"]), taxonomies["vsb"]))
    records.extend(
        scan_vnwoodknot(
            ctx,
            ctx.root(paths["vnwoodknot"]),
            taxonomies["vnwoodknot"],
        )
    )
    records.extend(scan_indiana(ctx, ctx.root(paths["indiana"])))
    records.extend(scan_oulu(ctx, ctx.root(paths["oulu"]), taxonomies["oulu"]))
    records.extend(scan_wvtec_wood(ctx, ctx.root(paths["wvtec_wood"])))
    records.extend(scan_coating(ctx, ctx.root(paths["coating"])))
    return records


def enrich_image_metadata(
    records: list[DataRecord],
    workspace_root: Path,
    cached: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    cached = cached or {}
    for record in records:
        path = workspace_root / record.image_path
        try:
            record.image_bytes = path.stat().st_size
            if record.image_width > 0 and record.image_height > 0 and record.image_bytes > 0:
                continue
            cached_value = cached.get(record.image_path)
            if cached_value and cached_value[2] == record.image_bytes:
                record.image_width, record.image_height, _ = cached_value
                continue
            with Image.open(path) as image:
                record.image_width, record.image_height = image.size
        except (OSError, UnidentifiedImageError) as error:
            detail = f"image_metadata_error={type(error).__name__}:{error}"
            record.notes = f"{record.notes}; {detail}" if record.notes else detail
