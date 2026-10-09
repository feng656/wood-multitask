from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .labelme import bridge_shape_polygon, load_labelme_without_image_data, parse_bridge_label


CATEGORY_IDS = {
    "knot": 1,
    "crack": 2,
}

VSB_PALETTE = {
    "live_knot": (0x00, 0xFF, 0x00),
    "death_know": (0xFF, 0x00, 0x00),
    "knot_missing": (0xFF, 0x64, 0x00),
    "knot_with_crack": (0xFF, 0xAF, 0x00),
    "crack": (0xFF, 0x00, 0x64),
    "quartzity": (0x64, 0x00, 0x64),
    "resin": (0xFF, 0x00, 0xFF),
    "marrow": (0x00, 0x00, 0xFF),
    "blue_stain": (0x10, 0xFF, 0xFF),
    "overgrown": (0x00, 0x40, 0x00),
}

_VSB_LUT_CACHE: np.ndarray | None = None


def canonical_vsb_label(value: str) -> str:
    key = value.strip().casefold().replace(" ", "_")
    aliases = {"live_knot": "live_knot", "live_knot_": "live_knot", "dead_knot": "death_know"}
    return aliases.get(key, key)


def taxonomy_lookup(mapping: dict[str, dict[str, object]], raw: str) -> dict[str, object]:
    if raw in mapping:
        return mapping[raw]
    canonical = canonical_vsb_label(raw)
    for key, value in mapping.items():
        if canonical_vsb_label(key) == canonical:
            return value
    return {"category": "other", "subtype": "", "attributes": ["unmapped_source_class"]}


def _annotation(category: str, bbox: list[float], **extra: object) -> dict[str, object]:
    x, y, width, height = bbox
    result: dict[str, object] = {
        "category_id": CATEGORY_IDS[category],
        "bbox": [round(float(x), 4), round(float(y), 4), round(float(width), 4), round(float(height), 4)],
        "area": round(float(max(0.0, width) * max(0.0, height)), 4),
        "iscrowd": 0,
        "segmentation": [],
        "category": category,
    }
    result.update(extra)
    return result


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict[str, object]:
    values = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in values:
        bit = int(value != 0)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def parse_vsb_boxes(path: Path, width: int, height: int, mapping: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    if not path.exists():
        return annotations
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        raw = parts[0]
        x1, y1, x2, y2 = (float(value.replace(",", ".")) for value in parts[1:])
        mapped = taxonomy_lookup(mapping, raw)
        if mapped.get("category") not in CATEGORY_IDS or mapped.get("train_action") not in {None, "keep"}:
            continue
        annotations.append(_annotation(
            str(mapped["category"]),
            [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
            subtype=str(mapped.get("subtype", "")),
            attributes=list(mapped.get("attributes", [])),
            source_class=raw,
        ))
    return annotations


def parse_vn_boxes(path: Path, width: int, height: int, mapped: dict[str, object]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, box_width, box_height = parts
        cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
        result.append(_annotation(
            str(mapped["category"]),
            [(cx - box_width / 2) * width, (cy - box_height / 2) * height, box_width * width, box_height * height],
            subtype=str(mapped.get("subtype", "")),
            attributes=list(mapped.get("attributes", [])),
            source_class="folder_class",
        ))
    return result


def parse_oulu_boxes_and_ignores(
    path: Path, mapping: dict[str, dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not path.exists():
        return [], []
    root = ET.parse(path).getroot()
    result: list[dict[str, object]] = []
    ignored: list[dict[str, object]] = []
    image_width = float(root.findtext("size/width") or 0)
    image_height = float(root.findtext("size/height") or 0)
    for node in root.findall("object"):
        raw = (node.findtext("name") or "").strip()
        if not raw:
            continue
        mapped = mapping.get(raw, mapping.get("UNKNOWN", {"category": "other", "subtype": "", "attributes": []}))
        box = node.find("bndbox")
        if box is None:
            continue
        x1 = max(0.0, float(box.findtext("xmin") or 0))
        y1 = max(0.0, float(box.findtext("ymin") or 0))
        x2 = min(image_width, float(box.findtext("xmax") or 0)) if image_width else float(box.findtext("xmax") or 0)
        y2 = min(image_height, float(box.findtext("ymax") or 0)) if image_height else float(box.findtext("ymax") or 0)
        if x2 <= x1 or y2 <= y1:
            continue
        attributes = list(mapped.get("attributes", []))
        if (node.findtext("truncated") or "0") == "1" and "truncated" not in attributes:
            attributes.append("truncated")
        if mapped.get("train_action") == "keep" and mapped.get("category") in CATEGORY_IDS:
            result.append(_annotation(
                str(mapped["category"]), [x1, y1, x2 - x1, y2 - y1],
                subtype=str(mapped.get("subtype", "")), attributes=attributes, source_class=raw,
            ))
        else:
            ignored.append({
                "bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)],
                "source_class": raw,
                "reason": "outside_primary_knot_crack_taxonomy",
            })
    return result, ignored


def parse_oulu_boxes(path: Path, mapping: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return parse_oulu_boxes_and_ignores(path, mapping)[0]


def parse_bridge_polygons_and_ignores(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = load_labelme_without_image_data(path)
    result: list[dict[str, object]] = []
    ignored: list[dict[str, object]] = []
    for shape in payload.get("shapes", []):
        label = parse_bridge_label(str(shape.get("label", "")))
        points = bridge_shape_polygon(shape)
        if label is None or len(points) < 3:
            continue
        values = np.asarray(points, dtype=float)
        x1, y1 = values.min(axis=0)
        x2, y2 = values.max(axis=0)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        if label.category not in CATEGORY_IDS:
            ignored.append({
                "bbox": [round(float(value), 4) for value in bbox],
                "segmentation": [[float(value) for point in points for value in point]],
                "source_class": shape["label"],
                "reason": "outside_primary_knot_crack_taxonomy",
            })
            continue
        attributes = []
        if label.truncated:
            attributes.append("truncated")
        if label.visibility != "clear":
            attributes.append(f"visibility_{label.visibility}")
        if label.affects_rings:
            attributes.append("affects_rings")
        result.append(_annotation(
            label.category, bbox, subtype=label.subtype,
            attributes=attributes, source_class=shape["label"], instance_id=label.instance_id,
            segmentation=[[float(value) for point in points for value in point]],
            source_shape_type=shape.get("shape_type", ""),
            geometry_type=("true_polygon" if shape.get("shape_type") == "polygon" else "axis_aligned_rectangle_from_polygon"),
        ))
        result[-1]["area"] = abs(float(cv2.contourArea(values.astype(np.float32))))
    return result, ignored


def parse_bridge_polygons(path: Path) -> list[dict[str, object]]:
    return parse_bridge_polygons_and_ignores(path)[0]


def encode_rgb(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.uint32)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def _vsb_lut(mapping: dict[str, dict[str, object]]) -> np.ndarray:
    global _VSB_LUT_CACHE
    if _VSB_LUT_CACHE is None:
        lookup = np.zeros(1 << 24, dtype=np.uint8)
        for raw, color in VSB_PALETTE.items():
            color_id = (color[0] << 16) | (color[1] << 8) | color[2]
            mapped = taxonomy_lookup(mapping, raw)
            category = mapped.get("category")
            if category in CATEGORY_IDS:
                lookup[color_id] = CATEGORY_IDS[str(category)]
        _VSB_LUT_CACHE = lookup
    return _VSB_LUT_CACHE


def decode_vsb_encoded(encoded: np.ndarray, mapping: dict[str, dict[str, object]]) -> tuple[np.ndarray, dict[str, int]]:
    unified = _vsb_lut(mapping)[encoded]
    unknown_mask = (encoded != 0) & (unified == 0)
    unknown = encoded[unknown_mask]
    unknown_counts: dict[str, int] = {}
    if unknown.size:
        colors, counts = np.unique(unknown, return_counts=True)
        unknown_counts = {f"{int(color):06X}": int(count) for color, count in zip(colors, counts)}
    return unified, unknown_counts


def decode_vsb_rgb(rgb: np.ndarray, mapping: dict[str, dict[str, object]]) -> tuple[np.ndarray, dict[str, int]]:
    return decode_vsb_encoded(encode_rgb(rgb), mapping)


def decode_vsb_semantic(path: Path, mapping: dict[str, dict[str, object]]) -> tuple[np.ndarray, dict[str, int]]:
    return decode_vsb_rgb(np.asarray(Image.open(path).convert("RGB")), mapping)


def attach_vsb_segmentations(
    annotations: list[dict[str, object]], mask_path: Path, rgb: np.ndarray | None = None,
    encoded: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], int]:
    if rgb is None:
        rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    if encoded is None:
        encoded = encode_rgb(rgb)
    unmatched = 0
    used: dict[str, set[int]] = defaultdict(set)
    components: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    for annotation in annotations:
        raw = canonical_vsb_label(str(annotation["source_class"]))
        color = VSB_PALETTE.get(raw)
        if color is None:
            unmatched += 1
            continue
        if raw not in components:
            color_id = (color[0] << 16) | (color[1] << 8) | color[2]
            binary = (encoded == color_id).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            components[raw] = (count, labels, stats)
        count, labels, stats = components[raw]
        x, y, width, height = annotation["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(rgb.shape[1], int(np.ceil(x + width))), min(rgb.shape[0], int(np.ceil(y + height)))
        best_label = 0
        best_overlap = 0
        for component in range(1, count):
            if component in used[raw]:
                continue
            overlap = int(np.count_nonzero(labels[y1:y2, x1:x2] == component))
            if overlap > best_overlap:
                best_label, best_overlap = component, overlap
        if not best_label:
            unmatched += 1
            continue
        used[raw].add(best_label)
        component_mask = (labels == best_label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.reshape(-1, 2).astype(float).ravel().tolist() for contour in contours if len(contour) >= 3]
        annotation["segmentation"] = polygons if polygons else mask_to_uncompressed_rle(component_mask)
        annotation["area"] = int(stats[best_label, cv2.CC_STAT_AREA])
        annotation["mask_match_overlap_pixels"] = best_overlap
    return annotations, unmatched


def process_vsb_task(task: tuple[str, str, int, int, dict[str, dict[str, object]], str]) -> tuple[list[dict[str, object]], int, int]:
    annotation_value, mask_value, width, height, mapping, destination_value = task
    annotations = parse_vsb_boxes(Path(annotation_value), width, height, mapping)
    bgr = cv2.imread(mask_value, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read VSB semantic mask: {mask_value}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    encoded = encode_rgb(rgb)
    annotations, unmatched = attach_vsb_segmentations(annotations, Path(mask_value), rgb, encoded)
    unified, unknown = decode_vsb_encoded(encoded, mapping)
    destination = Path(destination_value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), unified, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"cannot write unified VSB mask: {destination}")
    return annotations, unmatched, sum(unknown.values())


def coco_document(images: list[dict[str, object]], annotations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "info": {"description": "Unified wood defects; extended COCO fields retain source taxonomy and physical groups"},
        "licenses": [],
        "categories": [
            {"id": identifier, "name": name, "supercategory": "wood_defect"}
            for name, identifier in CATEGORY_IDS.items()
        ],
        "images": images,
        "annotations": annotations,
    }


def save_coco(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
