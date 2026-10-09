from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

from .labelme import (
    PRIMARY_BRIDGE_CATEGORIES,
    bridge_shape_polygon,
    load_labelme_without_image_data,
    parse_bridge_label,
)


def _touches_frame(points: list[list[float]], width: int, height: int, tolerance: float = 1.5) -> bool:
    return any(
        x <= tolerance or y <= tolerance or x >= width - 1 - tolerance or y >= height - 1 - tolerance
        for x, y in points
    )


def _repair_polygon(points: list[list[float]]) -> tuple[list[list[float]], bool, str]:
    polygon = Polygon(points)
    if polygon.is_valid and polygon.area > 0:
        return points, False, ""
    repaired = make_valid(polygon)
    if isinstance(repaired, MultiPolygon):
        repaired = max(repaired.geoms, key=lambda item: item.area)
        note = "make_valid_largest_component"
    elif isinstance(repaired, Polygon):
        note = "make_valid"
    else:
        buffered = polygon.buffer(0)
        if isinstance(buffered, MultiPolygon):
            buffered = max(buffered.geoms, key=lambda item: item.area)
        if not isinstance(buffered, Polygon) or buffered.area <= 0:
            return points, False, "unrepairable"
        repaired = buffered
        note = "buffer0"
    return [[float(x), float(y)] for x, y in list(repaired.exterior.coords)[:-1]], True, note


def normalize_bridge_file(
    source: Path, destination: Path, image_width: int | None = None, image_height: int | None = None
) -> dict[str, object]:
    payload = load_labelme_without_image_data(source)
    width = int(image_width or payload.get("imageWidth") or 0)
    height = int(image_height or payload.get("imageHeight") or 0)
    seen: Counter[tuple[str, str]] = Counter()
    reserved: defaultdict[str, set[str]] = defaultdict(set)
    for original_shape in payload.get("shapes", []):
        parsed = parse_bridge_label(str(original_shape.get("label", "")))
        if parsed is not None:
            reserved[parsed.category].add(parsed.instance_id)
    changes: list[dict[str, object]] = []
    output_shapes: list[dict[str, object]] = []
    valid_instances = 0
    primary_instances = 0
    ignored_instances = 0

    for shape_index, original_shape in enumerate(payload.get("shapes", [])):
        shape = dict(original_shape)
        parsed = parse_bridge_label(str(shape.get("label", "")))
        points = bridge_shape_polygon(shape)
        if parsed is None or len(points) < 3:
            changes.append({"shape_index": shape_index, "action": "drop_invalid_shape"})
            continue
        valid_instances += 1
        if parsed.category in PRIMARY_BRIDGE_CATEGORIES:
            primary_instances += 1
        else:
            ignored_instances += 1
            changes.append({
                "shape_index": shape_index,
                "action": "retain_ignored_shape",
                "category": parsed.category,
            })
        fields = [field.strip() for field in str(shape["label"]).split("|")]
        key = (parsed.category, parsed.instance_id)
        seen[key] += 1
        if seen[key] > 1:
            prefix_match = re.match(r"^([A-Za-z]+)", parsed.instance_id)
            prefix = prefix_match.group(1) if prefix_match else ("K" if parsed.category == "knot" else "C")
            number = 1
            while f"{prefix}{number:02d}" in reserved[parsed.category]:
                number += 1
            old_id = fields[2]
            fields[2] = f"{prefix}{number:02d}"
            reserved[parsed.category].add(fields[2])
            seen[(parsed.category, fields[2])] += 1
            changes.append({"shape_index": shape_index, "action": "deduplicate_instance_id", "from": old_id, "to": fields[2]})

        touches = _touches_frame(points, width, height) if width > 0 and height > 0 else parsed.truncated
        if touches != parsed.truncated:
            changes.append({"shape_index": shape_index, "action": "normalize_truncated", "from": int(parsed.truncated), "to": int(touches)})
            fields[4] = "1" if touches else "0"

        if shape.get("shape_type") == "polygon":
            repaired_points, repaired, repair_note = _repair_polygon(points)
            if repair_note == "unrepairable":
                changes.append({"shape_index": shape_index, "action": "drop_unrepairable_polygon"})
                valid_instances -= 1
                continue
            if repaired:
                shape["points"] = repaired_points
                changes.append({"shape_index": shape_index, "action": "repair_polygon", "method": repair_note})
        shape["label"] = "|".join(fields)
        output_shapes.append(shape)

    normalized = {
        "version": payload.get("version", "5.1.0"),
        "flags": payload.get("flags", {}),
        "shapes": output_shapes,
        "imagePath": payload.get("imagePath", ""),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(normalized, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "source": source.as_posix(),
        "normalized": destination.as_posix(),
        "input_shapes": len(payload.get("shapes", [])),
        "valid_instances": valid_instances,
        "primary_instances": primary_instances,
        "ignored_instances": ignored_instances,
        "output_shapes": len(output_shapes),
        "changes": changes,
    }
