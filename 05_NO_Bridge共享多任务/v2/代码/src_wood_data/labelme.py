from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


BRIDGE_SOURCE_CATEGORIES = {"knot", "crack", "discoloration", "resin", "other"}
PRIMARY_BRIDGE_CATEGORIES = {"knot", "crack"}


@dataclass(frozen=True)
class BridgeLabel:
    category: str
    subtype: str
    instance_id: str
    visibility: str
    truncated: bool
    affects_rings: bool


def load_labelme_without_image_data(path: Path, max_prefix_bytes: int = 8_000_000) -> dict[str, object]:
    """Read LabelMe metadata without loading its often very large embedded image."""
    marker = b'"imageData"'
    payload = bytearray()
    with path.open("rb") as handle:
        while len(payload) < max_prefix_bytes:
            chunk = handle.read(min(1_048_576, max_prefix_bytes - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            marker_index = payload.find(marker)
            if marker_index >= 0:
                prefix = bytes(payload[:marker_index]).rstrip()
                if prefix.endswith(b","):
                    prefix = prefix[:-1]
                return json.loads(prefix + b"}")
    return json.loads(bytes(payload))


def parse_bridge_label(value: str) -> BridgeLabel | None:
    fields = [field.strip() for field in value.split("|")]
    if len(fields) != 6:
        return None
    category, subtype, instance_id, visibility, truncated, affects_rings = fields
    if category not in BRIDGE_SOURCE_CATEGORIES:
        return None
    if not subtype or not instance_id or visibility not in {"clear", "partial"}:
        return None
    if truncated not in {"0", "1"} or affects_rings not in {"0", "1"}:
        return None
    return BridgeLabel(
        category=category,
        subtype=subtype,
        instance_id=instance_id,
        visibility=visibility,
        truncated=truncated == "1",
        affects_rings=affects_rings == "1",
    )


def bridge_shape_polygon(shape: dict[str, object]) -> list[list[float]]:
    points = shape.get("points", [])
    if not isinstance(points, list):
        return []
    shape_type = str(shape.get("shape_type", ""))
    if shape_type == "polygon" and len(points) >= 3:
        return [[float(point[0]), float(point[1])] for point in points]
    if shape_type == "rectangle" and len(points) == 2:
        x1, y1 = map(float, points[0])
        x2, y2 = map(float, points[1])
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if left < right and top < bottom:
            return [[left, top], [right, top], [right, bottom], [left, bottom]]
    return []


def valid_bridge_shapes(
    path: Path | None, *, primary_only: bool = True
) -> list[tuple[dict[str, object], BridgeLabel]]:
    if path is None or not path.exists():
        return []
    try:
        payload = load_labelme_without_image_data(path)
    except (OSError, UnicodeError, ValueError):
        return []
    valid: list[tuple[dict[str, object], BridgeLabel]] = []
    for shape in payload.get("shapes", []):
        if not bridge_shape_polygon(shape):
            continue
        label = parse_bridge_label(str(shape.get("label", "")))
        if label is not None and (not primary_only or label.category in PRIMARY_BRIDGE_CATEGORIES):
            valid.append((shape, label))
    return valid


def bridge_summary(path: Path | None) -> tuple[int, str, str, str]:
    shapes = valid_bridge_shapes(path, primary_only=True)
    categories = sorted({label.category for _, label in shapes})
    subtypes = sorted({label.subtype for _, label in shapes if label.subtype != "na"})
    attributes: set[str] = set()
    for _, label in shapes:
        if label.visibility != "clear":
            attributes.add(f"visibility_{label.visibility}")
        if label.truncated:
            attributes.add("truncated")
        if label.affects_rings:
            attributes.add("affects_rings")
    return len(shapes), ";".join(categories), ";".join(subtypes), ";".join(sorted(attributes))


def bridge_ignored_count(path: Path | None) -> int:
    all_shapes = valid_bridge_shapes(path, primary_only=False)
    return sum(label.category not in PRIMARY_BRIDGE_CATEGORIES for _, label in all_shapes)


def mokume_bridge_annotation(cube_dir: Path, face_id: str) -> Path | None:
    directory = cube_dir / "defects"
    for name in (f"{face_id}_col.json", f"{face_id}.json"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def mokume_bridge_is_complete(cube_dir: Path) -> bool:
    return all(mokume_bridge_annotation(cube_dir, face_id) is not None for face_id in "ABCDEF")
