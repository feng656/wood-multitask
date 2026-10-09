from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import cv2
from PIL import Image, ImageDraw

from .labelme import load_labelme_without_image_data


@dataclass
class RingTarget:
    skeleton: np.ndarray
    boundary: np.ndarray
    distance: np.ndarray
    instance: np.ndarray
    valid: np.ndarray
    closed_ring_count: int = 0
    non_nested_pairs: int = 0


def _polygon_area(points: list[list[float]]) -> float:
    values = np.asarray(points, dtype=np.float64)
    x = values[:, 0]
    y = values[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _draw_line(shape: tuple[int, int], points: list[list[float]], close: bool) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    xy = [(round(point[0]), round(point[1])) for point in points]
    if close and xy:
        xy.append(xy[0])
    if len(xy) >= 2:
        draw.line(xy, fill=1, width=1)
    return np.asarray(image, dtype=bool)


def _fill_polygon(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(image).polygon(
        [(round(point[0]), round(point[1])) for point in points], fill=1
    )
    return np.asarray(image, dtype=bool)


def finish_targets(
    skeleton: np.ndarray,
    valid: np.ndarray,
    instance: np.ndarray | None = None,
    boundary_width: int = 5,
    distance_tau: float = 16.0,
    closed_ring_count: int = 0,
    non_nested_pairs: int = 0,
) -> RingTarget:
    if skeleton.shape != valid.shape:
        raise ValueError("skeleton and valid masks must have identical shapes")
    radius = max(0, (boundary_width - 1) // 2)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    boundary = cv2.dilate(skeleton.astype(np.uint8), kernel) > 0
    distance = np.minimum(cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / distance_tau, 1.0)
    distance[~valid] = 1.0
    boundary &= valid
    if instance is None:
        instance = np.zeros(skeleton.shape, dtype=np.uint16)
    instance = instance.astype(np.uint16, copy=False)
    instance[~valid] = 0
    return RingTarget(
        skeleton=skeleton.astype(bool),
        boundary=boundary,
        distance=distance.astype(np.float32),
        instance=instance,
        valid=valid.astype(bool),
        closed_ring_count=closed_ring_count,
        non_nested_pairs=non_nested_pairs,
    )


def labelme_closed_targets(
    annotation_path: Path,
    image_shape: tuple[int, int],
    boundary_width: int = 5,
    distance_tau: float = 16.0,
) -> RingTarget:
    payload = load_labelme_without_image_data(annotation_path)
    polygons = [
        shape["points"]
        for shape in payload.get("shapes", [])
        if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3
    ]
    if not polygons:
        raise ValueError(f"no closed ring polygons in {annotation_path}")
    polygons.sort(key=_polygon_area)
    fills = [_fill_polygon(image_shape, points) for points in polygons]
    skeleton = np.zeros(image_shape, dtype=bool)
    for points in polygons:
        skeleton |= _draw_line(image_shape, points, close=True)

    instance = np.zeros(image_shape, dtype=np.uint16)
    non_nested_pairs = 0
    for index in range(1, len(fills)):
        inner = fills[index - 1]
        outer = fills[index]
        if np.any(inner & ~outer):
            non_nested_pairs += 1
        instance[outer & ~inner] = index
    valid = fills[-1]
    return finish_targets(
        skeleton,
        valid,
        instance,
        boundary_width,
        distance_tau,
        closed_ring_count=len(polygons),
        non_nested_pairs=non_nested_pairs,
    )


def mokume_targets(
    annotation_path: Path,
    boundary_width: int = 5,
    distance_tau: float = 16.0,
) -> RingTarget:
    labels = np.asarray(Image.open(annotation_path))
    if labels.ndim == 3:
        labels = labels[..., 0]
    # Mokume stores each manually traced 1 px ring under an integer ID;
    # 255 is unlabeled background, not an invalid-region mask.
    skeleton = labels != 255
    valid = np.ones(labels.shape, dtype=bool)
    return finish_targets(skeleton, valid, boundary_width=boundary_width, distance_tau=distance_tau)


class IndianaAnnotations:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, list[list[list[float]]]]] = {}

    def polylines(self, xml_path: Path, image_name: str) -> list[list[list[float]]]:
        xml_path = xml_path.resolve()
        if xml_path not in self._cache:
            by_name: dict[str, list[list[list[float]]]] = {}
            root = ET.parse(xml_path).getroot()
            for image_node in root.findall(".//image"):
                name = PurePosixPath(image_node.attrib.get("name", "")).name
                rings: list[list[list[float]]] = []
                for node in image_node.findall("polyline"):
                    if node.attrib.get("label") != "Ring":
                        continue
                    points = [
                        [float(x), float(y)]
                        for x, y in (
                            pair.split(",", 1)
                            for pair in node.attrib.get("points", "").split(";")
                            if "," in pair
                        )
                    ]
                    if len(points) >= 2:
                        rings.append(points)
                by_name[name] = rings
            self._cache[xml_path] = by_name
        return self._cache[xml_path].get(image_name, [])


def indiana_targets(
    xml_path: Path,
    image_name: str,
    image_shape: tuple[int, int],
    ignore_mask_path: Path | None,
    annotations: IndianaAnnotations,
    boundary_width: int = 5,
    distance_tau: float = 16.0,
) -> RingTarget:
    polylines = annotations.polylines(xml_path, image_name)
    if not polylines:
        raise ValueError(f"no Ring polylines for {image_name} in {xml_path}")
    skeleton = np.zeros(image_shape, dtype=bool)
    for points in polylines:
        skeleton |= _draw_line(image_shape, points, close=False)
    if ignore_mask_path and ignore_mask_path.exists():
        ignore_image = Image.open(ignore_mask_path).convert("L")
        valid = np.asarray(ignore_image) >= 128
    else:
        valid = np.ones(image_shape, dtype=bool)
    skeleton &= valid
    return finish_targets(skeleton, valid, boundary_width=boundary_width, distance_tau=distance_tau)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)


def save_target(root: Path, dataset_id: str, sample_name: str, target: RingTarget) -> dict[str, str]:
    relative = Path(dataset_id) / f"{sample_name}.png"
    paths = {
        "skeleton_path": root / "ring_targets" / "skeleton" / relative,
        "boundary_path": root / "ring_boundary" / relative,
        "distance_path": root / "ring_distance" / relative,
        "instance_path": root / "ring_instance" / relative,
        "valid_path": root / "ring_valid" / relative,
    }
    _save_png(paths["skeleton_path"], target.skeleton.astype(np.uint8) * 255)
    _save_png(paths["boundary_path"], target.boundary.astype(np.uint8) * 255)
    _save_png(paths["distance_path"], np.round(target.distance * 65535).astype(np.uint16))
    _save_png(paths["instance_path"], target.instance.astype(np.uint16))
    _save_png(paths["valid_path"], target.valid.astype(np.uint8) * 255)
    return {key: path.as_posix() for key, path in paths.items()}


def write_target_index(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "dataset_id", "split", "source_annotation", "model_image_path",
        "skeleton_path",
        "boundary_path", "distance_path", "instance_path", "valid_path", "width",
        "height", "target_width", "target_height", "skeleton_pixels", "valid_pixels", "closed_ring_count",
        "instance_count", "non_nested_pairs", "status", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
