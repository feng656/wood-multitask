"""Native-resolution overlapping tiles for Stage-3 defect supervision.

Tiles are derived from the immutable catalog at read time.  The source image,
ring targets, and defect annotations are never rewritten.  A tile keeps its
parent sample id and origin so predictions can be mapped back to full-image
coordinates for evaluation or visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from .loaders import _open_image_bytes


@dataclass(frozen=True)
class TileRef:
    sample_id: str
    task: str
    x: int
    y: int
    source_width: int
    source_height: int
    scale: float
    labels: tuple[int, ...]

    @property
    def tile_id(self) -> str:
        return f"{self.sample_id}:tile:{self.x}:{self.y}:{self.scale:g}"


def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return values


def _scaled_box(annotation: dict[str, Any], scale: float) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) * scale for value in annotation.get("bbox", [0, 0, 0, 0])]
    return x, y, x + max(0.0, width), y + max(0.0, height)


def _intersects(box: tuple[float, float, float, float], x: int, y: int, size: int) -> bool:
    return box[2] > x and box[0] < x + size and box[3] > y and box[1] < y + size


def _tile_scale(width: int, height: int, tile_size: int) -> float:
    # Small Mokume faces are enlarged to fill the detector input.  Large
    # UruDendro/VSB images remain native-resolution and are tiled.
    return tile_size / max(width, height) if max(width, height) < tile_size else 1.0


def build_tile_refs(
    catalog: Any,
    records: list[Any],
    task: str,
    tile_size: int = 640,
    overlap: int = 160,
) -> list[TileRef]:
    if task not in {"defect", "bridge"}:
        raise ValueError("native tiles are only supported for defect and bridge tasks")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile overlap must be smaller than tile size")
    refs: list[TileRef] = []
    for record in records:
        indexed = catalog.coco_by_sample[record.sample_id]
        image_reference = indexed["image"]["file_name"]
        sx = sy = 1.0
        if task == "bridge":
            ring_row = catalog.ring_by_sample[record.sample_id]
            image_reference = ring_row["model_image_path"]
            source_width = int(indexed["image"]["width"])
            source_height = int(indexed["image"]["height"])
            sx = int(ring_row["target_width"]) / max(1, source_width)
            sy = int(ring_row["target_height"]) / max(1, source_height)
        with _open_image_bytes(image_reference) as opened:
            width, height = opened.size
        scale = _tile_scale(width, height, tile_size)
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        annotations = [catalog._normalize_annotation(annotation, sx=sx, sy=sy) for annotation in indexed["annotations"]]
        x_positions = axis_positions(scaled_width, tile_size, stride)
        y_positions = axis_positions(scaled_height, tile_size, stride)
        for y in y_positions:
            for x in x_positions:
                labels = tuple(
                    int(annotation.get("category_id", 0))
                    for annotation in annotations
                    if 0 < int(annotation.get("category_id", 0)) < 3
                    and _intersects(_scaled_box(annotation, scale), x, y, tile_size)
                )
                refs.append(TileRef(record.sample_id, task, x, y, width, height, scale, labels))
    return refs


def _decode_uncompressed_rle(segmentation: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    counts = segmentation.get("counts")
    size = segmentation.get("size", [height, width])
    if not isinstance(counts, list) or len(size) != 2:
        return None
    total = int(size[0]) * int(size[1])
    flat = np.zeros(total, dtype=np.uint8)
    offset = 0
    value = 0
    for count_value in counts:
        count = max(0, int(count_value))
        if value == 1 and count:
            flat[offset : min(total, offset + count)] = 1
        offset += count
        value = 1 - value
        if offset >= total:
            break
    return flat.reshape((int(size[0]), int(size[1])), order="F")


def _source_annotation_mask(
    annotation: dict[str, Any],
    source_width: int,
    source_height: int,
) -> np.ndarray:
    """Rasterize once in source coordinates, matching the full-image target."""
    segmentation = annotation.get("segmentation")
    canvas = Image.new("L", (source_width, source_height), 0)
    draw = ImageDraw.Draw(canvas)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = []
            for index in range(0, len(polygon) - 1, 2):
                points.append((float(polygon[index]), float(polygon[index + 1])))
            if len(points) >= 3:
                draw.polygon(points, fill=1)
    elif isinstance(segmentation, dict):
        decoded = _decode_uncompressed_rle(segmentation, source_width, source_height)
        if decoded is not None and decoded.any():
            return decoded.astype(np.uint8)
    if not np.asarray(canvas).any():
        x, y, width, height = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
        x1 = max(0, min(source_width, int(np.floor(x))))
        y1 = max(0, min(source_height, int(np.floor(y))))
        x2 = max(0, min(source_width, int(np.ceil(x + width))))
        y2 = max(0, min(source_height, int(np.ceil(y + height))))
        if x2 > x1 and y2 > y1:
            array = np.asarray(canvas).copy()
            array[y1:y2, x1:x2] = 1
            return array
    return np.asarray(canvas, dtype=np.uint8)


def _annotation_mask(
    annotation: dict[str, Any],
    source_width: int,
    source_height: int,
    scale: float,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> np.ndarray:
    source_mask = _source_annotation_mask(annotation, source_width, source_height)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    if scale != 1.0:
        source_mask = (
            np.asarray(
                Image.fromarray(source_mask * 255, mode="L").resize(
                    (scaled_width, scaled_height), Image.Resampling.NEAREST
                )
            )
            > 0
        ).astype(np.uint8)
    tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
    crop_width = min(tile_size, max(0, scaled_width - tile_x))
    crop_height = min(tile_size, max(0, scaled_height - tile_y))
    if crop_width > 0 and crop_height > 0:
        tile[:crop_height, :crop_width] = source_mask[
            tile_y : tile_y + crop_height, tile_x : tile_x + crop_width
        ]
    return tile


def build_tile_item_from_sample(sample: dict[str, Any], ref: TileRef, tile_size: int = 640) -> dict[str, np.ndarray]:
    source = np.asarray(sample["image"]).astype(np.uint8)
    source_height, source_width = source.shape[:2]
    if ref.scale != 1.0:
        scaled_width = max(1, int(round(source_width * ref.scale)))
        scaled_height = max(1, int(round(source_height * ref.scale)))
        source = np.asarray(Image.fromarray(source, mode="RGB").resize((scaled_width, scaled_height), Image.Resampling.BILINEAR))
    else:
        scaled_width, scaled_height = source_width, source_height
    crop_width = min(tile_size, max(0, scaled_width - ref.x))
    crop_height = min(tile_size, max(0, scaled_height - ref.y))
    image = np.full((tile_size, tile_size, 3), (123, 116, 103), dtype=np.uint8)
    if crop_width > 0 and crop_height > 0:
        image[:crop_height, :crop_width] = source[ref.y : ref.y + crop_height, ref.x : ref.x + crop_width]
    item: dict[str, np.ndarray] = {"sample_id": np.asarray(ref.tile_id), "image": image}
    if ref.task == "bridge":
        for name in ("boundary", "distance", "valid"):
            value = np.asarray(sample["ring_targets"][name])
            if ref.scale != 1.0:
                resample = Image.Resampling.BILINEAR if name == "distance" else Image.Resampling.NEAREST
                value = np.asarray(Image.fromarray(value.astype(np.float32 if name == "distance" else np.uint8), mode="F" if name == "distance" else "L").resize((scaled_width, scaled_height), resample))
            padded = np.zeros((tile_size, tile_size), dtype=np.float32)
            if crop_width > 0 and crop_height > 0:
                padded[:crop_height, :crop_width] = value[ref.y : ref.y + crop_height, ref.x : ref.x + crop_width]
            item[f"ring_{name}"] = padded
    annotations = sample["annotations"] if ref.task == "defect" else sample["defect"]["annotations"]
    boxes: list[list[float]] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    areas: list[float] = []
    crowds: list[int] = []
    for annotation in annotations:
        label = int(annotation.get("category_id", 0))
        if label <= 0 or label >= 3:
            continue
        # Rasterizing every full-resolution instance mask for every tile is
        # prohibitively expensive on images with many cracks. A mask cannot
        # contribute to this tile when its bounding box does not intersect it.
        if not _intersects(_scaled_box(annotation, ref.scale), ref.x, ref.y, tile_size):
            continue
        mask = _annotation_mask(annotation, source_width, source_height, ref.scale, ref.x, ref.y, tile_size)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
        masks.append(mask)
        areas.append(float(mask.sum()))
        crowds.append(int(annotation.get("iscrowd", 0)))
    item.update(
        {
            "target_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "target_labels": np.asarray(labels, dtype=np.int64),
            "target_masks": np.asarray(masks, dtype=np.uint8).reshape(-1, tile_size, tile_size),
            "target_area": np.asarray(areas, dtype=np.float32),
            "target_iscrowd": np.asarray(crowds, dtype=np.int64),
        }
    )
    return item


def build_tile_item(catalog: Any, ref: TileRef, tile_size: int = 640) -> dict[str, np.ndarray]:
    return build_tile_item_from_sample(catalog.load(ref.task, ref.sample_id), ref, tile_size)


__all__ = ["TileRef", "axis_positions", "build_tile_refs", "build_tile_item", "build_tile_item_from_sample"]
