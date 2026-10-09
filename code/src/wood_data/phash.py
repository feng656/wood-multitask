from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _sample_bmp24(path: Path, size: int = 32) -> np.ndarray | None:
    with path.open("rb") as handle:
        header = handle.read(54)
        if len(header) < 54 or header[:2] != b"BM":
            return None
        offset = struct.unpack_from("<I", header, 10)[0]
        width = struct.unpack_from("<i", header, 18)[0]
        signed_height = struct.unpack_from("<i", header, 22)[0]
        bits = struct.unpack_from("<H", header, 28)[0]
        compression = struct.unpack_from("<I", header, 30)[0]
        if width <= 0 or signed_height == 0 or bits != 24 or compression != 0:
            return None
        height = abs(signed_height)
        stride = ((width * 3 + 3) // 4) * 4
        xs = np.linspace(0, width - 1, size).round().astype(int)
        ys = np.linspace(0, height - 1, size).round().astype(int)
        output = np.empty((size, size), dtype=np.float32)
        rows = [
            (height - 1 - source_y if signed_height > 0 else source_y, output_y)
            for output_y, source_y in enumerate(ys)
        ]
        # BMP with positive height is stored bottom-up. Reading by ascending
        # file offset avoids 32 reverse seeks per image on rotating disks.
        for storage_y, output_y in sorted(rows):
            handle.seek(offset + storage_y * stride)
            row = np.frombuffer(handle.read(stride), dtype=np.uint8)
            pixels = row[: width * 3].reshape(width, 3)[xs]
            output[output_y] = pixels[:, 0] * 0.114 + pixels[:, 1] * 0.587 + pixels[:, 2] * 0.299
        return output


def low_resolution_gray(path: Path, size: int = 32) -> np.ndarray:
    if path.suffix.lower() == ".bmp":
        sampled = _sample_bmp24(path, size)
        if sampled is not None:
            return sampled
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32)


def perceptual_hash(path: Path) -> int:
    gray = low_resolution_gray(path)
    transform = cv2.dct(gray)
    low = transform[:8, :8]
    median = float(np.median(low.ravel()[1:]))
    bits = low > median
    value = 0
    for enabled in bits.ravel():
        value = (value << 1) | int(enabled)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass
class _Node:
    value: int
    indices: list[int]
    children: dict[int, "_Node"]


class BKTree:
    def __init__(self) -> None:
        self.root: _Node | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _Node(value, [index], {})
            return
        node = self.root
        while True:
            distance = hamming(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            if distance not in node.children:
                node.children[distance] = _Node(value, [index], {})
                return
            node = node.children[distance]

    def query(self, value: int, threshold: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        found: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node.value)
            if distance <= threshold:
                found.extend((index, distance) for index in node.indices)
            low, high = distance - threshold, distance + threshold
            stack.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return found


def audit_cross_split_hashes(
    records: list[dict[str, str]],
    workspace_root: Path,
    output_csv: Path,
    threshold: int = 4,
    *,
    precomputed_hashes: dict[str, int] | None = None,
) -> dict[str, object]:
    tree = BKTree()
    pairs: list[dict[str, object]] = []
    hash_index = output_csv.parent / "perceptual_hashes.csv"
    cached_by_path: dict[str, int] = {}
    cached_by_sample: dict[str, int] = {}
    if hash_index.exists():
        with hash_index.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("phash_hex"):
                    value = int(row["phash_hex"], 16)
                    if row.get("image_path"):
                        cached_by_path[row["image_path"]] = value
                    if row.get("sample_id"):
                        cached_by_sample[row["sample_id"]] = value
    precomputed_hashes = precomputed_hashes or {}
    hash_index.parent.mkdir(parents=True, exist_ok=True)
    hash_rows: list[dict[str, object]] = []
    reused_precomputed = 0
    reused_local_cache = 0
    computed = 0
    for index, record in enumerate(records):
        sample_id = record["sample_id"]
        image_path = record["image_path"]
        value = precomputed_hashes.get(sample_id)
        source = "precomputed"
        if value is not None:
            reused_precomputed += 1
        else:
            value = cached_by_path.get(image_path, cached_by_sample.get(sample_id))
            source = "local_cache"
        if value is not None and source == "local_cache":
            reused_local_cache += 1
        if value is None:
            value = perceptual_hash(workspace_root / record["image_path"])
            source = "computed"
            computed += 1
        hash_rows.append({
            "sample_id": sample_id, "image_path": image_path,
            "split": record["split"], "group_id": record["group_id"],
            "phash_hex": f"{value:016x}", "hash_source": source,
        })
        for previous, distance in tree.query(value, threshold):
            other = records[previous]
            if other["split"] == record["split"]:
                continue
            pairs.append({
                "left_sample_id": other["sample_id"], "right_sample_id": record["sample_id"],
                "left_split": other["split"], "right_split": record["split"],
                "left_group_id": other["group_id"], "right_group_id": record["group_id"],
                "left_path": other["image_path"], "right_path": record["image_path"],
                "hamming_distance": distance,
            })
        tree.add(value, index)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with hash_index.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "split", "group_id", "phash_hex", "hash_source"],
        )
        writer.writeheader()
        writer.writerows(hash_rows)
    fields = ["left_sample_id", "right_sample_id", "left_split", "right_split", "left_group_id", "right_group_id", "left_path", "right_path", "hamming_distance"]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    return {
        "algorithm": "64-bit pHash (32x32 grayscale DCT; sparse row sampling for uncompressed 24-bit BMP)",
        "threshold": threshold,
        "images_hashed": len(records),
        "cross_split_candidates": len(pairs),
        "exact_hash_candidates": sum(int(pair["hamming_distance"]) == 0 for pair in pairs),
        "hashes_reused_precomputed": reused_precomputed,
        "hashes_reused_local_cache": reused_local_cache,
        "hashes_computed": computed,
        "candidate_csv": output_csv.as_posix(),
        "hash_index": hash_index.as_posix(),
        "interpretation": "Candidates require visual or pixel registration review; pHash proximity alone is not proof of duplicated source data.",
    }
