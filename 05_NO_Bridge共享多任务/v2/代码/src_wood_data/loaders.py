from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .labelme import parse_bridge_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data_processed"

try:  # pragma: no cover - optional dependency on the training host
    from torch.utils.data import Dataset as TorchDatasetBase  # type: ignore
except Exception:  # pragma: no cover - local data engineering host has no torch
    class TorchDatasetBase:  # type: ignore
        pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (WORKSPACE_ROOT, PROJECT_ROOT, DATA_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (WORKSPACE_ROOT / path).resolve()


def _open_image_bytes(reference: str) -> Image.Image:
    if reference.startswith("zip://"):
        archive_value, member = reference[len("zip://") :].split("::", 1)
        archive = _resolve_path(archive_value)
        with ZipFile(archive) as handle:
            payload = handle.read(member)
        return Image.open(io.BytesIO(payload))
    return Image.open(_resolve_path(reference))


def _load_image(reference: str, mode: str = "RGB") -> np.ndarray:
    with _open_image_bytes(reference) as image:
        if mode:
            image = image.convert(mode)
        return np.asarray(image)


def _resize_array(array: np.ndarray, width: int, height: int, resample: int = Image.Resampling.NEAREST) -> np.ndarray:
    if array.shape[1] == width and array.shape[0] == height:
        return array
    image = Image.fromarray(array)
    return np.asarray(image.resize((width, height), resample=resample))


def _scale_bbox(bbox: list[float], sx: float, sy: float) -> list[float]:
    x, y, width, height = bbox
    return [float(x) * sx, float(y) * sy, float(width) * sx, float(height) * sy]


def _scale_segmentation(segmentation: Any, sx: float, sy: float) -> Any:
    if isinstance(segmentation, list):
        scaled: list[list[float]] = []
        for polygon in segmentation:
            if not isinstance(polygon, list):
                continue
            values = list(map(float, polygon))
            points: list[float] = []
            for index in range(0, len(values), 2):
                points.extend([values[index] * sx, values[index + 1] * sy])
            scaled.append(points)
        return scaled
    return segmentation


def _attributes_from_text(value: str) -> list[str]:
    return [item for item in (part.strip() for part in value.split(";")) if item]


@dataclass(frozen=True)
class LoaderRecord:
    sample_id: str
    mode: str
    source_image: str
    split: str
    group_id: str
    task_mask: dict[str, int]


class WoodDataset(TorchDatasetBase):
    def __init__(self, catalog: "WoodCatalog", mode: str, records: list[LoaderRecord]):
        self.catalog = catalog
        self.mode = mode
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return self.catalog.load(record.mode, record.sample_id)


class WoodCatalog:
    def __init__(self, workspace_root: Path | None = None, project_root: Path | None = None):
        self.workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.data_root = self.project_root / "data_processed"
        self.manifest_rows = _read_csv(self.data_root / "manifests" / "manifest.csv")
        self.manifest_by_sample = {row["sample_id"]: row for row in self.manifest_rows}
        self.ring_rows = _read_csv(self.data_root / "ring_targets" / "ring_targets.csv")
        self.ring_by_sample = {row["sample_id"]: row for row in self.ring_rows if row.get("status") == "ok"}
        self.crop_rows = _read_csv(self.data_root / "classification_crops" / "classification_crops.csv")
        self.crop_by_path = {row["crop_path"]: row for row in self.crop_rows}
        self.crops_by_source: dict[str, list[dict[str, str]]] = {}
        for row in self.crop_rows:
            self.crops_by_source.setdefault(row["source_image"], []).append(row)

        self.coco_by_sample: dict[str, dict[str, Any]] = {}
        self.coco_by_split: dict[str, dict[str, Any]] = {}
        for path in sorted((self.data_root / "defect_coco").glob("instances_*.json")):
            payload = _read_json(path)
            split = path.stem.replace("instances_", "")
            self.coco_by_split[split] = payload
            images = {int(item["id"]): item for item in payload.get("images", [])}
            annotations_by_image: dict[str, list[dict[str, Any]]] = {}
            for annotation in payload.get("annotations", []):
                image = images.get(int(annotation["image_id"]))
                if image is None:
                    continue
                annotations_by_image.setdefault(str(image["sample_id"]), []).append(annotation)
            for image_id, image in images.items():
                sample_id = str(image["sample_id"])
                self.coco_by_sample[sample_id] = {
                    "split": split,
                    "image": image,
                    "annotations": annotations_by_image.get(sample_id, []),
                }

    @classmethod
    def default(cls) -> "WoodCatalog":
        return cls()

    def manifest_records(
        self,
        *,
        dataset_ids: set[str] | None = None,
        splits: set[str] | None = None,
        roles: set[str] | None = None,
    ) -> list[dict[str, str]]:
        rows = self.manifest_rows
        if dataset_ids is not None:
            rows = [row for row in rows if row["dataset_id"] in dataset_ids]
        if splits is not None:
            rows = [row for row in rows if row["split"] in splits]
        if roles is not None:
            rows = [row for row in rows if row["dataset_role"] in roles]
        return rows

    def ring_samples(
        self,
        *,
        dataset_ids: set[str] | None = None,
        splits: set[str] | None = None,
    ) -> WoodDataset:
        rows: list[LoaderRecord] = []
        for row in self.manifest_records(dataset_ids=dataset_ids, splits=splits):
            if row["has_ring"] != "1":
                continue
            if row["sample_id"] not in self.ring_by_sample:
                continue
            rows.append(
                LoaderRecord(
                    sample_id=row["sample_id"],
                    mode="ring",
                    source_image=row["image_path"],
                    split=row["split"],
                    group_id=row["group_id"],
                    task_mask={"ring": 1, "defect": 0, "classification": 0},
                )
            )
        return WoodDataset(self, "ring", rows)

    def defect_samples(
        self,
        *,
        dataset_ids: set[str] | None = None,
        splits: set[str] | None = None,
    ) -> WoodDataset:
        rows: list[LoaderRecord] = []
        for row in self.manifest_records(dataset_ids=dataset_ids, splits=splits):
            if row["defect_label_state"] == "missing":
                continue
            if row["sample_id"] not in self.coco_by_sample:
                continue
            rows.append(
                LoaderRecord(
                    sample_id=row["sample_id"],
                    mode="defect",
                    source_image=row["image_path"],
                    split=row["split"],
                    group_id=row["group_id"],
                    task_mask={"ring": 0, "defect": 1, "classification": 0},
                )
            )
        return WoodDataset(self, "defect", rows)

    def bridge_samples(
        self,
        *,
        dataset_ids: set[str] | None = None,
        splits: set[str] | None = None,
    ) -> WoodDataset:
        rows: list[LoaderRecord] = []
        for row in self.manifest_records(dataset_ids=dataset_ids, splits=splits, roles={"bridge"}):
            if row["defect_label_state"] == "missing":
                continue
            if row["sample_id"] not in self.ring_by_sample:
                continue
            if row["sample_id"] not in self.coco_by_sample:
                continue
            rows.append(
                LoaderRecord(
                    sample_id=row["sample_id"],
                    mode="bridge",
                    source_image=row["image_path"],
                    split=row["split"],
                    group_id=row["group_id"],
                    task_mask={"ring": 1, "defect": 1, "classification": 0},
                )
            )
        return WoodDataset(self, "bridge", rows)

    def classification_samples(
        self,
        *,
        splits: set[str] | None = None,
        categories: set[str] | None = None,
    ) -> WoodDataset:
        rows: list[LoaderRecord] = []
        selected = self.crop_rows
        if splits is not None:
            selected = [row for row in selected if row["split"] in splits]
        if categories is not None:
            selected = [row for row in selected if row["category"] in categories]
        for row in selected:
            rows.append(
                LoaderRecord(
                    sample_id=row["crop_path"],
                    mode="classification",
                    source_image=row["source_image"],
                    split=row["split"],
                    group_id=row["group_id"],
                    task_mask={"ring": 0, "defect": 0, "classification": 1},
                )
            )
        return WoodDataset(self, "classification", rows)

    def load(self, mode: str, sample_id: str) -> dict[str, Any]:
        if mode == "ring":
            return self._load_ring(sample_id)
        if mode == "defect":
            return self._load_defect(sample_id)
        if mode == "classification":
            return self._load_classification(sample_id)
        if mode == "bridge":
            return self._load_bridge(sample_id)
        raise KeyError(f"unknown loader mode: {mode}")

    def _task_mask(self, row: dict[str, str], ring: bool = False, defect: bool = False, classification: bool = False) -> dict[str, int]:
        return {
            "ring": int(ring and row.get("has_ring") == "1"),
            "defect": int(defect and row.get("defect_label_state") != "missing"),
            "classification": int(classification),
        }

    def _load_ring(self, sample_id: str) -> dict[str, Any]:
        row = self.manifest_by_sample[sample_id]
        ring_row = self.ring_by_sample[sample_id]
        image = _load_image(ring_row["model_image_path"], "RGB")
        target = {
            "skeleton": _load_image(ring_row["skeleton_path"], "L") > 0,
            "boundary": _load_image(ring_row["boundary_path"], "L") > 0,
            "distance": _load_image(ring_row["distance_path"], "I").astype(np.float32) / 65535.0,
            "instance": _load_image(ring_row["instance_path"], "I").astype(np.uint16),
            "valid": _load_image(ring_row["valid_path"], "L") > 0,
        }
        return {
            "mode": "ring",
            "sample_id": sample_id,
            "image": image,
            "image_path": ring_row["model_image_path"],
            "split": row["split"],
            "group_id": row["group_id"],
            "task_mask": {"ring": 1, "defect": 0, "classification": 0},
            "targets": target,
            "metadata": row,
        }

    def _load_defect(self, sample_id: str) -> dict[str, Any]:
        row = self.manifest_by_sample[sample_id]
        coco = self.coco_by_sample[sample_id]
        image_info = coco["image"]
        image = _load_image(image_info["file_name"], "RGB")
        mask_path = image_info.get("unified_semantic_mask") or image_info.get("source_semantic_mask") or row.get("defect_mask_path", "")
        semantic_mask = _load_image(mask_path, "RGB") if mask_path and str(mask_path).endswith(".bmp") else (
            _load_image(mask_path, "L") if mask_path and str(mask_path).endswith(".png") else None
        )
        annotations = [self._normalize_annotation(annotation) for annotation in coco["annotations"]]
        return {
            "mode": "defect",
            "sample_id": sample_id,
            "image": image,
            "image_path": image_info["file_name"],
            "split": row["split"],
            "group_id": row["group_id"],
            "task_mask": {"ring": 0, "defect": 1, "classification": 0},
            "semantic_mask": semantic_mask,
            "annotations": annotations,
            "ignore_bboxes": list(image_info.get("ignore_bboxes", [])),
            "metadata": row,
        }

    def _load_classification(self, sample_id: str) -> dict[str, Any]:
        row = self.crop_by_path[sample_id]
        image = _load_image(row["crop_path"], "RGB")
        return {
            "mode": "classification",
            "sample_id": sample_id,
            "image": image,
            "image_path": row["crop_path"],
            "split": row["split"],
            "group_id": row["group_id"],
            "task_mask": {"ring": 0, "defect": 0, "classification": 1},
            "label": {
                "category": row["category"],
                "subtype": row["subtype"],
                "attributes": _attributes_from_text(row["attributes"]),
            },
            "metadata": row,
        }

    def _load_bridge(self, sample_id: str) -> dict[str, Any]:
        row = self.manifest_by_sample[sample_id]
        ring_row = self.ring_by_sample[sample_id]
        coco = self.coco_by_sample[sample_id]
        image_info = coco["image"]
        ring_image = _load_image(ring_row["model_image_path"], "RGB")
        source_width = int(image_info["width"])
        source_height = int(image_info["height"])
        target_width = int(ring_row["target_width"])
        target_height = int(ring_row["target_height"])
        sx = target_width / max(1, source_width)
        sy = target_height / max(1, source_height)
        annotations = [self._normalize_annotation(annotation, sx=sx, sy=sy) for annotation in coco["annotations"]]
        semantic_mask = None
        mask_path = image_info.get("unified_semantic_mask") or image_info.get("source_semantic_mask")
        if mask_path:
            loaded = _load_image(mask_path, "RGB" if str(mask_path).endswith(".bmp") else "L")
            if loaded.shape[:2] != (target_height, target_width):
                loaded = _resize_array(loaded, target_width, target_height)
            semantic_mask = loaded
        return {
            "mode": "bridge",
            "sample_id": sample_id,
            "image": ring_image,
            "image_path": ring_row["model_image_path"],
            "split": row["split"],
            "group_id": row["group_id"],
            "task_mask": {"ring": 1, "defect": 1, "classification": 0},
            "ring_targets": {
                "skeleton": _load_image(ring_row["skeleton_path"], "L") > 0,
                "boundary": _load_image(ring_row["boundary_path"], "L") > 0,
                "distance": _load_image(ring_row["distance_path"], "I").astype(np.float32) / 65535.0,
                "instance": _load_image(ring_row["instance_path"], "I").astype(np.uint16),
                "valid": _load_image(ring_row["valid_path"], "L") > 0,
            },
            "defect": {
                "annotations": annotations,
                "semantic_mask": semantic_mask,
                "ignore_bboxes": list(image_info.get("ignore_bboxes", [])),
            },
            "metadata": row,
        }

    def _normalize_annotation(self, annotation: dict[str, Any], sx: float = 1.0, sy: float = 1.0) -> dict[str, Any]:
        result = dict(annotation)
        if "bbox" in result:
            result["bbox"] = [round(value, 4) for value in _scale_bbox([float(value) for value in result["bbox"]], sx, sy)]
        if "segmentation" in result:
            result["segmentation"] = _scale_segmentation(result["segmentation"], sx, sy)
        if "area" in result:
            result["area"] = float(result["area"]) * sx * sy
        result["attributes"] = list(result.get("attributes", []))
        return result


__all__ = ["WoodCatalog", "WoodDataset", "LoaderRecord"]
