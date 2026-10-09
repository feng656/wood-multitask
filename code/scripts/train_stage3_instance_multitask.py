"""Train the complete stage-3 shared instance multitask model on the remote GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wood_data.loaders import LoaderRecord, WoodCatalog  # noqa: E402
from wood_data.instance_multitask import (  # noqa: E402
    SharedInstanceMultitaskModel,
    load_semantic_encoder_checkpoint,
)


PRIMARY_CATEGORIES = ["normal", "knot", "crack"]
CATEGORY_TO_INDEX = {name: index for index, name in enumerate(PRIMARY_CATEGORIES)}
INDEX_TO_CATEGORY = {index: name for name, index in CATEGORY_TO_INDEX.items()}
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
CACHE_VERSION = "instance_v3_two_class_ignore_excluded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=["resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--anchor-profile", choices=["default", "small_crack"], default="default")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=512)
    parser.add_argument("--ring-batch", type=int, default=16)
    parser.add_argument("--detection-batch", type=int, default=4)
    parser.add_argument("--classification-batch", type=int, default=64)
    parser.add_argument("--bridge-batch", type=int, default=2)
    parser.add_argument("--task-ratios", default="ring:1,defect:6,classification:4,bridge:1")
    parser.add_argument("--score-weights", default="ring:1,defect_box:1,defect_mask:1,classification:1,bridge_ring:0.5,bridge_box:0.5,bridge_mask:0.5")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wood_stage3_instance_cache"))
    parser.add_argument("--prewarm-cache", action="store_true")
    parser.add_argument("--balanced-defect", action="store_true")
    parser.add_argument("--balanced-bridge", action="store_true")
    parser.add_argument("--max-train-samples-per-task", type=int, default=0)
    parser.add_argument("--max-val-samples-per-task", type=int, default=0)
    parser.add_argument("--init-semantic-checkpoint", type=Path, default=None)
    parser.add_argument("--init-instance-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_task_ratios(value: str) -> list[str]:
    weights: dict[str, int] = {}
    for item in value.split(","):
        task, count_value = item.split(":", 1)
        task = task.strip()
        count = int(count_value)
        if task not in {"ring", "defect", "classification", "bridge"} or count <= 0:
            raise ValueError(f"invalid task ratio item: {item}")
        weights[task] = count
    if not weights:
        raise ValueError("task schedule must not be empty")
    total = sum(weights.values())
    current = {task: 0 for task in weights}
    schedule: list[str] = []
    for _ in range(total):
        for task, weight in weights.items():
            current[task] += weight
        selected = max(current, key=current.get)
        schedule.append(selected)
        current[selected] -= total
    return schedule


def parse_score_weights(value: str) -> dict[str, float]:
    defaults = {
        "ring": 1.0,
        "defect_box": 1.0,
        "defect_mask": 1.0,
        "classification": 1.0,
        "bridge_ring": 0.5,
        "bridge_box": 0.5,
        "bridge_mask": 0.5,
    }
    weights = defaults.copy()
    for item in value.split(","):
        if not item.strip():
            continue
        key, weight_value = item.split(":", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"invalid validation score weight: {key}")
        weights[key] = float(weight_value)
    return weights


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resample = Image.Resampling.BOX if max(mask.shape) > size else Image.Resampling.NEAREST
    return np.asarray(pil.resize((size, size), resample), dtype=np.uint8) > 0


def resize_float(array: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def polygon_mask(polygons: list[list[float]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.asarray(canvas, dtype=np.uint8)


def decode_uncompressed_rle(segmentation: dict[str, Any], width: int, height: int) -> np.ndarray | None:
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


def bbox_mask(bbox: Iterable[float], width: int, height: int) -> np.ndarray:
    x, y, box_width, box_height = [float(value) for value in bbox]
    x1 = max(0, min(width, int(math.floor(x))))
    y1 = max(0, min(height, int(math.floor(y))))
    x2 = max(0, min(width, int(math.ceil(x + box_width))))
    y2 = max(0, min(height, int(math.ceil(y + box_height))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def build_instance_target(
    annotations: list[dict[str, Any]],
    source_shape: tuple[int, int],
    size: int,
) -> dict[str, np.ndarray]:
    source_height, source_width = source_shape
    boxes: list[list[float]] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []
    areas: list[float] = []
    crowds: list[int] = []
    sources: list[str] = []

    for annotation in annotations:
        label = int(annotation.get("category_id", 0))
        if label <= 0 or label >= len(PRIMARY_CATEGORIES):
            continue
        segmentation = annotation.get("segmentation")
        source = "bbox"
        mask: np.ndarray | None = None
        if isinstance(segmentation, list) and any(
            isinstance(polygon, list) and len(polygon) >= 6 for polygon in segmentation
        ):
            mask = polygon_mask(segmentation, source_width, source_height)
            source = "polygon"
        elif isinstance(segmentation, dict):
            mask = decode_uncompressed_rle(segmentation, source_width, source_height)
            source = "rle" if mask is not None else "bbox"
        if mask is None or not mask.any():
            mask = bbox_mask(annotation.get("bbox", [0, 0, 0, 0]), source_width, source_height)
            source = "bbox"
        resized = resize_bool(mask, size).astype(np.uint8)
        ys, xs = np.nonzero(resized)
        if not len(xs):
            continue
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(label)
        masks.append(resized)
        areas.append(float(resized.sum()))
        crowds.append(int(annotation.get("iscrowd", 0)))
        sources.append(source)

    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=np.int64),
        "masks": np.asarray(masks, dtype=np.uint8).reshape(-1, size, size),
        "area": np.asarray(areas, dtype=np.float32),
        "iscrowd": np.asarray(crowds, dtype=np.int64),
        "mask_sources": np.asarray(sources, dtype="U8"),
    }


class InstanceTaskDataset:
    def __init__(
        self,
        catalog: WoodCatalog,
        task: str,
        records: list[LoaderRecord],
        imgsz: int,
        cache_dir: Path | None,
    ) -> None:
        self.catalog = catalog
        self.task = task
        self.records = records
        self.imgsz = imgsz
        self.cache_dir = cache_dir / task if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, sample_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{CACHE_VERSION}:{self.task}:{self.imgsz}:{sample_id}"
        return self.cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.npz"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self._cache_path(record.sample_id)
        if path is not None and path.exists():
            try:
                with np.load(path) as data:
                    return {key: data[key] for key in data.files}
            except Exception:
                path.unlink(missing_ok=True)
        item = self._build_item(record)
        if path is not None:
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **item)
            temporary.replace(path)
        return item

    def _build_item(self, record: LoaderRecord) -> dict[str, np.ndarray]:
        sample = self.catalog.load(self.task, record.sample_id)
        item: dict[str, np.ndarray] = {
            "sample_id": np.asarray(sample["sample_id"]),
            "image": resize_rgb(np.asarray(sample["image"]), self.imgsz),
        }
        if self.task in {"ring", "bridge"}:
            target = sample["targets"] if self.task == "ring" else sample["ring_targets"]
            item["ring_boundary"] = resize_bool(np.asarray(target["boundary"]), self.imgsz).astype(np.float32)
            item["ring_distance"] = resize_float(np.asarray(target["distance"]), self.imgsz)
            item["ring_valid"] = resize_bool(np.asarray(target["valid"]), self.imgsz).astype(np.float32)
        if self.task in {"defect", "bridge"}:
            annotations = sample["annotations"] if self.task == "defect" else sample["defect"]["annotations"]
            target = build_instance_target(annotations, tuple(sample["image"].shape[:2]), self.imgsz)
            item.update({f"target_{key}": value for key, value in target.items()})
        if self.task == "classification":
            item["class_label"] = np.asarray(
                CATEGORY_TO_INDEX[str(sample["label"]["category"])], dtype=np.int64
            )
        return item

    def warm_cache(self) -> Counter[str]:
        sources: Counter[str] = Counter()
        for index in range(len(self)):
            item = self[index]
            for source in item.get("target_mask_sources", []):
                sources[str(source)] += 1
            if (index + 1) % 500 == 0 or index + 1 == len(self):
                print(json.dumps({"cache_task": self.task, "cached": index + 1, "total": len(self)}), flush=True)
        return sources


def image_tensor(image: np.ndarray) -> "torch.Tensor":
    import torch

    return torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float() / 255.0


def collate_instance_task(task: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch: dict[str, Any] = {
        "task": task,
        "sample_ids": [str(item["sample_id"]) for item in items],
        "images": [image_tensor(item["image"]) for item in items],
    }
    if task in {"ring", "bridge"}:
        for key in ("ring_boundary", "ring_distance", "ring_valid"):
            batch[key] = torch.from_numpy(np.stack([item[key] for item in items])).float()
    if task in {"defect", "bridge"}:
        targets = []
        for item in items:
            targets.append(
                {
                    "boxes": torch.from_numpy(item["target_boxes"]).float(),
                    "labels": torch.from_numpy(item["target_labels"]).long(),
                    "masks": torch.from_numpy(item["target_masks"]).to(torch.uint8),
                    "area": torch.from_numpy(item["target_area"]).float(),
                    "iscrowd": torch.from_numpy(item["target_iscrowd"]).long(),
                }
            )
        batch["targets"] = targets
    if task == "classification":
        batch["class_labels"] = torch.from_numpy(
            np.asarray([int(item["class_label"]) for item in items], dtype=np.int64)
        )
    return batch


def detection_sample_labels(dataset: InstanceTaskDataset) -> list[list[int]]:
    labels_by_record: list[list[int]] = []
    for record in dataset.records:
        annotations = dataset.catalog.coco_by_sample[record.sample_id]["annotations"]
        labels = [
            int(annotation.get("category_id", 0))
            for annotation in annotations
            if 0 < int(annotation.get("category_id", 0)) < len(PRIMARY_CATEGORIES)
        ]
        labels_by_record.append(labels)
    return labels_by_record


def make_loader(
    dataset: InstanceTaskDataset,
    batch_size: int,
    workers: int,
    train: bool,
    balanced: bool,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    sampler = None
    if balanced and dataset.records:
        if dataset.task == "classification":
            sample_labels = [
                [CATEGORY_TO_INDEX[str(dataset.catalog.crop_by_path[record.sample_id]["category"])]]
                for record in dataset.records
            ]
        elif dataset.task in {"defect", "bridge"}:
            sample_labels = detection_sample_labels(dataset)
        else:
            sample_labels = []
        if sample_labels:
            flat_labels = [label for labels in sample_labels for label in labels]
            counts = np.bincount(flat_labels, minlength=len(PRIMARY_CATEGORIES)).astype(np.float64)
            class_weights = np.zeros_like(counts)
            class_weights[counts > 0] = 1.0 / counts[counts > 0]
            empty_count = max(1, sum(1 for labels in sample_labels if not labels))
            empty_weight = 0.25 / empty_count
            sample_weights = [
                max(float(class_weights[label]) for label in labels) if labels else empty_weight
                for labels in sample_labels
            ]
            generator = torch.Generator().manual_seed(seed)
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights), replacement=True, generator=generator
            )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda items: collate_instance_task(dataset.task, items),
    )


def limit_records(records: list[LoaderRecord], limit: int) -> list[LoaderRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    indices = np.linspace(0, len(records) - 1, num=limit, dtype=int)
    return [records[int(index)] for index in indices]


def build_loaders(
    catalog: WoodCatalog,
    split: str,
    imgsz: int,
    workers: int,
    cache_dir: Path | None,
    limit: int,
    batch_sizes: dict[str, int],
    seed: int,
    balanced_detection_tasks: set[str] | None = None,
) -> dict[str, Any]:
    record_sets = {
        "ring": list(catalog.ring_samples(splits={split}).records),
        "defect": list(catalog.defect_samples(splits={split}).records),
        "classification": list(catalog.classification_samples(splits={split}).records),
        "bridge": list(catalog.bridge_samples(splits={split}).records),
    }
    for task in ("defect", "bridge"):
        record_sets[task] = [
            record
            for record in record_sets[task]
            if not catalog.coco_by_sample[record.sample_id]["image"].get("ignore_bboxes")
        ]
    loaders = {}
    balanced_detection_tasks = balanced_detection_tasks or set()
    for task, records in record_sets.items():
        records = limit_records(records, limit)
        if not records:
            continue
        dataset = InstanceTaskDataset(
            catalog, task, records, imgsz, cache_dir / split if cache_dir is not None else None
        )
        loaders[task] = make_loader(
            dataset, batch_sizes[task], workers, split == "train",
            split == "train" and (task == "classification" or task in balanced_detection_tasks), seed
        )
    return loaders


def masked_ring_losses(outputs: dict[str, Any], batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    target = batch["ring_boundary"].to(device).unsqueeze(1)
    distance = batch["ring_distance"].to(device).unsqueeze(1)
    valid = batch["ring_valid"].to(device).unsqueeze(1)
    positive_weight = torch.tensor([8.0], device=device)
    bce = F.binary_cross_entropy_with_logits(
        outputs["ring_boundary"], target, reduction="none", pos_weight=positive_weight
    )
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probability = outputs["ring_boundary"].sigmoid() * valid
    target_valid = target * valid
    intersection = (probability * target_valid).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target_valid.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    distance_loss = (torch.abs(outputs["ring_distance"] - distance) * valid).sum() / valid.sum().clamp_min(1.0)
    return {"ring_bce": bce, "ring_dice": dice, "ring_distance": 0.5 * distance_loss}


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in INDEX_TO_CATEGORY.items():
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def ring_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    distance_predictions: list[np.ndarray],
    distance_targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> dict[str, float]:
    dice_scores: list[float] = []
    distance_errors: list[float] = []
    for prediction, target, distance_prediction, distance_target, valid in zip(
        predictions, targets, distance_predictions, distance_targets, valid_masks
    ):
        prediction = prediction.astype(bool)
        target = target.astype(bool)
        valid = valid.astype(bool)
        denominator = int((prediction & valid).sum() + (target & valid).sum())
        intersection = int((prediction & target & valid).sum())
        dice_scores.append(2.0 * intersection / denominator if denominator else 0.0)
        if valid.any():
            distance_errors.append(float(np.abs(distance_prediction - distance_target)[valid].mean()))
    return {
        "boundary_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "distance_mae": float(np.mean(distance_errors)) if distance_errors else 0.0,
    }


def interpolated_ap(records: list[tuple[float, int]], ground_truth_count: int) -> tuple[float, float]:
    if ground_truth_count <= 0:
        return 0.0, 0.0
    records = sorted(records, key=lambda item: item[0], reverse=True)
    if not records:
        return 0.0, 0.0
    true_positive = np.cumsum([item[1] for item in records], dtype=np.float64)
    false_positive = np.cumsum([1 - item[1] for item in records], dtype=np.float64)
    recall = true_positive / ground_truth_count
    precision = true_positive / np.maximum(1.0, true_positive + false_positive)
    ap = np.mean([
        float(precision[recall >= level].max()) if np.any(recall >= level) else 0.0
        for level in np.linspace(0.0, 1.0, 101)
    ])
    return float(ap), float(recall[-1])


class DetectionMetrics:
    def __init__(self, score_threshold: float = 0.25) -> None:
        self.score_threshold = score_threshold
        self.gt_counts: Counter[int] = Counter()
        self.records: dict[str, dict[float, dict[int, list[tuple[float, int]]]]] = {
            kind: {threshold: defaultdict(list) for threshold in IOU_THRESHOLDS}
            for kind in ("box", "mask")
        }
        self.mask_quality_sum: dict[str, Counter[int]] = {"iou": Counter(), "dice": Counter()}
        self.mask_quality_count: Counter[int] = Counter()

    @staticmethod
    def _box_iou(predicted: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        import torch
        from torchvision.ops import box_iou

        if predicted.numel() == 0 or target.numel() == 0:
            return torch.zeros((len(predicted), len(target)), dtype=torch.float32)
        return box_iou(predicted.float(), target.float())

    @staticmethod
    def _mask_iou(predicted: "torch.Tensor", target: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        if predicted.numel() == 0 or target.numel() == 0:
            empty = torch.zeros((len(predicted), len(target)), dtype=torch.float32)
            return empty, empty
        predicted_flat = predicted.reshape(len(predicted), -1).float()
        target_flat = target.reshape(len(target), -1).float()
        intersection = predicted_flat @ target_flat.transpose(0, 1)
        predicted_area = predicted_flat.sum(dim=1, keepdim=True)
        target_area = target_flat.sum(dim=1).unsqueeze(0)
        union = predicted_area + target_area - intersection
        iou = intersection / union.clamp_min(1.0)
        dice = 2.0 * intersection / (predicted_area + target_area).clamp_min(1.0)
        return iou, dice

    @staticmethod
    def _matches(iou: np.ndarray, threshold: float) -> list[int]:
        matched: set[int] = set()
        result: list[int] = []
        for prediction_index in range(iou.shape[0]):
            candidates = np.argsort(iou[prediction_index])[::-1]
            selected = next(
                (int(index) for index in candidates if index not in matched and iou[prediction_index, index] >= threshold),
                None,
            )
            if selected is None:
                result.append(0)
            else:
                matched.add(selected)
                result.append(1)
        return result

    def update(self, prediction: dict[str, Any], target: dict[str, Any]) -> None:
        import torch

        predicted_labels = prediction["labels"].detach().cpu()
        predicted_scores = prediction["scores"].detach().cpu()
        predicted_boxes = prediction["boxes"].detach().cpu()
        predicted_masks = (prediction["masks"].detach().cpu()[:, 0] >= 0.5).to(torch.uint8)
        target_labels = target["labels"].detach().cpu()
        target_boxes = target["boxes"].detach().cpu()
        target_masks = target["masks"].detach().cpu().to(torch.uint8)

        for class_index in range(1, len(PRIMARY_CATEGORIES)):
            prediction_indices = torch.where(predicted_labels == class_index)[0]
            target_indices = torch.where(target_labels == class_index)[0]
            scores = predicted_scores[prediction_indices]
            order = torch.argsort(scores, descending=True)
            prediction_indices = prediction_indices[order]
            scores = scores[order]
            class_target_boxes = target_boxes[target_indices]
            class_target_masks = target_masks[target_indices]
            self.gt_counts[class_index] += len(target_indices)

            box_iou = self._box_iou(predicted_boxes[prediction_indices], class_target_boxes).numpy()
            mask_iou, mask_dice = self._mask_iou(
                predicted_masks[prediction_indices], class_target_masks
            )
            mask_iou_array = mask_iou.numpy()
            for threshold in IOU_THRESHOLDS:
                for score, matched in zip(scores.tolist(), self._matches(box_iou, threshold)):
                    self.records["box"][threshold][class_index].append((float(score), matched))
                for score, matched in zip(scores.tolist(), self._matches(mask_iou_array, threshold)):
                    self.records["mask"][threshold][class_index].append((float(score), matched))

            retained = torch.where(scores >= self.score_threshold)[0]
            for target_position in range(len(target_indices)):
                self.mask_quality_count[class_index] += 1
                if len(retained):
                    best_position = retained[mask_iou[retained, target_position].argmax()]
                    self.mask_quality_sum["iou"][class_index] += float(mask_iou[best_position, target_position])
                    self.mask_quality_sum["dice"][class_index] += float(mask_dice[best_position, target_position])

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {"score_threshold_for_dice": self.score_threshold}
        for kind in ("box", "mask"):
            class_metrics: dict[str, Any] = {}
            all_aps: list[float] = []
            ap50_values: list[float] = []
            recall50_values: list[float] = []
            for class_index in range(1, len(PRIMARY_CATEGORIES)):
                threshold_aps = []
                recall50 = 0.0
                for threshold in IOU_THRESHOLDS:
                    ap, recall = interpolated_ap(
                        self.records[kind][threshold][class_index], self.gt_counts[class_index]
                    )
                    threshold_aps.append(ap)
                    if threshold == 0.5:
                        recall50 = recall
                if self.gt_counts[class_index] > 0:
                    all_aps.append(float(np.mean(threshold_aps)))
                    ap50_values.append(threshold_aps[0])
                    recall50_values.append(recall50)
                class_metrics[INDEX_TO_CATEGORY[class_index]] = {
                    "support": self.gt_counts[class_index],
                    "ap50_95": float(np.mean(threshold_aps)),
                    "ap50": threshold_aps[0],
                    "recall50": recall50,
                }
            result[kind] = {
                "map50_95": float(np.mean(all_aps)) if all_aps else 0.0,
                "map50": float(np.mean(ap50_values)) if ap50_values else 0.0,
                "macro_recall50": float(np.mean(recall50_values)) if recall50_values else 0.0,
                "per_class": class_metrics,
            }
        quality_classes = []
        per_class_quality = {}
        for class_index in range(1, len(PRIMARY_CATEGORIES)):
            count = self.mask_quality_count[class_index]
            iou = self.mask_quality_sum["iou"][class_index] / count if count else 0.0
            dice = self.mask_quality_sum["dice"][class_index] / count if count else 0.0
            if count:
                quality_classes.append((iou, dice))
            per_class_quality[INDEX_TO_CATEGORY[class_index]] = {
                "support": count, "gt_best_mask_iou": iou, "gt_best_mask_dice": dice
            }
        result["mask_quality"] = {
            "mean_gt_best_iou": float(np.mean([item[0] for item in quality_classes])) if quality_classes else 0.0,
            "mean_gt_best_dice": float(np.mean([item[1] for item in quality_classes])) if quality_classes else 0.0,
            "per_class": per_class_quality,
        }
        return result


def move_targets(targets: list[dict[str, Any]], device: Any) -> list[dict[str, Any]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def train_epoch(
    model: SharedInstanceMultitaskModel,
    loaders: dict[str, Any],
    optimizer: Any,
    scaler: Any,
    device: Any,
    steps_per_epoch: int,
    schedule: list[str],
    freeze_batch_norm_stats: bool = False,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.train()
    if freeze_batch_norm_stats:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    active_schedule = [task for task in schedule if task in loaders]
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    component_totals: dict[str, float] = defaultdict(float)

    for step in range(steps_per_epoch):
        task = active_schedule[step % len(active_schedule)]
        try:
            batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(loaders[task])
            batch = next(iterators[task])
        images = [image.to(device, non_blocking=True) for image in batch["images"]]
        targets = move_targets(batch["targets"], device) if "targets" in batch else None
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(
                images,
                targets=targets,
                ring=task in {"ring", "bridge"},
                detection=task in {"defect", "bridge"},
                classification=task == "classification",
            )
            components: dict[str, Any] = {}
            if task in {"ring", "bridge"}:
                components.update(masked_ring_losses(outputs, batch, device))
            if task in {"defect", "bridge"}:
                components.update(outputs["detection_losses"])
            if task == "classification":
                components["classification"] = F.cross_entropy(
                    outputs["classification_logits"], batch["class_labels"].to(device)
                )
            loss = torch.stack([value for value in components.values()]).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {task} loss at step {step}: {components}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        totals[task] += float(loss.item())
        counts[task] += 1
        for name, value in components.items():
            component_totals[f"{task}/{name}"] += float(value.detach().item())

    result = {f"{task}_loss": totals[task] / max(1, counts[task]) for task in counts}
    result.update({name: value / max(1, counts[name.split("/", 1)[0]]) for name, value in component_totals.items()})
    return result


def evaluate_loader(
    model: SharedInstanceMultitaskModel,
    loader: Any,
    device: Any,
    task: str,
) -> dict[str, Any]:
    import torch

    model.eval()
    classification_confusion = np.zeros((len(PRIMARY_CATEGORIES), len(PRIMARY_CATEGORIES)), dtype=np.int64)
    detector_metrics = DetectionMetrics()
    boundary_predictions: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    distance_predictions: list[np.ndarray] = []
    distance_targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = [image.to(device, non_blocking=True) for image in batch["images"]]
            targets = move_targets(batch["targets"], device) if "targets" in batch else None
            outputs = model(
                images,
                targets=targets,
                ring=task in {"ring", "bridge"},
                detection=task in {"defect", "bridge"},
                classification=task == "classification",
            )
            if task == "classification":
                predictions = outputs["classification_logits"].argmax(dim=1).cpu().numpy()
                for target, prediction in zip(batch["class_labels"].numpy(), predictions):
                    classification_confusion[int(target), int(prediction)] += 1
            if task in {"defect", "bridge"}:
                assert targets is not None
                for prediction, target in zip(outputs["detections"], targets):
                    detector_metrics.update(prediction, target)
            if task in {"ring", "bridge"}:
                boundary_predictions.extend((outputs["ring_boundary"].sigmoid().cpu().numpy()[:, 0] >= 0.5))
                boundary_targets.extend(batch["ring_boundary"].numpy() >= 0.5)
                distance_predictions.extend(outputs["ring_distance"].cpu().numpy()[:, 0])
                distance_targets.extend(batch["ring_distance"].numpy())
                valid_masks.extend(batch["ring_valid"].numpy() >= 0.5)

    if task == "classification":
        return classification_metrics(classification_confusion)
    result: dict[str, Any] = {}
    if task in {"ring", "bridge"}:
        result["ring"] = ring_metrics(
            boundary_predictions, boundary_targets, distance_predictions, distance_targets, valid_masks
        )
    if task in {"defect", "bridge"}:
        result["detection"] = detector_metrics.compute()
    return result


def validation_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    ring = metrics.get("ring", {}).get("ring", {})
    defect = metrics.get("defect", {}).get("detection", {})
    classification = metrics.get("classification", {})
    bridge = metrics.get("bridge", {})
    bridge_detection = bridge.get("detection", {})
    return (
        weights["ring"] * float(ring.get("boundary_dice", 0.0))
        + weights["defect_box"] * float(defect.get("box", {}).get("map50_95", 0.0))
        + weights["defect_mask"] * float(defect.get("mask", {}).get("map50_95", 0.0))
        + weights["classification"] * float(classification.get("macro_f1", 0.0))
        + weights["bridge_ring"] * float(bridge.get("ring", {}).get("boundary_dice", 0.0))
        + weights["bridge_box"] * float(bridge_detection.get("box", {}).get("map50_95", 0.0))
        + weights["bridge_mask"] * float(bridge_detection.get("mask", {}).get("map50_95", 0.0))
    )


def load_instance_model_checkpoint(model: Any, checkpoint_path: Path, device: Any) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model", checkpoint)
    destination = model.state_dict()
    transferred = {}
    skipped_shape = []
    skipped_missing = []
    for name, value in source.items():
        if name not in destination:
            skipped_missing.append(name)
            continue
        if destination[name].shape != value.shape:
            skipped_shape.append({
                "name": name,
                "source_shape": list(value.shape),
                "target_shape": list(destination[name].shape),
            })
            continue
        transferred[name] = value
    incompatible = model.load_state_dict(transferred, strict=False)
    return {
        "source": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "transferred_tensors": len(transferred),
        "skipped_shape_mismatch": skipped_shape,
        "skipped_missing": skipped_missing,
        "missing_after_partial_load": len(incompatible.missing_keys),
        "unexpected_after_partial_load": incompatible.unexpected_keys,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    metrics: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    import torch

    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.cpu_threads)

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if device.type != "cuda" and args.max_train_samples_per_task <= 0:
        raise RuntimeError("formal stage-3 instance training requires CUDA")

    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        or project_root / "outputs" / "stage3_multitask" / "runs" / f"{args.backbone}_instance_final100"
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.exist_ok or args.resume):
        raise FileExistsError(f"non-empty output directory requires --exist-ok or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    catalog = WoodCatalog(project_root=project_root)
    batch_sizes = {
        "ring": args.ring_batch,
        "defect": args.detection_batch,
        "classification": args.classification_batch,
        "bridge": args.bridge_batch,
    }
    train_loaders = build_loaders(
        catalog, "train", args.imgsz, args.workers, cache_dir,
        args.max_train_samples_per_task, batch_sizes, args.seed,
        {task for task, enabled in {"defect": args.balanced_defect, "bridge": args.balanced_bridge}.items() if enabled}
    )
    val_loaders = build_loaders(
        catalog, "val", args.imgsz, args.workers, cache_dir,
        args.max_val_samples_per_task, batch_sizes, args.seed
    )
    test_loaders = build_loaders(
        catalog, "test", args.imgsz, args.workers, cache_dir,
        args.max_val_samples_per_task, batch_sizes, args.seed
    )
    schedule = parse_task_ratios(args.task_ratios)
    score_weights = parse_score_weights(args.score_weights)
    missing_tasks = set(schedule) - set(train_loaders)
    if missing_tasks:
        raise RuntimeError(f"missing formal training tasks: {sorted(missing_tasks)}")

    cache_audit: Counter[str] = Counter()
    if args.prewarm_cache:
        for split, loaders in (("train", train_loaders), ("val", val_loaders)):
            for task, loader in loaders.items():
                print(json.dumps({"cache_split": split, "cache_task": task, "event": "start"}), flush=True)
                cache_audit.update(loader.dataset.warm_cache())

    model = SharedInstanceMultitaskModel(args.backbone, args.imgsz, anchor_profile=args.anchor_profile).to(device)
    initialization: dict[str, Any] | None = None
    if args.init_instance_checkpoint is not None and args.resume is None:
        initialization = load_instance_model_checkpoint(model, args.init_instance_checkpoint.resolve(), device)
    else:
        init_path = args.init_semantic_checkpoint
        if init_path is None:
            candidate = project_root / "outputs" / "stage3_multitask" / "runs" / "resnet34_shared_final100" / "best.pt"
            init_path = candidate if candidate.exists() else None
        if init_path is not None and args.resume is None:
            initialization = load_semantic_encoder_checkpoint(model, init_path.resolve())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 1
    best_score = -math.inf
    best_metrics: dict[str, Any] = {}
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("metrics"):
            best_metrics = checkpoint["metrics"]
            best_score = validation_score(best_metrics, score_weights)

    config = {
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "architecture": "shared_resnet_fpn_mask_rcnn_ring_roi_classifier",
        "backbone": args.backbone,
        "anchor_profile": args.anchor_profile,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_sizes": batch_sizes,
        "task_ratios": args.task_ratios,
        "score_weights": score_weights,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "balanced_defect": args.balanced_defect,
        "balanced_bridge": args.balanced_bridge,
        "eval_interval": args.eval_interval,
        "device": str(device),
        "seed": args.seed,
        "max_train_samples_per_task": args.max_train_samples_per_task,
        "max_val_samples_per_task": args.max_val_samples_per_task,
        "train_records": {task: len(loader.dataset) for task, loader in train_loaders.items()},
        "val_records": {task: len(loader.dataset) for task, loader in val_loaders.items()},
        "test_records": {task: len(loader.dataset) for task, loader in test_loaders.items()},
        "initialization": initialization,
        "cache_audit": dict(cache_audit),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False), flush=True)

    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history: list[dict[str, Any]] = []
    if history_path.exists() and args.resume is not None:
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(
            model, train_loaders, optimizer, scaler, device, args.steps_per_epoch, schedule
        )
        should_evaluate = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        metrics = (
            {task: evaluate_loader(model, loader, device, task) for task, loader in val_loaders.items()}
            if should_evaluate else None
        )
        score = validation_score(metrics, score_weights) if metrics is not None else None
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_losses": losses,
            "val": metrics,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_checkpoint(latest_path, epoch, model, optimizer, scheduler, scaler, metrics, config)
        if metrics is not None and score is not None and score >= best_score:
            best_score = score
            best_metrics = metrics
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, metrics, config)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    model.eval()

    print(json.dumps({"event": "final_test_eval", "start": True}, ensure_ascii=False), flush=True)
    test_metrics = {}
    for task, loader in test_loaders.items():
        test_metrics[task] = evaluate_loader(model, loader, device, task)
    test_score = validation_score(test_metrics, score_weights)

    results = {
        "config": config,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_score": best_score,
        "best_val_metrics": best_checkpoint["metrics"],
        "best_checkpoint": str(best_path),
        "test_score": test_score,
        "test_metrics": test_metrics,
        "history": history,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()



