"""Train the stage-3 shared encoder multitask baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wood_data.defects import CATEGORY_IDS
from wood_data.loaders import WoodCatalog, LoaderRecord  # type: ignore
from wood_data.multitask import build_shared_multitask_model


PRIMARY_CATEGORIES = ["normal", "knot", "crack"]
CATEGORY_TO_INDEX = {name: index for index, name in enumerate(PRIMARY_CATEGORIES)}
INDEX_TO_CATEGORY = {index: name for name, index in CATEGORY_TO_INDEX.items()}
IGNORE_INDEX = 255
CACHE_VERSION = "two_class_v1_ignore255"


@dataclass(frozen=True)
class TaskBatch:
    task: str
    images: np.ndarray
    ring_boundary: np.ndarray | None = None
    ring_distance: np.ndarray | None = None
    ring_valid: np.ndarray | None = None
    defect_mask: np.ndarray | None = None
    class_labels: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples-per-task", type=int, default=0)
    parser.add_argument("--max-val-samples-per-task", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("WOOD_STAGE3_CACHE", "/tmp/wood_stage3_cache")))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prewarm-cache", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(cpu_threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def encode_cache_image(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def decode_cache_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(pil.resize((size, size), resample=Image.Resampling.NEAREST), dtype=np.uint8) > 0


def resize_float(array: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(pil.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(pil.resize((size, size), resample=Image.Resampling.NEAREST), dtype=np.uint8)


def apply_ignore_bboxes(mask: np.ndarray, ignored: list[dict[str, Any]]) -> np.ndarray:
    height, width = mask.shape
    result = mask.copy()
    for region in ignored:
        x, y, box_width, box_height = map(float, region.get("bbox", []))
        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(width, int(math.ceil(x + box_width)))
        y2 = min(height, int(math.ceil(y + box_height)))
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = IGNORE_INDEX
    return result


def polygon_to_mask(points: list[float], width: int, height: int, value: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    coords = [(points[index], points[index + 1]) for index in range(0, len(points), 2)]
    if len(coords) >= 3:
        draw.polygon(coords, fill=int(value))
    return np.asarray(canvas, dtype=np.uint8)


def annotation_mask(annotations: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        category_id = int(annotation.get("category_id", 0))
        if category_id <= 0:
            continue
        segmentation = annotation.get("segmentation")
        painted = None
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6:
                    continue
                polygon_mask = polygon_to_mask([float(value) for value in polygon], width, height, category_id)
                painted = polygon_mask if painted is None else np.maximum(painted, polygon_mask)
        if painted is None:
            bbox = annotation.get("bbox", [0, 0, 0, 0])
            x, y, box_width, box_height = [float(value) for value in bbox]
            x1 = max(0, int(math.floor(x)))
            y1 = max(0, int(math.floor(y)))
            x2 = min(width, int(math.ceil(x + box_width)))
            y2 = min(height, int(math.ceil(y + box_height)))
            if x2 > x1 and y2 > y1:
                painted = np.zeros_like(mask)
                painted[y1:y2, x1:x2] = int(category_id)
        if painted is not None:
            mask = np.maximum(mask, painted.astype(np.uint8))
    return mask


def _to_tensor_image(images: list[np.ndarray]) -> "torch.Tensor":
    import torch

    stacked = np.stack(images, axis=0)
    tensor = torch.from_numpy(np.transpose(stacked, (0, 3, 1, 2)).copy())
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    return (tensor.float() - mean) / std


def _to_tensor_batch(masks: list[np.ndarray], dtype: str = "long") -> "torch.Tensor":
    import torch

    stacked = np.stack(masks, axis=0)
    tensor = torch.from_numpy(stacked.copy())
    if dtype == "float":
        return tensor.float()
    return tensor.long()


def collate_task(task: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": task,
        "sample_ids": [item["sample_id"] for item in batch],
        "images": _to_tensor_image([item["image"] for item in batch]),
    }
    if task in {"ring", "bridge"}:
        result["ring_boundary"] = _to_tensor_batch([item["ring_boundary"] for item in batch], dtype="float")
        result["ring_distance"] = _to_tensor_batch([item["ring_distance"] for item in batch], dtype="float")
        result["ring_valid"] = _to_tensor_batch([item["ring_valid"] for item in batch], dtype="float")
    if task in {"defect", "bridge"}:
        result["defect_mask"] = _to_tensor_batch([item["defect_mask"] for item in batch], dtype="long")
    if task == "classification":
        result["class_labels"] = _to_tensor_batch([item["class_labels"] for item in batch], dtype="long")
    return result


class TaskDataset:
    def __init__(
        self,
        catalog: WoodCatalog,
        task: str,
        records: list[LoaderRecord],
        imgsz: int,
        cache_dir: Path | None = None,
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cached = self._read_cache(record.sample_id)
        if cached is not None:
            return cached
        item = self._build_item(record)
        self._write_cache(record.sample_id, item)
        return item

    def _cache_path(self, sample_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(
            f"{CACHE_VERSION}:{self.task}:{self.imgsz}:{sample_id}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def _read_cache(self, sample_id: str) -> dict[str, Any] | None:
        path = self._cache_path(sample_id)
        if path is None or not path.exists():
            return None
        try:
            with np.load(path) as data:
                item: dict[str, Any] = {
                    "sample_id": sample_id,
                    "image": decode_cache_image(data["image"]),
                }
                if "ring_boundary" in data:
                    item["ring_boundary"] = data["ring_boundary"].astype(np.float32)
                    item["ring_distance"] = data["ring_distance"].astype(np.float32)
                    item["ring_valid"] = data["ring_valid"].astype(np.float32)
                if "defect_mask" in data:
                    item["defect_mask"] = data["defect_mask"].astype(np.uint8)
                if "class_labels" in data:
                    item["class_labels"] = np.asarray(int(data["class_labels"]), dtype=np.int64)
                return item
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, sample_id: str, item: dict[str, Any]) -> None:
        path = self._cache_path(sample_id)
        if path is None:
            return
        payload: dict[str, np.ndarray] = {"image": encode_cache_image(item["image"])}
        for key in ("ring_boundary", "ring_distance", "ring_valid", "defect_mask", "class_labels"):
            if key in item:
                payload[key] = np.asarray(item[key])
        temp_path = path.with_suffix(".tmp.npz")
        np.savez(temp_path, **payload)
        temp_path.replace(path)

    def _build_item(self, record: LoaderRecord) -> dict[str, Any]:
        sample = self.catalog.load(self.task, record.sample_id)
        image = resize_rgb(sample["image"], self.imgsz)
        item: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "image": image,
        }
        if self.task in {"ring", "bridge"}:
            ring_source = sample["targets"] if self.task == "ring" else sample["ring_targets"]
            item["ring_boundary"] = resize_bool(np.asarray(ring_source["boundary"], dtype=bool), self.imgsz).astype(np.float32)
            item["ring_distance"] = resize_float(np.asarray(ring_source["distance"], dtype=np.float32), self.imgsz)
            item["ring_valid"] = resize_bool(np.asarray(ring_source["valid"], dtype=bool), self.imgsz).astype(np.float32)
        if self.task in {"defect", "bridge"}:
            annotations = sample["annotations"] if self.task == "defect" else sample["defect"]["annotations"]
            ignored = sample["ignore_bboxes"] if self.task == "defect" else sample["defect"]["ignore_bboxes"]
            target_shape = sample["image"].shape[:2]
            defect_mask = apply_ignore_bboxes(annotation_mask(annotations, target_shape), ignored)
            item["defect_mask"] = resize_mask(defect_mask, self.imgsz)
        if self.task == "classification":
            item["class_labels"] = np.asarray(CATEGORY_TO_INDEX[str(sample["label"]["category"])] , dtype=np.int64)
        return item

    def warm_cache(self) -> None:
        if self.cache_dir is None:
            return
        total = len(self.records)
        for index in range(total):
            _ = self[index]
            if (index + 1) % 500 == 0 or index + 1 == total:
                print(
                    json.dumps(
                        {"cache_task": self.task, "cached": index + 1, "total": total},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


def make_loader(
    dataset: TaskDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    balanced: bool = False,
    seed: int = 42,
) -> "torch.utils.data.DataLoader":
    import torch
    from torch.utils.data import DataLoader

    sampler = None
    if balanced:
        labels = [
            CATEGORY_TO_INDEX[str(dataset.catalog.crop_by_path[record.sample_id]["category"])]
            for record in dataset.records
        ]
        counts = np.bincount(labels, minlength=len(PRIMARY_CATEGORIES)).astype(np.float64)
        class_weights = np.zeros_like(counts)
        present = counts > 0
        class_weights[present] = 1.0 / counts[present]
        sample_weights = torch.as_tensor([class_weights[label] for label in labels], dtype=torch.double)
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": lambda batch: collate_task(dataset.task, batch),
    }
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def masked_bce_with_logits(
    logits: "torch.Tensor",
    target: "torch.Tensor",
    valid: "torch.Tensor",
    pos_weight: float = 1.0,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    positive_weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=positive_weight)
    loss = loss * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def masked_l1_loss(prediction: "torch.Tensor", target: "torch.Tensor", valid: "torch.Tensor") -> "torch.Tensor":
    import torch.nn.functional as F

    loss = F.l1_loss(prediction, target, reduction="none")
    loss = loss * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def binary_dice_loss(logits: "torch.Tensor", target: "torch.Tensor", valid: "torch.Tensor") -> "torch.Tensor":
    import torch

    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def multiclass_dice_loss(logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    import torch

    probability = logits.softmax(dim=1)
    valid = (target != IGNORE_INDEX).float()
    losses: list[torch.Tensor] = []
    for class_index in range(1, logits.shape[1]):
        predicted = probability[:, class_index] * valid
        actual = (target == class_index).float() * valid
        intersection = (predicted * actual).sum(dim=(1, 2))
        denominator = predicted.sum(dim=(1, 2)) + actual.sum(dim=(1, 2))
        present = actual.sum(dim=(1, 2)) > 0
        if present.any():
            dice = (2.0 * intersection[present] + 1.0) / (denominator[present] + 1.0)
            losses.append(1.0 - dice.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    num_classes = confusion.shape[0]
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "per_class_precision": {PRIMARY_CATEGORIES[i]: per_class_precision[i] for i in range(num_classes)},
        "per_class_recall": {PRIMARY_CATEGORIES[i]: per_class_recall[i] for i in range(num_classes)},
        "per_class_f1": {PRIMARY_CATEGORIES[i]: per_class_f1[i] for i in range(num_classes)},
        "confusion_matrix": confusion.tolist(),
    }


def defect_metrics(confusion: np.ndarray) -> dict[str, Any]:
    num_classes = confusion.shape[0]
    ious: list[float] = []
    per_class_iou: dict[str, float] = {}
    for class_index in range(num_classes):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        per_class_iou[str(class_index)] = iou
        if class_index > 0:
            ious.append(iou)
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion.tolist(),
    }


def ring_metrics(
    boundary_predictions: list[np.ndarray],
    boundary_targets: list[np.ndarray],
    distance_predictions: list[np.ndarray],
    distance_targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
) -> dict[str, Any]:
    dice_scores: list[float] = []
    distance_errors: list[float] = []
    for boundary_prediction, boundary_target, distance_prediction, distance_target, valid in zip(
        boundary_predictions, boundary_targets, distance_predictions, distance_targets, valid_masks
    ):
        prediction = boundary_prediction.astype(bool)
        target = boundary_target.astype(bool)
        valid = valid.astype(bool)
        intersection = np.logical_and(prediction, target) & valid
        denom = (prediction & valid).sum() + (target & valid).sum()
        dice_scores.append(float(2.0 * intersection.sum() / denom) if denom > 0 else 0.0)
        if valid.any():
            distance_errors.append(float(np.abs(distance_prediction.astype(np.float32) - distance_target.astype(np.float32))[valid].mean()))
    return {
        "boundary_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "distance_mae": float(np.mean(distance_errors)) if distance_errors else 0.0,
    }


def evaluate_task(model: Any, loader: Any, device: Any, task: str) -> dict[str, Any]:
    import torch

    model.eval()
    if task == "classification":
        confusion = np.zeros((len(PRIMARY_CATEGORIES), len(PRIMARY_CATEGORIES)), dtype=int)
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device)
                labels = batch["class_labels"].to(device)
                logits = model(images, tasks={"classification"})["classification_logits"]
                predictions = logits.argmax(dim=1)
                for label, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                    confusion[int(label), int(prediction)] += 1
        return classification_metrics(confusion)

    if task == "defect":
        num_classes = len(PRIMARY_CATEGORIES)
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(device)
                target = batch["defect_mask"].to(device)
                logits = model(images, tasks={"defect"})["defect_mask"]
                predictions = logits.argmax(dim=1)
                flat_target = target.view(-1).cpu().numpy()
                flat_prediction = predictions.view(-1).cpu().numpy()
                valid = (flat_target >= 0) & (flat_target < num_classes)
                bincount = np.bincount(
                    flat_target[valid] * num_classes + flat_prediction[valid],
                    minlength=num_classes * num_classes,
                )
                confusion += bincount.reshape(num_classes, num_classes)
        return defect_metrics(confusion)

    ring_predictions: list[np.ndarray] = []
    ring_targets: list[np.ndarray] = []
    distance_predictions: list[np.ndarray] = []
    distance_targets: list[np.ndarray] = []
    ring_valid_masks: list[np.ndarray] = []
    num_classes = len(PRIMARY_CATEGORIES)
    defect_confusion = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            requested_tasks = {"ring"}
            if "defect_mask" in batch:
                requested_tasks.add("defect")
            outputs = model(images, tasks=requested_tasks)
            if "ring_boundary" in batch:
                boundary = outputs["ring_boundary"].sigmoid().cpu().numpy()[:, 0] > 0.5
                distance = outputs["ring_distance"].cpu().numpy()[:, 0]
                ring_predictions.extend(boundary)
                ring_targets.extend(batch["ring_boundary"].cpu().numpy() > 0.5)
                distance_predictions.extend(distance)
                distance_targets.extend(batch["ring_distance"].cpu().numpy())
                ring_valid_masks.extend(batch["ring_valid"].cpu().numpy() > 0.5)
            if "defect_mask" in batch:
                predictions = outputs["defect_mask"].argmax(dim=1)
                target = batch["defect_mask"].cpu()
                flat_target = target.view(-1).numpy()
                flat_prediction = predictions.view(-1).cpu().numpy()
                valid = (flat_target >= 0) & (flat_target < num_classes)
                bincount = np.bincount(
                    flat_target[valid] * num_classes + flat_prediction[valid],
                    minlength=num_classes * num_classes,
                )
                defect_confusion += bincount.reshape(num_classes, num_classes)
    result: dict[str, Any] = {}
    if ring_predictions:
        result.update(ring_metrics(ring_predictions, ring_targets, distance_predictions, distance_targets, ring_valid_masks))
    if defect_confusion.sum() > 0:
        result["defect"] = defect_metrics(defect_confusion)
    return result


def _limit_records(records: list[LoaderRecord], limit: int) -> list[LoaderRecord]:
    if limit <= 0 or len(records) <= limit:
        return records
    return records[:limit]


def build_task_sets(
    catalog: WoodCatalog,
    split: str,
    imgsz: int,
    limit: int = 0,
    cache_dir: Path | None = None,
) -> dict[str, TaskDataset]:
    task_sets: dict[str, TaskDataset] = {}
    split_cache_dir = cache_dir / split if cache_dir is not None else None
    task_sets["ring"] = TaskDataset(catalog, "ring", _limit_records(list(catalog.ring_samples(splits={split}).records), limit), imgsz, split_cache_dir)
    task_sets["defect"] = TaskDataset(catalog, "defect", _limit_records(list(catalog.defect_samples(splits={split}).records), limit), imgsz, split_cache_dir)
    task_sets["classification"] = TaskDataset(catalog, "classification", _limit_records(list(catalog.classification_samples(splits={split}).records), limit), imgsz, split_cache_dir)
    task_sets["bridge"] = TaskDataset(catalog, "bridge", _limit_records(list(catalog.bridge_samples(splits={split}).records), limit), imgsz, split_cache_dir)
    return task_sets


def build_dataloaders(
    catalog: WoodCatalog,
    split: str,
    imgsz: int,
    batch_size: int,
    workers: int,
    limit: int = 0,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    loaders: dict[str, Any] = {}
    for task, dataset in build_task_sets(catalog, split, imgsz, limit=limit, cache_dir=cache_dir).items():
        if len(dataset) == 0:
            continue
        loaders[task] = make_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=(split == "train"),
            balanced=(split == "train" and task == "classification"),
            seed=seed,
        )
    return loaders


def warm_loaders_cache(loaders: dict[str, Any], split: str) -> None:
    for task, loader in loaders.items():
        dataset = loader.dataset
        if not isinstance(dataset, TaskDataset):
            continue
        print(json.dumps({"cache_split": split, "cache_task": task, "total": len(dataset)}, ensure_ascii=False), flush=True)
        dataset.warm_cache()


def train_epoch(
    model: Any,
    loaders: dict[str, Any],
    optimizer: Any,
    scaler: Any,
    device: Any,
    steps_per_epoch: int,
) -> dict[str, float]:
    import torch

    model.train()
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    task_order = list(loaders.keys())
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()

    for step in range(steps_per_epoch):
        task = task_order[step % len(task_order)]
        try:
            batch = next(iterators[task])
        except StopIteration:
            iterators[task] = iter(loaders[task])
            batch = next(iterators[task])
        optimizer.zero_grad(set_to_none=True)
        requested_tasks = set()
        if "ring_boundary" in batch:
            requested_tasks.add("ring")
        if "defect_mask" in batch:
            requested_tasks.add("defect")
        if "class_labels" in batch:
            requested_tasks.add("classification")
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(batch["images"].to(device), tasks=requested_tasks)
            task_loss = torch.zeros((), device=device)
            if "ring_boundary" in batch:
                ring_boundary = batch["ring_boundary"].to(device).unsqueeze(1)
                ring_distance = batch["ring_distance"].to(device).unsqueeze(1)
                ring_valid = batch["ring_valid"].to(device).unsqueeze(1)
                boundary_loss = masked_bce_with_logits(
                    outputs["ring_boundary"], ring_boundary, ring_valid, pos_weight=8.0
                )
                distance_loss = masked_l1_loss(outputs["ring_distance"], ring_distance, ring_valid)
                ring_dice = binary_dice_loss(outputs["ring_boundary"], ring_boundary, ring_valid)
                task_loss = task_loss + boundary_loss + ring_dice + 0.5 * distance_loss
            if "defect_mask" in batch:
                defect_target = batch["defect_mask"].to(device)
                defect_weight = torch.tensor([0.05, 1.0, 1.0], device=device)
                defect_loss = torch.nn.functional.cross_entropy(
                    outputs["defect_mask"],
                    defect_target,
                    weight=defect_weight,
                    ignore_index=IGNORE_INDEX,
                )
                task_loss = task_loss + defect_loss + 0.5 * multiclass_dice_loss(
                    outputs["defect_mask"], defect_target
                )
            if "class_labels" in batch:
                cls_loss = torch.nn.functional.cross_entropy(outputs["classification_logits"], batch["class_labels"].to(device))
                task_loss = task_loss + cls_loss
        loss = task_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals[task] += float(loss.item())
        counts[task] += 1

    return {f"{task}_loss": totals[task] / max(1, counts[task]) for task in task_order}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_runtime(args.cpu_threads)

    import torch

    project_root = args.project_root.resolve()
    if torch.cuda.is_available():
        # The remote container has a small /dev/shm; file-backed sharing avoids
        # worker crashes while keeping multiprocessing enabled.
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = (args.output_dir or project_root / "outputs" / "stage3_multitask" / "runs" / f"{args.backbone}_shared").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = WoodCatalog(project_root=project_root)
    cache_dir = None if args.no_cache else args.cache_dir.resolve()
    train_loaders = build_dataloaders(
        catalog,
        "train",
        args.imgsz,
        args.batch,
        args.workers,
        limit=args.max_train_samples_per_task,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    val_loaders = build_dataloaders(
        catalog,
        "val",
        args.imgsz,
        args.batch,
        args.workers,
        limit=args.max_val_samples_per_task,
        seed=args.seed,
        cache_dir=cache_dir,
    )
    if not train_loaders:
        raise RuntimeError("no training loaders found")

    if args.prewarm_cache and cache_dir is not None:
        print(json.dumps({"cache_dir": str(cache_dir), "prewarm": "start"}, ensure_ascii=False), flush=True)
        warm_loaders_cache(train_loaders, "train")
        warm_loaders_cache(val_loaders, "val")
        print(json.dumps({"cache_dir": str(cache_dir), "prewarm": "done"}, ensure_ascii=False), flush=True)

    model = build_shared_multitask_model(
        backbone=args.backbone,
        defect_channels=len(PRIMARY_CATEGORIES),
        classification_classes=len(PRIMARY_CATEGORIES),
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(max(1, args.cpu_threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(json.dumps(
        {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "backbone": args.backbone,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch": args.batch,
            "device": args.device,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "prewarm_cache": bool(args.prewarm_cache),
            "max_train_samples_per_task": args.max_train_samples_per_task,
            "max_val_samples_per_task": args.max_val_samples_per_task,
            "train_tasks": {task: len(loader.dataset) for task, loader in train_loaders.items()},
            "val_tasks": {task: len(loader.dataset) for task, loader in val_loaders.items()},
        },
        ensure_ascii=False,
    ))

    best_score = -math.inf
    best_val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loaders, optimizer, scaler, device, args.steps_per_epoch)
        val_metrics = {task: evaluate_task(model, loader, device, task) for task, loader in val_loaders.items()}
        scheduler.step()
        score = 0.0
        score += float(val_metrics.get("ring", {}).get("boundary_dice", 0.0))
        score += float(val_metrics.get("defect", {}).get("mean_iou", 0.0))
        score += float(val_metrics.get("classification", {}).get("macro_f1", 0.0))
        bridge_metrics = val_metrics.get("bridge", {})
        score += 0.5 * float(bridge_metrics.get("boundary_dice", 0.0))
        score += 0.5 * float(bridge_metrics.get("defect", {}).get("mean_iou", 0.0))
        if score >= best_score:
            best_score = score
            best_val_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "backbone": args.backbone,
                    "imgsz": args.imgsz,
                    "task_scores": val_metrics,
                },
                best_path,
            )
        record = {
            "epoch": epoch,
            "train_losses": train_losses,
            "val": val_metrics,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # The best epoch metrics were already computed above. Re-running every
    # validation loader here doubles remote storage reads and can make a
    # completed run appear hung before results.json is written.
    final_metrics = checkpoint.get("task_scores", best_val_metrics)
    results = {
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "backbone": args.backbone,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch": args.batch,
        "best_checkpoint": str(best_path),
        "history": history,
        "final_metrics": final_metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
