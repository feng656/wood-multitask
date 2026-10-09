"""Evaluate a YOLO segmentation checkpoint on prepared defect datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label-format", choices=["box", "segment"], required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def xywh_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return [x1, y1, x2, y2]


def polygon_to_mask(points: list[float], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    coords = [(points[index] * width, points[index + 1] * height) for index in range(0, len(points), 2)]
    if len(coords) >= 3:
        draw.polygon(coords, fill=1)
    return np.asarray(image, dtype=bool)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / union)


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def box_iou(pred: list[float], gt: list[float]) -> float:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    return 0.0 if union <= 0 else float(inter / union)


def ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[points + 1] - mrec[points]) * mpre[points + 1]))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def label_paths(dataset_root: Path, split: str) -> list[Path]:
    return sorted((dataset_root / "images" / split).glob("*"))


def parse_labels(path: Path, width: int, height: int, label_format: str) -> list[dict[str, Any]]:
    label_path = path.parent.parent.parent / "labels" / path.parent.name / f"{path.stem}.txt"
    if not label_path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if label_format == "box":
            if len(values) != 4:
                continue
            objects.append({"cls": cls, "box": xywh_to_xyxy(values, width, height)})
        else:
            if len(values) < 6:
                continue
            objects.append({
                "cls": cls,
                "box": xywh_to_xyxy(
                    [min(values[::2]), min(values[1::2]), max(values[::2]) - min(values[::2]), max(values[1::2]) - min(values[1::2])],
                    width,
                    height,
                ),
                "mask": polygon_to_mask(values, width, height),
            })
    return objects


def flatten_classes(items: dict[int, list[dict[str, Any]]], class_id: int) -> list[dict[str, Any]]:
    return items.get(class_id, [])


def compute_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: dict[int, dict[int, list[dict[str, Any]]]],
    class_ids: list[int],
    thresholds: np.ndarray,
    use_masks: bool,
) -> dict[str, Any]:
    box_ap: dict[str, float] = {}
    box_recall: dict[str, float] = {}
    mask_ap: dict[str, float] = {}
    mask_recall: dict[str, float] = {}
    dice_scores: list[float] = []
    miou_scores: list[float] = []

    for threshold in thresholds:
        per_class_boxes: list[float] = []
        per_class_masks: list[float] = []
        for class_id in class_ids:
            preds = [item for item in predictions if item["cls"] == class_id]
            preds.sort(key=lambda item: item["conf"], reverse=True)
            gt_map = {image_id: flatten_classes(items, class_id) for image_id, items in ground_truths.items()}
            npos = sum(len(items) for items in gt_map.values())
            if npos == 0:
                continue
            matched = {image_id: np.zeros(len(items), dtype=bool) for image_id, items in gt_map.items()}
            tp: list[int] = []
            fp: list[int] = []
            for item in preds:
                items = gt_map.get(item["image_id"], [])
                if not items:
                    tp.append(0)
                    fp.append(1)
                    continue
                best_iou = 0.0
                best_index = -1
                for index, gt in enumerate(items):
                    if matched[item["image_id"]][index]:
                        continue
                    if use_masks:
                        predicted_mask = resize_mask(item["mask"], gt["mask"].shape)
                        current_iou = mask_iou(predicted_mask, gt["mask"])
                    else:
                        current_iou = box_iou(item["box"], gt["box"])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_index = index
                if best_iou >= threshold and best_index >= 0:
                    matched[item["image_id"]][best_index] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            tp_array = np.cumsum(np.asarray(tp))
            fp_array = np.cumsum(np.asarray(fp))
            recalls = tp_array / max(1, npos)
            precisions = tp_array / np.maximum(1, tp_array + fp_array)
            ap = ap_from_pr(recalls, precisions)
            if use_masks:
                per_class_masks.append(ap)
            else:
                per_class_boxes.append(ap)
            if threshold == 0.5:
                recall = float(recalls[-1]) if recalls.size else 0.0
                if use_masks:
                    mask_recall[str(class_id)] = recall
                else:
                    box_recall[str(class_id)] = recall

            if use_masks and threshold == 0.5:
                for image_id, items in gt_map.items():
                    if not items:
                        continue
                    matched_indices = matched.get(image_id)
                    if matched_indices is None:
                        continue
                    for index, gt in enumerate(items):
                        if not matched_indices[index]:
                            continue
                        candidate_pred = next(
                            (
                                pred
                                for pred in preds
                                if pred["image_id"] == image_id
                                and mask_iou(resize_mask(pred["mask"], gt["mask"].shape), gt["mask"]) >= 0.5
                            ),
                            None,
                        )
                        if candidate_pred is not None:
                            resized_pred = resize_mask(candidate_pred["mask"], gt["mask"].shape)
                            dice_scores.append(mask_dice(resized_pred, gt["mask"]))
                            miou_scores.append(mask_iou(resized_pred, gt["mask"]))

        if per_class_boxes and not use_masks:
            box_ap[f"{threshold:.2f}"] = float(np.mean(per_class_boxes))
        if per_class_masks and use_masks:
            mask_ap[f"{threshold:.2f}"] = float(np.mean(per_class_masks))

    result: dict[str, Any] = {}
    if box_ap:
        result["box_ap"] = box_ap
        result["box_map"] = float(np.mean(list(box_ap.values())))
        result["box_recall"] = box_recall
    if mask_ap:
        result["mask_ap"] = mask_ap
        result["mask_map"] = float(np.mean(list(mask_ap.values())))
        result["mask_recall"] = mask_recall
    if dice_scores:
        result["mask_dice"] = float(np.mean(dice_scores))
    if miou_scores:
        result["mask_miou"] = float(np.mean(miou_scores))
    return result


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    data = load_yaml(args.data.resolve())
    dataset_root = Path(data["path"])
    split_dir = dataset_root / "images" / args.split
    image_paths = label_paths(dataset_root, args.split)
    if not image_paths:
        raise FileNotFoundError(split_dir)
    model = YOLO(str(args.weights.resolve()))

    predictions: list[dict[str, Any]] = []
    ground_truths: dict[int, dict[int, list[dict[str, Any]]]] = {}
    class_ids = list(range(len(data["names"]))) if isinstance(data["names"], list) else list(sorted(int(key) for key in data["names"].keys()))
    batch_size = max(1, int(args.batch))
    for offset in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[offset : offset + batch_size]
        results = model.predict(source=[str(path) for path in batch_paths], imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        for local_index, (path, result) in enumerate(zip(batch_paths, results), start=offset):
            width, height = result.orig_shape[1], result.orig_shape[0]
            ground_truths[local_index] = defaultdict(list)
            for item in parse_labels(path, width, height, args.label_format):
                ground_truths[local_index][item["cls"]].append(item)
            boxes = result.boxes
            masks = getattr(result, "masks", None)
            mask_data = masks.data.cpu().numpy() > 0.5 if masks is not None else None
            if boxes is None:
                continue
            for index in range(len(boxes)):
                box = boxes.xyxy[index].cpu().numpy().tolist()
                pred: dict[str, Any] = {
                    "image_id": local_index,
                    "cls": int(boxes.cls[index].item()),
                    "conf": float(boxes.conf[index].item()),
                    "box": box,
                }
                if mask_data is not None and index < mask_data.shape[0]:
                    pred["mask"] = mask_data[index]
                elif args.label_format == "segment":
                    pred["mask"] = np.zeros((height, width), dtype=bool)
                predictions.append(pred)

    thresholds = np.arange(0.5, 0.96, 0.05)
    metrics = compute_metrics(predictions, ground_truths, class_ids, thresholds, use_masks=False)
    if args.label_format == "segment":
        metrics.update(compute_metrics(predictions, ground_truths, class_ids, thresholds, use_masks=True))
    summary = {
        "data": str(args.data.resolve()),
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "label_format": args.label_format,
        "images": len(image_paths),
        "predictions": len(predictions),
        "classes": data["names"],
        "metrics": metrics,
    }
    output = args.output or (dataset_root / "eval" / f"{args.split}_{args.label_format}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
