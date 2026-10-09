"""Run the formal stage-4 bridge-set joint fine-tuning experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from train_stage3_instance_multitask import (
    DetectionMetrics,
    InstanceTaskDataset,
    SharedInstanceMultitaskModel,
    WoodCatalog,
    build_loaders,
    collate_instance_task,
    evaluate_loader,
    load_instance_model_checkpoint,
    move_targets,
    parse_score_weights,
    parse_task_ratios,
    ring_metrics,
    save_checkpoint,
    set_seed,
    train_epoch,
    validation_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--backbone", choices=["resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--anchor-profile", choices=["default", "small_crack"], default="small_crack")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=534)
    parser.add_argument("--ring-batch", type=int, default=8)
    parser.add_argument("--detection-batch", type=int, default=2)
    parser.add_argument("--classification-batch", type=int, default=16)
    parser.add_argument("--bridge-batch", type=int, default=1)
    parser.add_argument("--task-ratios", default="ring:1,defect:4,classification:2,bridge:4")
    parser.add_argument(
        "--score-weights",
        default="ring:0.25,defect_box:0.5,defect_mask:0.5,classification:0.25,"
        "bridge_ring:1,bridge_box:2,bridge_mask:2",
    )
    parser.add_argument("--near-ring-score-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/wood_stage4_bridge_cache640"))
    parser.add_argument("--bridge-positive-fraction", type=float, default=0.75)
    parser.add_argument("--neighborhood-radius", type=int, default=16)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def make_positive_bridge_loader(
    dataset: InstanceTaskDataset,
    batch_size: int,
    workers: int,
    positive_fraction: float,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("bridge-positive-fraction must be between 0 and 1")
    is_positive = [
        bool(dataset.catalog.coco_by_sample[record.sample_id]["annotations"])
        for record in dataset.records
    ]
    positive_count = sum(is_positive)
    negative_count = len(is_positive) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("bridge positive-aware sampling requires positive and verified-negative records")
    weights = [
        positive_fraction / positive_count if positive else (1.0 - positive_fraction) / negative_count
        for positive in is_positive
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda items: collate_instance_task("bridge", items),
    )


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    dilated = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return dilated[0, 0].numpy() > 0


def regional_ring_metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    distance_predictions: list[np.ndarray],
    distance_targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
    regions: list[np.ndarray],
) -> dict[str, float | int]:
    macro_dice: list[float] = []
    distance_error_sum = 0.0
    distance_pixel_count = 0
    intersection_sum = 0
    denominator_sum = 0
    evaluated_images = 0
    informative_images = 0
    target_boundary_pixels = 0
    for prediction, target, distance_prediction, distance_target, valid, region in zip(
        predictions,
        targets,
        distance_predictions,
        distance_targets,
        valid_masks,
        regions,
    ):
        mask = valid.astype(bool) & region.astype(bool)
        if not mask.any():
            continue
        evaluated_images += 1
        predicted_region = prediction.astype(bool) & mask
        target_region = target.astype(bool) & mask
        intersection = int((predicted_region & target_region).sum())
        denominator = int(predicted_region.sum() + target_region.sum())
        intersection_sum += intersection
        denominator_sum += denominator
        target_boundary_pixels += int(target_region.sum())
        if denominator:
            informative_images += 1
            macro_dice.append(2.0 * intersection / denominator)
        errors = np.abs(distance_prediction - distance_target)[mask]
        distance_error_sum += float(errors.sum())
        distance_pixel_count += int(errors.size)
    return {
        "boundary_dice_macro": float(np.mean(macro_dice)) if macro_dice else 0.0,
        "boundary_dice_micro": 2.0 * intersection_sum / denominator_sum if denominator_sum else 0.0,
        "distance_mae": distance_error_sum / distance_pixel_count if distance_pixel_count else 0.0,
        "evaluated_images": evaluated_images,
        "informative_images": informative_images,
        "valid_pixels": distance_pixel_count,
        "target_boundary_pixels": target_boundary_pixels,
    }


def evaluate_bridge(
    model: SharedInstanceMultitaskModel,
    loader: Any,
    device: Any,
    neighborhood_radius: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    detector_metrics = DetectionMetrics()
    boundary_predictions: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    distance_predictions: list[np.ndarray] = []
    distance_targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    near_regions: list[np.ndarray] = []
    far_regions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = [image.to(device, non_blocking=True) for image in batch["images"]]
            cpu_targets = batch["targets"]
            targets = move_targets(cpu_targets, device)
            outputs = model(images, targets=targets, ring=True, detection=True)
            for prediction, target in zip(outputs["detections"], targets):
                detector_metrics.update(prediction, target)
            boundary_predictions.extend(outputs["ring_boundary"].sigmoid().cpu().numpy()[:, 0] >= 0.5)
            boundary_targets.extend(batch["ring_boundary"].numpy() >= 0.5)
            distance_predictions.extend(outputs["ring_distance"].cpu().numpy()[:, 0])
            distance_targets.extend(batch["ring_distance"].numpy())
            valid_masks.extend(batch["ring_valid"].numpy() >= 0.5)
            for target in cpu_targets:
                masks = target["masks"].numpy().astype(bool)
                union = masks.any(axis=0) if len(masks) else np.zeros((model.image_size, model.image_size), dtype=bool)
                near = dilate_mask(union, neighborhood_radius)
                near_regions.append(near)
                far_regions.append(~near)

    return {
        "ring": ring_metrics(
            boundary_predictions,
            boundary_targets,
            distance_predictions,
            distance_targets,
            valid_masks,
        ),
        "ring_regions": {
            "defect_neighborhood": regional_ring_metrics(
                boundary_predictions,
                boundary_targets,
                distance_predictions,
                distance_targets,
                valid_masks,
                near_regions,
            ),
            "non_defect_region": regional_ring_metrics(
                boundary_predictions,
                boundary_targets,
                distance_predictions,
                distance_targets,
                valid_masks,
                far_regions,
            ),
            "neighborhood_radius_pixels": neighborhood_radius,
        },
        "detection": detector_metrics.compute(),
    }


def evaluate_all(
    model: SharedInstanceMultitaskModel,
    loaders: dict[str, Any],
    device: Any,
    neighborhood_radius: int,
) -> dict[str, Any]:
    metrics = {}
    for task, loader in loaders.items():
        metrics[task] = (
            evaluate_bridge(model, loader, device, neighborhood_radius)
            if task == "bridge"
            else evaluate_loader(model, loader, device, task)
        )
    return metrics


def stage4_score(metrics: dict[str, Any], weights: dict[str, float], near_weight: float) -> float:
    near_dice = float(
        metrics.get("bridge", {})
        .get("ring_regions", {})
        .get("defect_neighborhood", {})
        .get("boundary_dice_micro", 0.0)
    )
    return validation_score(metrics, weights) + near_weight * near_dice


def bridge_split_audit(catalog: WoodCatalog, loaders: dict[str, Any]) -> dict[str, Any]:
    split_groups: dict[str, set[str]] = {}
    result: dict[str, Any] = {}
    for split, loader in loaders.items():
        records = loader.dataset.records
        groups = {record.group_id for record in records}
        split_groups[split] = groups
        datasets = Counter(catalog.manifest_by_sample[record.sample_id]["dataset_id"] for record in records)
        positives = sum(bool(catalog.coco_by_sample[record.sample_id]["annotations"]) for record in records)
        result[split] = {
            "records": len(records),
            "groups": len(groups),
            "dataset_records": dict(sorted(datasets.items())),
            "positive_records": positives,
            "verified_negative_records": len(records) - positives,
        }
    overlaps = {}
    names = sorted(split_groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_{right}"] = sorted(split_groups[left] & split_groups[right])
    result["group_overlaps"] = overlaps
    if any(overlaps.values()):
        raise RuntimeError(f"bridge physical-group split leakage: {overlaps}")
    return result


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
    if device.type != "cuda":
        raise RuntimeError("formal stage-4 bridge fine-tuning requires CUDA")
    if args.workers != 0:
        raise ValueError("formal remote configuration requires workers=0")

    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        or project_root / "outputs" / "stage4_bridge_finetune" / "runs" / "resnet34_bridge_final640"
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.exist_ok or args.resume):
        raise FileExistsError(f"non-empty output directory requires --exist-ok or --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = args.init_checkpoint
    if init_checkpoint is None:
        init_checkpoint = (
            project_root
            / "outputs"
            / "stage3_multitask"
            / "runs"
            / "resnet34_instance_stage3_final640"
            / "best.pt"
        )
    init_checkpoint = init_checkpoint.resolve()
    if not init_checkpoint.exists() and args.resume is None:
        raise FileNotFoundError(f"stage-3 best checkpoint not found: {init_checkpoint}")

    catalog = WoodCatalog(project_root=project_root)
    batch_sizes = {
        "ring": args.ring_batch,
        "defect": args.detection_batch,
        "classification": args.classification_batch,
        "bridge": args.bridge_batch,
    }
    train_loaders = build_loaders(
        catalog,
        "train",
        args.imgsz,
        args.workers,
        args.cache_dir.resolve(),
        0,
        batch_sizes,
        args.seed,
        {"defect"},
    )
    train_loaders["bridge"] = make_positive_bridge_loader(
        train_loaders["bridge"].dataset,
        args.bridge_batch,
        args.workers,
        args.bridge_positive_fraction,
        args.seed,
    )
    val_loaders = build_loaders(
        catalog,
        "val",
        args.imgsz,
        args.workers,
        args.cache_dir.resolve(),
        0,
        batch_sizes,
        args.seed,
    )
    test_bridge_loader = build_loaders(
        catalog,
        "test",
        args.imgsz,
        args.workers,
        args.cache_dir.resolve(),
        0,
        batch_sizes,
        args.seed,
    )["bridge"]
    split_audit = bridge_split_audit(
        catalog,
        {
            "train": train_loaders["bridge"],
            "val": val_loaders["bridge"],
            "test": test_bridge_loader,
        },
    )

    schedule = parse_task_ratios(args.task_ratios)
    score_weights = parse_score_weights(args.score_weights)
    missing_tasks = set(schedule) - set(train_loaders)
    if missing_tasks:
        raise RuntimeError(f"missing formal training tasks: {sorted(missing_tasks)}")

    model = SharedInstanceMultitaskModel(
        args.backbone,
        args.imgsz,
        anchor_profile=args.anchor_profile,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_epoch = 1
    initial_metrics: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    best_score = -math.inf
    initialization: dict[str, Any] | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_best = torch.load(output_dir / "best.pt", map_location="cpu") if (output_dir / "best.pt").exists() else checkpoint
        best_metrics = saved_best.get("metrics") or {}
        if best_metrics:
            best_score = stage4_score(best_metrics, score_weights, args.near_ring_score_weight)
        initial_path = output_dir / "initial_metrics.json"
        if initial_path.exists():
            initial_metrics = json.loads(initial_path.read_text(encoding="utf-8"))["metrics"]
    else:
        initialization = load_instance_model_checkpoint(model, init_checkpoint, device)
        if initialization["missing_after_partial_load"] or initialization["skipped_shape_mismatch"]:
            raise RuntimeError(f"stage-4 initialization was not strict-compatible: {initialization}")

    config = {
        "stage": "stage4_bridge_joint_finetune",
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
        "near_ring_score_weight": args.near_ring_score_weight,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "eval_interval": args.eval_interval,
        "bridge_positive_fraction": args.bridge_positive_fraction,
        "freeze_batch_norm_stats": True,
        "neighborhood_radius": args.neighborhood_radius,
        "device": str(device),
        "seed": args.seed,
        "train_records": {task: len(loader.dataset) for task, loader in train_loaders.items()},
        "val_records": {task: len(loader.dataset) for task, loader in val_loaders.items()},
        "bridge_split_audit": split_audit,
        "initialization": initialization,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "config", **config}, ensure_ascii=False), flush=True)

    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history_path = output_dir / "history.jsonl"
    history: list[dict[str, Any]] = []
    if history_path.exists() and args.resume is not None:
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]

    if args.resume is None:
        initial_metrics = evaluate_all(model, val_loaders, device, args.neighborhood_radius)
        best_metrics = initial_metrics
        best_score = stage4_score(initial_metrics, score_weights, args.near_ring_score_weight)
        initial_record = {"epoch": 0, "score": best_score, "metrics": initial_metrics}
        (output_dir / "initial_metrics.json").write_text(
            json.dumps(initial_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        save_checkpoint(best_path, 0, model, optimizer, scheduler, scaler, initial_metrics, config)
        print(json.dumps({"event": "initial_validation", **initial_record}, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_epoch(
            model,
            train_loaders,
            optimizer,
            scaler,
            device,
            args.steps_per_epoch,
            schedule,
            freeze_batch_norm_stats=True,
        )
        should_evaluate = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        metrics = evaluate_all(model, val_loaders, device, args.neighborhood_radius) if should_evaluate else None
        score = stage4_score(metrics, score_weights, args.near_ring_score_weight) if metrics else None
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
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_bridge_metrics = evaluate_bridge(
        model,
        test_bridge_loader,
        device,
        args.neighborhood_radius,
    )
    results = {
        "config": config,
        "initial_metrics": initial_metrics,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_score": best_score,
        "best_checkpoint": str(best_path),
        "best_metrics": best_checkpoint["metrics"],
        "final_test_bridge_metrics": test_bridge_metrics,
        "history": history,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "best_epoch": results["best_epoch"], "best_score": best_score}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
