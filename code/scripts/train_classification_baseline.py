"""Train a classification baseline on the prepared wood defect crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropSample:
    path: Path
    label: int
    category: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--crops-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test", "external_test"])
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(cpu_threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)


class CropDataset:
    def __init__(self, samples: list[CropSample], class_names: list[str], imgsz: int):
        self.samples = samples
        self.class_names = class_names
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB").resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        array = (array - mean) / std
        return array, sample.label


def read_samples(crops_csv: Path, data_root: Path, workspace_root: Path | None = None, splits: set[str] | None = None) -> tuple[list[CropSample], list[str]]:
    """Read classification crop samples from CSV.

    Args:
        crops_csv: Path to the classification CSV index.
        data_root: Fallback root for resolving crop_path (if not already absolute).
        workspace_root: Root that crop_path entries are relative to.
                        Defaults to *data_root* for backward compatibility.
        splits: Optional set of split names to filter to.
    """
    if workspace_root is None:
        workspace_root = data_root
    rows: list[dict[str, str]]
    with crops_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if splits is not None:
        rows = [row for row in rows if row["split"] in splits]
    class_names = sorted({row["category"] for row in rows})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    samples: list[CropSample] = []
    for row in rows:
        path = (workspace_root / row["crop_path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        samples.append(CropSample(path=path, label=class_to_index[row["category"]], category=row["category"], split=row["split"]))
    return samples, class_names


def filter_split(samples: list[CropSample], split_name: str) -> list[CropSample]:
    return [sample for sample in samples if sample.split == split_name]


def make_torch_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, sampler=None):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_sampler(samples: list[CropSample]):
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(sample.label for sample in samples)
    weights = [1.0 / counts[sample.label] for sample in samples]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(samples), replacement=True)


def build_model(name: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    model = getattr(models, name)(weights=None)
    if not hasattr(model, "fc"):
        raise AttributeError(f"unsupported model head for {name}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def move_batch(batch, device):
    import torch

    images, labels = batch
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def evaluate(model, loader, device, class_names: list[str]) -> dict[str, Any]:
    import torch

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images, labels = move_batch(batch, device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for label, pred in zip(all_labels, all_preds):
        confusion[label, pred] += 1
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
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
        "per_class_precision": {class_names[i]: per_class_precision[i] for i in range(num_classes)},
        "per_class_recall": {class_names[i]: per_class_recall[i] for i in range(num_classes)},
        "per_class_f1": {class_names[i]: per_class_f1[i] for i in range(num_classes)},
        "confusion_matrix": confusion.tolist(),
        "predictions": len(all_preds),
    }


def train_epoch(model, loader, optimizer, scaler, device) -> float:
    import torch

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_runtime(args.cpu_threads)

    import torch
    from torchvision import transforms  # noqa: F401

    project_root = args.project_root.resolve()
    workspace_root = project_root.parent.resolve()  # crop_path entries in CSV are relative to workspace root
    data_root = (args.data_root or project_root / "data_processed").resolve()
    crops_csv = (args.crops_csv or data_root / "classification_crops" / "classification_crops.csv").resolve()
    output_dir = (args.output_dir or project_root / "outputs" / "classification_baseline" / "runs" / f"{args.model}_balanced").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "project_root": str(project_root),
        "workspace_root": str(workspace_root),
        "data_root": str(data_root),
        "crops_csv": str(crops_csv),
        "output_dir": str(output_dir),
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "cpu_threads": args.cpu_threads,
        "device": args.device,
    }, ensure_ascii=False))

    all_samples, class_names = read_samples(crops_csv, data_root, workspace_root=workspace_root)
    train_samples = filter_split(all_samples, "train")
    val_samples = filter_split(all_samples, "val")
    test_samples = [sample for sample in all_samples if sample.split in set(args.eval_splits)]

    if not train_samples:
        raise RuntimeError("no training samples found")
    if not val_samples:
        raise RuntimeError("no validation samples found")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    model = build_model(args.model, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_dataset = CropDataset(train_samples, class_names, args.imgsz)
    val_dataset = CropDataset(val_samples, class_names, args.imgsz)
    train_loader = make_torch_loader(train_dataset, args.batch, shuffle=True, num_workers=args.workers, sampler=build_sampler(train_samples))
    val_loader = make_torch_loader(val_dataset, args.batch, shuffle=False, num_workers=args.workers)

    best_score = -math.inf
    history: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"

    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(max(1, args.cpu_threads))
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if val_metrics["macro_f1"] >= best_score:
            best_score = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "imgsz": args.imgsz,
                    "model_name": args.model,
                },
                best_path,
            )
        print(json.dumps(record, ensure_ascii=False))

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    outputs: dict[str, Any] = {
        "project_root": str(project_root),
        "crops_csv": str(crops_csv),
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "class_names": class_names,
        "best_checkpoint": str(best_path),
        "history": history,
    }

    for split in args.eval_splits:
        eval_samples = [sample for sample in all_samples if sample.split == split]
        if not eval_samples:
            continue
        split_loader = make_torch_loader(CropDataset(eval_samples, class_names, args.imgsz), args.batch, shuffle=False, num_workers=args.workers)
        outputs[f"{split}_metrics"] = evaluate(model, split_loader, device, class_names)

    (output_dir / "results.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
