"""Train or smoke-test the provided YOLOv8s defect segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--name", default="yolov8s_defect_baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    project_root = args.project_root.resolve()
    data = (args.data or project_root / "outputs" / "defect_baseline" / "yolo_dataset" / "data.yaml").resolve()
    weights = args.weights.resolve()
    if not data.exists():
        raise FileNotFoundError(data)
    if not weights.exists():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights))
    print(f"task={model.task} weights={weights} data={data}")
    model.train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(project_root / "outputs" / "defect_baseline" / "runs"),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
