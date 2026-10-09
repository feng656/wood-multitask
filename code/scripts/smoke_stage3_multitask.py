"""Smoke test for the stage-3 shared encoder multitask prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def to_tensor(array: np.ndarray):
    import torch

    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def main() -> int:
    import torch

    from wood_data.loaders import WoodCatalog
    from wood_data.multitask import build_shared_multitask_model, tensor_summary

    catalog = WoodCatalog.default()
    model = build_shared_multitask_model("resnet34")
    model.eval()

    samples = {
        "ring": catalog.ring_samples(splits={"train"})[0],
        "defect": catalog.defect_samples(splits={"train"})[0],
        "classification": catalog.classification_samples(splits={"train"})[0],
    }
    bridge_dataset = catalog.bridge_samples(splits={"train"})
    bridge_sample = None
    for index in range(len(bridge_dataset)):
        candidate = bridge_dataset[index]
        if candidate["defect"]["annotations"]:
            bridge_sample = candidate
            break
    if bridge_sample is None:
        raise RuntimeError("no bridge sample with annotations found")
    samples["bridge"] = bridge_sample

    summaries: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for name, sample in samples.items():
            image = to_tensor(sample["image"]).float()
            outputs = model(image)
            summaries[name] = {
                "sample_id": sample["sample_id"],
                "input_shape": list(image.shape),
                "outputs": tensor_summary(outputs),
                "task_mask": sample["task_mask"],
            }

    output = PROJECT_ROOT / "outputs" / "annotation_audit" / "stage3_multitask_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
