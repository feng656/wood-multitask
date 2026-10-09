"""Audit a PyTorch checkpoint without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PyTorch model checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    path = args.checkpoint.resolve()
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    tensor_count = 0
    parameter_count = 0
    floating_value_count = 0
    nonfinite_value_count = 0
    nonfinite_tensors: list[str] = []
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        parameter_count += value.numel()
        if value.is_floating_point() or value.is_complex():
            floating_value_count += value.numel()
            invalid = int((~torch.isfinite(value)).sum().item())
            nonfinite_value_count += invalid
            if invalid:
                nonfinite_tensors.append(name)

    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "epoch": checkpoint.get("epoch"),
        "state_tensor_count": tensor_count,
        "state_value_count": parameter_count,
        "floating_value_count": floating_value_count,
        "nonfinite_value_count": nonfinite_value_count,
        "nonfinite_tensors": nonfinite_tensors,
        "has_metrics": checkpoint.get("metrics") is not None,
        "architecture": checkpoint.get("config", {}).get("architecture"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
