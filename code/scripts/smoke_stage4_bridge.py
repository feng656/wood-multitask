"""Run a no-artifact smoke test for the formal stage-4 bridge pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_stage3_instance_multitask import (
    SharedInstanceMultitaskModel,
    WoodCatalog,
    build_loaders,
    load_instance_model_checkpoint,
    train_epoch,
)
from train_stage4_bridge_finetune import bridge_split_audit, make_positive_bridge_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("stage-4 smoke test requires CUDA")
    device = torch.device(f"cuda:{args.device}")
    project_root = args.project_root.resolve()
    catalog = WoodCatalog(project_root=project_root)
    batch_sizes = {"ring": 8, "defect": 2, "classification": 16, "bridge": 1}
    train_loaders = build_loaders(
        catalog,
        "train",
        640,
        0,
        Path("/tmp/wood_stage4_bridge_cache640"),
        0,
        batch_sizes,
        42,
        {"defect"},
    )
    train_loaders["bridge"] = make_positive_bridge_loader(
        train_loaders["bridge"].dataset, 1, 0, 0.75, 42
    )
    val_bridge = build_loaders(
        catalog,
        "val",
        640,
        0,
        Path("/tmp/wood_stage4_bridge_cache640"),
        0,
        batch_sizes,
        42,
    )["bridge"]
    test_bridge = build_loaders(
        catalog,
        "test",
        640,
        0,
        Path("/tmp/wood_stage4_bridge_cache640"),
        0,
        batch_sizes,
        42,
    )["bridge"]
    split_audit = bridge_split_audit(
        catalog,
        {"train": train_loaders["bridge"], "val": val_bridge, "test": test_bridge},
    )

    model = SharedInstanceMultitaskModel(
        "resnet34", 640, anchor_profile="small_crack"
    ).to(device)
    initialization = load_instance_model_checkpoint(
        model, args.checkpoint.resolve(), device
    )
    if initialization["missing_after_partial_load"] or initialization["skipped_shape_mismatch"]:
        raise RuntimeError(f"checkpoint is not strict-compatible: {initialization}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = train_epoch(
        model,
        train_loaders,
        optimizer,
        scaler,
        device,
        steps_per_epoch=1,
        schedule=["bridge"],
        freeze_batch_norm_stats=True,
    )
    if not losses or not all(torch.isfinite(torch.tensor(value)) for value in losses.values()):
        raise RuntimeError(f"non-finite smoke losses: {losses}")
    print(
        json.dumps(
            {
                "status": "ok",
                "initialization": initialization,
                "bridge_split_audit": split_audit,
                "one_step_losses": losses,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
