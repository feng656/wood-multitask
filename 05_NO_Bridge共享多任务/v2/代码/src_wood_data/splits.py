from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from .model import DataRecord


def _stable_rank(value: str, seed: int) -> str:
    payload = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate_groups(
    group_ids: Iterable[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    groups = sorted(set(group_ids), key=lambda item: _stable_rank(item, seed))
    if not groups:
        return {}
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    exact = {name: len(groups) * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = len(groups) - sum(counts.values())
    order = sorted(ratios, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1

    assignment: dict[str, str] = {}
    cursor = 0
    for split_name in ratios:
        for group_id in groups[cursor : cursor + counts[split_name]]:
            assignment[group_id] = split_name
        cursor += counts[split_name]
    return assignment


def assign_primary_splits(
    records: list[DataRecord], split_config: dict[str, object], seed: int
) -> None:
    bridge_group_ids: set[str] = set()
    bridge_ratios = split_config.get("bridge", {"train": 0.6, "val": 0.2, "test": 0.2})
    for dataset_id in sorted({r.dataset_id for r in records if r.dataset_role == "bridge"}):
        selected = [r for r in records if r.dataset_id == dataset_id and r.dataset_role == "bridge"]
        assignment = allocate_groups(
            (r.group_id for r in selected), bridge_ratios, seed + 1000 + len(dataset_id)
        )
        bridge_group_ids.update(assignment)
        for record in records:
            if record.dataset_id == dataset_id and record.group_id in assignment:
                record.split = assignment[record.group_id]

    ring_r1 = [
        r for r in records
        if r.dataset_id in {"urudendro", "urudendro2"} and not r.split
    ]
    assignment = allocate_groups(
        (r.group_id for r in ring_r1),
        split_config["ring_r1_train_val"],
        seed + 1,
    )
    for record in ring_r1:
        record.split = assignment[record.group_id]

    # VSB already carries the immutable group split from vsb-final.
    for dataset_id in ("mokume",):
        selected = [r for r in records if r.dataset_id == dataset_id and not r.split]
        assignment = allocate_groups(
            (r.group_id for r in selected), split_config[dataset_id], seed + len(dataset_id)
        )
        for record in selected:
            record.split = assignment[record.group_id]

    for record in records:
        if record.dataset_id in {"urudendro4", "vnwoodknot", "indiana", "oulu"}:
            record.split = "external_test"

    folds = int(split_config["ring_r2_folds"])
    r2_records = [r for r in records if r.dataset_id == "urudendro4"]
    fold_ratios = {str(i): 1.0 / folds for i in range(folds)}
    assignment = allocate_groups(
        (r.group_id for r in r2_records), fold_ratios, seed + 404
    )
    for record in r2_records:
        record.ring_r2_fold = assignment[record.group_id]


def split_leaks(records: list[DataRecord]) -> list[dict[str, object]]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.group_id].add(record.split)
    return [
        {"group_id": group_id, "splits": sorted(splits)}
        for group_id, splits in sorted(group_splits.items())
        if len(splits) > 1
    ]
