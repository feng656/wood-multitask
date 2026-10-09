from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wood_data.model import DataRecord
from wood_data.splits import allocate_groups, split_leaks


class SplitTests(unittest.TestCase):
    def test_allocation_is_deterministic_and_complete(self) -> None:
        groups = [f"cube-{index}" for index in range(20)]
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        first = allocate_groups(groups, ratios, seed=42)
        second = allocate_groups(reversed(groups), ratios, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(groups))
        self.assertEqual({name: list(first.values()).count(name) for name in ratios}, {"train": 14, "val": 3, "test": 3})

    def test_leak_detector_uses_physical_group(self) -> None:
        records = [
            DataRecord("a.png", "mokume", "a", "mokume:C01", split="train"),
            DataRecord("b.png", "mokume", "b", "mokume:C01", split="val"),
        ]
        self.assertEqual(len(split_leaks(records)), 1)


if __name__ == "__main__":
    unittest.main()

