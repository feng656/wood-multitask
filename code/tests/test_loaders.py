from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wood_data.loaders import WoodCatalog


class LoaderSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = WoodCatalog.default()

    def test_ring_sample_returns_aligned_targets(self) -> None:
        dataset = self.catalog.ring_samples(splits={"train"})
        sample = dataset[0]
        self.assertEqual(sample["task_mask"], {"ring": 1, "defect": 0, "classification": 0})
        self.assertEqual(sample["image"].shape[:2], sample["targets"]["skeleton"].shape)
        self.assertEqual(sample["targets"]["boundary"].shape, sample["targets"]["valid"].shape)
        self.assertEqual(sample["targets"]["instance"].shape, sample["targets"]["skeleton"].shape)

    def test_missing_defect_label_stays_masked(self) -> None:
        sample = self.catalog.load("ring", "urudendro2:A10")
        self.assertEqual(sample["task_mask"]["ring"], 1)
        self.assertEqual(sample["task_mask"]["defect"], 0)
        self.assertIn("targets", sample)

    def test_defect_sample_returns_annotations_and_mask(self) -> None:
        dataset = self.catalog.defect_samples(splits={"train"})
        sample = dataset[0]
        self.assertEqual(sample["task_mask"], {"ring": 0, "defect": 1, "classification": 0})
        self.assertEqual(sample["image"].shape[2], 3)
        self.assertIsInstance(sample["annotations"], list)
        self.assertGreaterEqual(len(sample["annotations"]), 0)

    def test_bridge_sample_returns_joint_supervision(self) -> None:
        dataset = self.catalog.bridge_samples(splits={"train"})
        sample = None
        for index in range(len(dataset)):
            candidate = dataset[index]
            if candidate["defect"]["annotations"]:
                sample = candidate
                break
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["task_mask"], {"ring": 1, "defect": 1, "classification": 0})
        self.assertEqual(sample["image"].shape[:2], sample["ring_targets"]["skeleton"].shape)
        for annotation in sample["defect"]["annotations"]:
            bbox = annotation["bbox"]
            self.assertGreater(bbox[2], 0.0)
            self.assertGreater(bbox[3], 0.0)

    def test_classification_sample_returns_label(self) -> None:
        dataset = self.catalog.classification_samples(splits={"train"})
        sample = dataset[0]
        self.assertEqual(sample["task_mask"], {"ring": 0, "defect": 0, "classification": 1})
        self.assertEqual(sample["image"].shape[2], 3)
        self.assertIn("category", sample["label"])


if __name__ == "__main__":
    unittest.main()
