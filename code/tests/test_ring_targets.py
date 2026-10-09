from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wood_data.ring_targets import labelme_closed_targets, mokume_targets


class RingTargetTests(unittest.TestCase):
    def test_closed_polygons_create_annuli_and_valid_region(self) -> None:
        payload = {
            "shapes": [
                {"shape_type": "polygon", "points": [[4, 4], [11, 4], [11, 11], [4, 11]]},
                {"shape_type": "polygon", "points": [[1, 1], [14, 1], [14, 14], [1, 14]]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            target = labelme_closed_targets(path, (16, 16))
        self.assertEqual(target.closed_ring_count, 2)
        self.assertEqual(target.non_nested_pairs, 0)
        self.assertEqual(target.instance[2, 2], 1)
        self.assertEqual(target.instance[7, 7], 0)
        self.assertFalse(target.valid[0, 0])
        self.assertTrue(target.boundary.sum() > target.skeleton.sum())
        self.assertAlmostEqual(float(target.distance[target.skeleton][0]), 0.0)

    def test_mokume_255_is_background_not_ignore(self) -> None:
        labels = np.full((8, 8), 255, dtype=np.uint8)
        labels[:, 3] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "A_ann.png"
            Image.fromarray(labels).save(path)
            target = mokume_targets(path)
        self.assertEqual(int(target.skeleton.sum()), 8)
        self.assertTrue(target.valid.all())
        self.assertEqual(int(target.instance.max()), 0)


if __name__ == "__main__":
    unittest.main()
