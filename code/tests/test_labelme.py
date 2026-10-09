from __future__ import annotations

import unittest
import sys
import json
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wood_data.labelme import parse_bridge_label
from wood_data.defects import parse_bridge_polygons_and_ignores


class BridgeLabelTests(unittest.TestCase):
    def test_valid_label(self) -> None:
        label = parse_bridge_label("crack|na|C01|clear|0|1")
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.category, "crack")
        self.assertFalse(label.truncated)
        self.assertTrue(label.affects_rings)

    def test_invalid_label_is_rejected(self) -> None:
        self.assertIsNone(parse_bridge_label("1"))
        self.assertIsNone(parse_bridge_label("normal|na|N01|clear|0|0"))

    def test_non_target_bridge_shape_becomes_ignore_region(self) -> None:
        payload = {
            "shapes": [{
                "label": "other|na|O01|clear|0|0",
                "shape_type": "rectangle",
                "points": [[2, 3], [7, 11]],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            annotations, ignored = parse_bridge_polygons_and_ignores(path)
        self.assertEqual(annotations, [])
        self.assertEqual(ignored[0]["bbox"], [2.0, 3.0, 5.0, 8.0])


if __name__ == "__main__":
    unittest.main()
