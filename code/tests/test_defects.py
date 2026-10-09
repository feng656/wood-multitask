from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wood_data.defects import parse_vsb_boxes


class DefectConversionTests(unittest.TestCase):
    def test_vsb_uses_normalized_corner_coordinates(self) -> None:
        mapping = {"Live_Knot": {"category": "knot", "subtype": "sound_or_live", "attributes": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.txt"
            path.write_text("Live_Knot 0.1 0.2 0.4 0.6\n", encoding="utf-8")
            annotation = parse_vsb_boxes(path, 1000, 500, mapping)[0]
        self.assertEqual(annotation["bbox"], [100.0, 100.0, 300.0, 200.0])
        self.assertEqual(annotation["subtype"], "sound_or_live")

    def test_vsb_accepts_decimal_comma(self) -> None:
        mapping = {"Crack": {"category": "crack", "subtype": "", "attributes": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.txt"
            path.write_text("Crack 0,1 0,2 0,4 0,6\n", encoding="utf-8")
            annotation = parse_vsb_boxes(path, 100, 100, mapping)[0]
        self.assertEqual(annotation["bbox"], [10.0, 20.0, 30.0, 40.0])


if __name__ == "__main__":
    unittest.main()
