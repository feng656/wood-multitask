from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wood_data.loaders import WoodCatalog


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation_audit"


def _resize_panel(image: Image.Image, max_side: int = 420) -> Image.Image:
    panel = image.copy()
    panel.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return panel


def _as_image(array: np.ndarray) -> Image.Image:
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    if array.ndim == 2:
        return Image.fromarray(array)
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array.astype(np.uint8), mode="RGB")
    raise ValueError(f"unsupported array shape: {array.shape}")


def _draw_title(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    draw.rectangle((x, y, x + 400, y + 26), fill=(255, 255, 255))
    draw.text((x + 6, y + 5), text, fill=(0, 0, 0))


def _overlay_boundary(image: np.ndarray, boundary: np.ndarray, skeleton: np.ndarray | None = None) -> Image.Image:
    rgb = image.copy().astype(np.uint8)
    rgb[boundary] = [255, 48, 48]
    if skeleton is not None:
        rgb[skeleton] = [48, 255, 96]
    return Image.fromarray(rgb)


def _overlay_boxes(image: np.ndarray, annotations: list[dict[str, object]], color: tuple[int, int, int]) -> Image.Image:
    if isinstance(image, Image.Image):
        panel = image.convert("RGB")
    else:
        panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    for annotation in annotations:
        x, y, width, height = map(float, annotation["bbox"])
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
    return panel


def _label_panel(title: str, label: dict[str, object]) -> Image.Image:
    panel = Image.new("RGB", (420, 320), "white")
    draw = ImageDraw.Draw(panel)
    _draw_title(draw, title, 0, 0)
    text = f"{label['category']}\n{label['subtype']}\n{';'.join(label['attributes']) or 'no attributes'}"
    draw.multiline_text((18, 60), text, fill=(0, 0, 0), spacing=6)
    return panel


def _panel(title: str, image: Image.Image) -> Image.Image:
    panel = _resize_panel(image)
    canvas = Image.new("RGB", (panel.width, panel.height + 28), "white")
    canvas.paste(panel, (0, 28))
    draw = ImageDraw.Draw(canvas)
    _draw_title(draw, title, 0, 0)
    return canvas


def main() -> int:
    catalog = WoodCatalog.default()
    ring = catalog.ring_samples(splits={"train"})[0]
    defect = catalog.defect_samples(splits={"train"})[0]
    classification = catalog.classification_samples(splits={"train"})[0]
    bridge = None
    bridge_dataset = catalog.bridge_samples(splits={"train"})
    for index in range(len(bridge_dataset)):
        candidate = bridge_dataset[index]
        if candidate["defect"]["annotations"]:
            bridge = candidate
            break
    if bridge is None:
        raise RuntimeError("no bridge sample with annotations found")

    panels = [
        _panel("ring / image", _as_image(ring["image"])),
        _panel("ring / targets", _overlay_boundary(ring["image"], ring["targets"]["boundary"], ring["targets"]["skeleton"])),
        _panel("defect / image", _as_image(defect["image"])),
        _panel("defect / boxes", _overlay_boxes(defect["image"], defect["annotations"], (255, 215, 0))),
        _panel("classification / crop", _as_image(classification["image"])),
        _label_panel("classification / label", classification["label"]),
        _panel("bridge / image", _as_image(bridge["image"])),
        _panel(
            "bridge / joint",
            _overlay_boxes(
                _overlay_boundary(bridge["image"], bridge["ring_targets"]["boundary"], bridge["ring_targets"]["skeleton"]),
                bridge["defect"]["annotations"],
                (255, 215, 0),
            ),
        ),
    ]

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), "white")
    cursor = 0
    for panel in panels:
        sheet.paste(panel, (0, cursor))
        cursor += panel.height

    output_path = OUTPUT_DIR / "stage1_loader_preview.png"
    sheet.save(output_path, quality=95)

    summary = {
        "ring_sample": ring["sample_id"],
        "defect_sample": defect["sample_id"],
        "classification_sample": classification["sample_id"],
        "bridge_sample": bridge["sample_id"],
        "output": output_path.as_posix(),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
