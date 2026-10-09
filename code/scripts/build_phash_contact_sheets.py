from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CSV_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "duplicate_audit"
    / "primary"
    / "cross_split_phash_candidates.csv"
)
OUTPUT = PROJECT_ROOT / "data_processed" / "duplicate_audit" / "contact_sheets"


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    pairs_per_page = 10
    thumb_size = (400, 146)
    row_height = 205
    for page_index in range(0, len(rows), pairs_per_page):
        selected = rows[page_index : page_index + pairs_per_page]
        sheet = Image.new("RGB", (850, 35 + row_height * len(selected)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"Cross-split pHash candidates {page_index + 1}-{page_index + len(selected)}", fill="black", font=font)
        for local_index, row in enumerate(selected):
            y = 35 + local_index * row_height
            left = thumbnail(WORKSPACE_ROOT / row["left_path"], thumb_size)
            right = thumbnail(WORKSPACE_ROOT / row["right_path"], thumb_size)
            sheet.paste(left, (12, y + 28))
            sheet.paste(right, (438, y + 28))
            draw.text((12, y + 4), f"{row['left_sample_id']} [{row['left_split']}]", fill="black", font=font)
            draw.text((438, y + 4), f"{row['right_sample_id']} [{row['right_split']}] d={row['hamming_distance']}", fill="black", font=font)
            draw.line((0, y + row_height - 1, 850, y + row_height - 1), fill=(180, 180, 180), width=1)
        destination = OUTPUT / f"phash_candidates_{page_index // pairs_per_page + 1}.jpg"
        sheet.save(destination, quality=95, subsampling=0)
        print(destination)


if __name__ == "__main__":
    main()
