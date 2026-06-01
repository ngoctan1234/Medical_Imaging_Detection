from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SOURCE_IMAGE = Path("data/raw/pneumonia_xray_wikimedia.jpg")
OUTPUT_DIR = Path("data/demo_cases")
MANIFEST = OUTPUT_DIR / "manifest.csv"


def center_crop(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    crop_w = int(width * scale)
    crop_h = int(height * scale)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))


def make_variant(image: Image.Image, index: int) -> tuple[Image.Image, str]:
    variant = image.copy()
    transforms: list[str] = []

    if index % 2 == 0:
        variant = ImageEnhance.Contrast(variant).enhance(1.08 + (index % 5) * 0.06)
        transforms.append("contrast")
    if index % 3 == 0:
        variant = ImageEnhance.Brightness(variant).enhance(0.92 + (index % 4) * 0.04)
        transforms.append("brightness")
    if index % 4 == 0:
        variant = ImageEnhance.Sharpness(variant).enhance(1.25)
        transforms.append("sharpness")
    if index % 5 == 0:
        variant = variant.filter(ImageFilter.GaussianBlur(radius=0.45))
        transforms.append("mild blur")
    if index % 6 == 0:
        variant = ImageOps.autocontrast(variant, cutoff=1)
        transforms.append("autocontrast")
    if index % 7 == 0:
        variant = center_crop(variant, 0.94)
        transforms.append("center crop")
    if index % 8 == 0:
        variant = ImageOps.mirror(variant)
        transforms.append("horizontal flip")
    if index % 9 == 0:
        angle = -2 if index % 18 == 0 else 2
        variant = variant.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=0)
        transforms.append(f"rotate {angle} deg")

    if not transforms:
        variant = ImageEnhance.Contrast(variant).enhance(1.0 + index * 0.01)
        transforms.append("subtle contrast")

    return variant, ", ".join(transforms)


def main() -> None:
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(f"Missing source image: {SOURCE_IMAGE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE_IMAGE).convert("RGB").resize((512, 512))

    rows = []
    for index in range(1, 37):
        variant, transform_note = make_variant(source, index)
        filename = f"case_{index:02d}.png"
        path = OUTPUT_DIR / filename
        variant.save(path)
        rows.append(
            {
                "case_id": f"CASE-{index:02d}",
                "title": f"Demo X-ray Case {index:02d}",
                "path": str(path).replace("\\", "/"),
                "source": "Derived from Wikimedia pneumonia X-ray sample",
                "note": transform_note,
            }
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "title", "path", "source", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} demo cases to {OUTPUT_DIR}")
    print(f"Wrote manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
