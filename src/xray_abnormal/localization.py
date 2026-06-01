from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class LesionBox:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    lung_region: str
    area_ratio: float

    @property
    def coordinates(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def infer_lung_region(box: tuple[int, int, int, int], image_shape: tuple[int, int]) -> str:
    height, width = image_shape
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    side = "Right lung" if cx < width / 2 else "Left lung"
    if cy < height * 0.33:
        zone = "upper zone"
    elif cy < height * 0.66:
        zone = "middle zone"
    else:
        zone = "lower zone"
    return f"{side}, {zone}"


def boxes_from_heatmap(
    heatmap: np.ndarray,
    label: str,
    confidence: float,
    threshold: float = 0.58,
    max_boxes: int = 3,
    min_area_ratio: float = 0.008,
) -> list[LesionBox]:
    heatmap = np.nan_to_num(heatmap).astype(np.float32)
    height, width = heatmap.shape[:2]
    binary = (heatmap >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[LesionBox] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = float((w * h) / (width * height))
        if area_ratio < min_area_ratio:
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        boxes.append(
            LesionBox(
                label=label,
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                lung_region=infer_lung_region((x1, y1, x2, y2), (height, width)),
                area_ratio=area_ratio,
            )
        )
        if len(boxes) >= max_boxes:
            break

    if boxes:
        return boxes

    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    box_size = max(28, int(min(width, height) * 0.22))
    x1 = max(0, int(peak_x - box_size / 2))
    y1 = max(0, int(peak_y - box_size / 2))
    x2 = min(width, x1 + box_size)
    y2 = min(height, y1 + box_size)
    return [
        LesionBox(
            label=label,
            confidence=float(confidence),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            lung_region=infer_lung_region((x1, y1, x2, y2), (height, width)),
            area_ratio=float(((x2 - x1) * (y2 - y1)) / (width * height)),
        )
    ]


def draw_boxes(image_rgb: np.ndarray, boxes: list[LesionBox]) -> np.ndarray:
    image = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for box in boxes:
        color = (214, 75, 58) if box.confidence >= 0.65 else (226, 158, 48)
        draw.rectangle(box.coordinates, outline=color, width=3)
        text = f"{box.label} {box.confidence:.2f}"
        text_box = draw.textbbox((box.x1, box.y1), text, font=font)
        draw.rectangle((text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2), fill=color)
        draw.text((box.x1, max(0, box.y1 - 14)), text, fill=(255, 255, 255), font=font)

    return np.asarray(image)
