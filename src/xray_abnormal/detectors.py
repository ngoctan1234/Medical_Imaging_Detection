from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from xray_abnormal.localization import LesionBox, boxes_from_heatmap, infer_lung_region


class LesionDetector(Protocol):
    def detect(self, image: Image.Image, label_hint: str, confidence_hint: float) -> list[LesionBox]:
        ...


@dataclass
class GradCAMProposalDetector:
    heatmap: np.ndarray
    threshold: float = 0.58
    max_boxes: int = 3

    def detect(self, image: Image.Image, label_hint: str, confidence_hint: float) -> list[LesionBox]:
        return boxes_from_heatmap(
            self.heatmap,
            label_hint,
            confidence_hint,
            threshold=self.threshold,
            max_boxes=self.max_boxes,
        )


class YOLOv8LesionDetector:
    def __init__(self, weights_path: str | Path, image_size: int = 640, confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics to use YOLOv8 lesion detection.") from exc
        self.model = YOLO(str(weights_path))
        self.image_size = image_size
        self.confidence = confidence

    def detect(self, image: Image.Image, label_hint: str, confidence_hint: float) -> list[LesionBox]:
        results = self.model.predict(image, imgsz=self.image_size, conf=self.confidence, verbose=False)
        width, height = image.size
        boxes: list[LesionBox] = []
        for result in results:
            for raw_box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in raw_box.xyxy[0].tolist()]
                conf = float(raw_box.conf[0])
                cls_idx = int(raw_box.cls[0])
                label = result.names.get(cls_idx, label_hint)
                boxes.append(
                    LesionBox(
                        label=label,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        lung_region=infer_lung_region((x1, y1, x2, y2), (height, width)),
                        area_ratio=float(((x2 - x1) * (y2 - y1)) / (width * height)),
                    )
                )
        return boxes


class DETRLesionDetector:
    def __init__(self, model_name_or_path: str, threshold: float = 0.25) -> None:
        try:
            from transformers import AutoImageProcessor, DetrForObjectDetection
        except ImportError as exc:
            raise ImportError("Install transformers to use DETR lesion detection.") from exc
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = DetrForObjectDetection.from_pretrained(model_name_or_path)
        self.threshold = threshold

    def detect(self, image: Image.Image, label_hint: str, confidence_hint: float) -> list[LesionBox]:
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        width, height = image.size
        boxes: list[LesionBox] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=True):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label_text = self.model.config.id2label.get(int(label), label_hint)
            boxes.append(
                LesionBox(
                    label=label_text,
                    confidence=float(score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    lung_region=infer_lung_region((x1, y1, x2, y2), (height, width)),
                    area_ratio=float(((x2 - x1) * (y2 - y1)) / (width * height)),
                )
            )
        return boxes
