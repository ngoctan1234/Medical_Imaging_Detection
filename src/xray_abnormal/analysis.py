from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from xray_abnormal.calibration import (
    binary_entropy,
    calibrate_probabilities,
    risk_level,
    severity_level,
    uncertainty_label,
)
from xray_abnormal.detectors import GradCAMProposalDetector
from xray_abnormal.localization import LesionBox, draw_boxes
from xray_abnormal.pretrained import predict_pretrained_with_heatmap
from xray_abnormal.report import generate_report


@dataclass
class CaseAnalysis:
    predictions: pd.DataFrame
    heatmap: np.ndarray
    gradcam_overlay: np.ndarray
    detection_overlay: np.ndarray
    boxes: list[LesionBox]
    report: dict[str, str]
    risk: str
    uncertainty: str
    primary_label: str
    primary_confidence: float
    explanation: str


def analyze_pretrained_case(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    threshold: float,
    temperature: float = 1.4,
) -> CaseAnalysis:
    raw_predictions, heatmap, gradcam_overlay = predict_pretrained_with_heatmap(model, image, device)
    df = pd.DataFrame(raw_predictions)
    calibrated = calibrate_probabilities(df["probability"].to_numpy(), temperature=temperature)
    df["calibrated_probability"] = calibrated
    df["uncertainty_score"] = df["calibrated_probability"].map(binary_entropy)
    df["uncertainty"] = df["calibrated_probability"].map(uncertainty_label)
    df["risk_level"] = df["calibrated_probability"].map(risk_level)
    df["severity"] = df["calibrated_probability"].map(severity_level)
    df = df.sort_values("calibrated_probability", ascending=False).reset_index(drop=True)

    primary_label = str(df.iloc[0]["label"])
    primary_confidence = float(df.iloc[0]["calibrated_probability"])
    detector = GradCAMProposalDetector(heatmap=heatmap)
    boxes = detector.detect(image, primary_label, primary_confidence)

    image_rgb = np.asarray(image.convert("RGB"))
    image_rgb = cv2.resize(image_rgb, (heatmap.shape[1], heatmap.shape[0]))
    detection_overlay = draw_boxes(image_rgb, boxes)
    report = generate_report(df, boxes, threshold, visual_method="fallback CAM heatmap")

    if boxes:
        region = boxes[0].lung_region
        explanation = (
            f"The single-model fallback prioritized {primary_label}; the strongest visual evidence "
            f"overlaps the {region}. Calibrated score is {primary_confidence:.2f}; "
            f"uncertainty is {uncertainty_label(primary_confidence).lower()}. "
            "ConvNeXtV2 EigenCAM and RAD-DINO Attention Rollout should replace this fallback when checkpoints are available."
        )
    else:
        explanation = (
            f"The single-model fallback prioritized {primary_label} based on global image features, but no compact suspicious "
            "region exceeded the localization threshold. Ensemble agreement is unavailable in fallback mode."
        )

    return CaseAnalysis(
        predictions=df,
        heatmap=heatmap,
        gradcam_overlay=gradcam_overlay,
        detection_overlay=detection_overlay,
        boxes=boxes,
        report=report,
        risk=risk_level(primary_confidence),
        uncertainty=uncertainty_label(primary_confidence),
        primary_label=primary_label,
        primary_confidence=primary_confidence,
        explanation=explanation,
    )
