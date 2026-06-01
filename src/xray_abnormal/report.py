from __future__ import annotations

import pandas as pd

from xray_abnormal.localization import LesionBox


def generate_report(predictions: pd.DataFrame, boxes: list[LesionBox], threshold: float) -> dict[str, str]:
    positive = predictions[predictions["calibrated_probability"] >= threshold].copy()
    top = predictions.iloc[0]

    if positive.empty:
        findings = "No model finding exceeded the selected abnormality threshold. Grad-CAM attention remains low-specificity."
        impression = "No high-confidence radiographic abnormality detected by the AI model."
        recommendations = "Correlate with clinical history and prior imaging. Radiologist review remains required."
        return {"findings": findings, "impression": impression, "recommendations": recommendations}

    finding_lines = []
    for row in positive.head(5).itertuples():
        finding_lines.append(
            f"{row.label}: calibrated confidence {row.calibrated_probability:.2f}, "
            f"risk {row.risk_level.lower()}, uncertainty {row.uncertainty.lower()}."
        )
    if boxes:
        location_text = "; ".join(
            f"{box.label} at {box.lung_region} ({box.x1},{box.y1})-({box.x2},{box.y2})"
            for box in boxes
        )
        finding_lines.append(f"Suspicious localized regions: {location_text}.")

    findings = " ".join(finding_lines)
    impression = (
        f"AI impression favors {top['label']} with calibrated confidence "
        f"{top['calibrated_probability']:.2f} and {top['risk_level'].lower()} risk."
    )
    recommendations = (
        "Recommend radiologist confirmation, comparison with prior chest imaging, and clinical correlation. "
        "If symptoms or risk factors are present, consider follow-up imaging according to local protocol."
    )
    return {"findings": findings, "impression": impression, "recommendations": recommendations}
