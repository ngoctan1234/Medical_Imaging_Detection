from __future__ import annotations

import pandas as pd

from xray_abnormal.clinical_evidence import agreement_flag, clinical_evidence_for
from xray_abnormal.localization import LesionBox


def generate_report(
    predictions: pd.DataFrame,
    boxes: list[LesionBox],
    threshold: float,
    agreement: str | None = None,
    visual_method: str = "fallback CAM heatmap",
) -> dict[str, str]:
    positive = predictions[predictions["calibrated_probability"] >= threshold].copy()
    top = predictions.iloc[0]
    top_label = str(top["label"])
    top_confidence = float(top["calibrated_probability"])
    top_risk = str(top["risk_level"]).lower()
    top_uncertainty = str(top["uncertainty"]).lower()

    if positive.empty:
        findings = f"No model finding exceeded the selected abnormality threshold. {visual_method} remains low-specificity."
        impression = "No high-confidence radiographic abnormality detected by the AI model."
        recommendations = "Correlate with clinical history and prior imaging. Radiologist review remains required."
        flag = agreement_flag(agreement)
        explainability = (
            "Explainability note: AI did not identify a finding above the selected threshold. "
            f"{visual_method} is low-specificity and should not be interpreted as a focal lesion. "
            "Final interpretation should rely on radiologist review and clinical context."
        )
        return {
            "prediction": "No finding above threshold",
            "visual_evidence": f"{visual_method}: no high-confidence focal evidence.",
            "clinical_evidence": "No model-supported focal abnormality.",
            "clinical_reasoning": "No predicted class exceeded the selected abnormality threshold.",
            "agreement": agreement or "Unavailable",
            "flag": flag,
            "findings": findings,
            "impression": impression,
            "recommendations": recommendations,
            "explainability": explainability,
            "disclaimer": "AI-generated report for research support only. Radiologist review is required.",
        }

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
        main_region = boxes[0].lung_region
        localization_sentence = f"{visual_method} and proposal box focus mainly on the {main_region}."
    else:
        localization_sentence = f"{visual_method} is diffuse; no compact focal proposal exceeded the localization threshold."

    evidence = clinical_evidence_for(top_label, boxes)
    flag = agreement_flag(agreement)
    findings = " ".join(finding_lines)
    impression = (
        f"AI impression favors {top_label} with calibrated confidence "
        f"{top_confidence:.2f} and {top_risk} risk."
    )
    recommendations = (
        "Recommend radiologist confirmation, comparison with prior chest imaging, and clinical correlation. "
        "If symptoms or risk factors are present, consider follow-up imaging according to local protocol."
    )
    explainability = (
        f"Explainability note: AI prioritizes {top_label} "
        f"(calibrated confidence {top_confidence:.2f}, {top_risk} risk, {top_uncertainty} uncertainty). "
        f"{localization_sentence} Clinical evidence: {evidence.clinical_evidence}. "
        f"Reasoning: {evidence.clinical_reasoning} "
        "Use this as decision support only; confirm with full radiographic review and clinical history."
    )
    return {
        "prediction": f"{top_label} ({top_confidence:.0%})",
        "visual_evidence": f"{visual_method}: {evidence.visual_evidence}.",
        "clinical_evidence": evidence.clinical_evidence,
        "clinical_reasoning": evidence.clinical_reasoning,
        "agreement": agreement or "Unavailable",
        "flag": flag,
        "findings": findings,
        "impression": impression,
        "recommendations": recommendations,
        "explainability": explainability,
        "disclaimer": "AI-generated report for research support only. Radiologist review is required.",
    }
