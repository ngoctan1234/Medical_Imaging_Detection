from __future__ import annotations

from dataclasses import dataclass

from xray_abnormal.localization import LesionBox


VINBIG_CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]


@dataclass(frozen=True)
class ClinicalEvidence:
    label: str
    anatomy: str
    visual_evidence: str
    clinical_evidence: str
    clinical_reasoning: str


DISEASE_TO_EVIDENCE = {
    "aortic enlargement": (
        "Aortic knuckle",
        "attention over the aortic knuckle or upper mediastinal contour",
        "prominence of the aortic contour",
        "Aortic contour enlargement is a radiographic sign associated with aortic enlargement.",
    ),
    "atelectasis": (
        "Lung volume / fissure-adjacent region",
        "attention over a focal lung zone with possible volume-loss pattern",
        "regional opacity or volume-loss pattern",
        "Linear or regional opacity with volume loss can support atelectasis.",
    ),
    "calcification": (
        "Focal calcified density",
        "attention over a small high-density focus",
        "focal dense opacity compatible with calcification",
        "Calcified lesions typically appear as sharply dense foci on radiograph.",
    ),
    "cardiomegaly": (
        "Cardiac silhouette",
        "attention over the cardiac silhouette",
        "prominence of the cardiac contour",
        "Cardiac silhouette enlargement is consistent with cardiomegaly.",
    ),
    "consolidation": (
        "Lung parenchyma",
        "attention over a focal air-space opacity",
        "focal parenchymal opacity",
        "Air-space opacity may represent consolidation in the appropriate clinical context.",
    ),
    "ild": (
        "Interstitial lung markings",
        "attention over bilateral interstitial lung regions",
        "diffuse or regional interstitial pattern",
        "Interstitial prominence can support an ILD pattern when correlated with radiologist review.",
    ),
    "infiltration": (
        "Lung parenchyma",
        "attention over a subtle parenchymal opacity",
        "ill-defined parenchymal opacity",
        "Ill-defined opacity may reflect infiltrative change in the correct clinical context.",
    ),
    "lung opacity": (
        "Lung parenchyma",
        "attention over an abnormal lung opacity",
        "increased lung parenchymal opacity",
        "Increased parenchymal opacity is a direct radiographic correlate of lung opacity.",
    ),
    "nodule/mass": (
        "Focal lung lesion",
        "attention over a focal rounded opacity",
        "focal nodular or mass-like opacity",
        "A focal rounded opacity can support nodule or mass detection.",
    ),
    "mass": (
        "Focal lung lesion",
        "attention over a focal mass-like opacity",
        "focal mass-like opacity",
        "A focal mass-like opacity can support nodule or mass detection.",
    ),
    "nodule": (
        "Focal lung lesion",
        "attention over a focal nodular opacity",
        "focal nodular opacity",
        "A focal rounded opacity can support nodule or mass detection.",
    ),
    "other lesion": (
        "Abnormal focal region",
        "attention over a non-specific abnormal region",
        "non-specific radiographic abnormality",
        "The finding is non-specific and requires radiologist characterization.",
    ),
    "pleural effusion": (
        "Lower lung zone / costophrenic angle",
        "attention over the lower hemithorax or costophrenic angle",
        "lower-zone pleural opacity or blunting pattern",
        "Dependent lower-zone pleural opacity is consistent with pleural effusion.",
    ),
    "effusion": (
        "Lower lung zone / costophrenic angle",
        "attention over the lower hemithorax or costophrenic angle",
        "lower-zone pleural opacity or blunting pattern",
        "Dependent lower-zone pleural opacity is consistent with pleural effusion.",
    ),
    "pleural thickening": (
        "Pleural surface",
        "attention along the pleural margin",
        "pleural-based thickening pattern",
        "Pleural margin thickening can support pleural thickening.",
    ),
    "pneumothorax": (
        "Pleural apex",
        "attention near the pleural apex or peripheral pleural line",
        "apical or peripheral pleural abnormality",
        "A peripheral pleural line with absent lung markings can suggest pneumothorax.",
    ),
    "pulmonary fibrosis": (
        "Fibrotic lung regions",
        "attention over reticular or fibrotic lung regions",
        "reticular/fibrotic opacity pattern",
        "Reticular opacities and architectural distortion can support pulmonary fibrosis.",
    ),
    "fibrosis": (
        "Fibrotic lung regions",
        "attention over reticular or fibrotic lung regions",
        "reticular/fibrotic opacity pattern",
        "Reticular opacities and architectural distortion can support pulmonary fibrosis.",
    ),
    "no finding": (
        "No focal abnormal anatomy",
        "no high-confidence focal attention pattern",
        "no model-supported focal abnormality",
        "No finding means no predicted class exceeded the selected abnormality threshold.",
    ),
}


def clinical_evidence_for(label: str, boxes: list[LesionBox]) -> ClinicalEvidence:
    normalized = label.strip().lower()
    anatomy, visual, evidence, reasoning = DISEASE_TO_EVIDENCE.get(
        normalized,
        (
            "Relevant radiographic region",
            "attention over the model-highlighted region",
            "model-highlighted radiographic abnormality",
            "The highlighted region should be reviewed by a radiologist for clinical correlation.",
        ),
    )
    if boxes:
        region = boxes[0].lung_region
        visual = f"{visual}; strongest localized proposal in the {region}"
        evidence = f"{evidence} in the {region}"
    return ClinicalEvidence(
        label=label,
        anatomy=anatomy,
        visual_evidence=visual,
        clinical_evidence=evidence,
        clinical_reasoning=reasoning,
    )


def agreement_flag(agreement: str | None) -> str:
    if agreement is None:
        return "Agreement unavailable: single-model fallback is active. Ensemble reliability cannot be estimated."
    if agreement.lower() == "low":
        return "Low agreement: treat prediction as lower reliability and prioritize radiologist review."
    return f"{agreement} agreement between ensemble members."
