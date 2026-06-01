# Research-Grade Chest X-ray AI Workstation

## System Architecture

```text
Streamlit Workstation
  ├── Patient / study input
  ├── Demo case selector or image upload
  ├── Imaging review panels
  ├── AI findings and probability dashboard
  ├── Structured radiology-style report
  └── Model card and performance dashboard

Backend analysis service
  ├── Classification model
  │   └── TorchXRayVision DenseNet121 pretrained model
  ├── Explainability
  │   └── Grad-CAM heatmap
  ├── Localization
  │   ├── Grad-CAM connected-component lesion proposal
  │   ├── Optional YOLOv8 adapter
  │   └── Optional DETR adapter
  ├── Calibration and uncertainty
  │   ├── Temperature scaling approximation
  │   ├── Entropy-based uncertainty
  │   └── Risk/severity buckets
  └── Report generator
      ├── Findings
      ├── Impression
      └── Recommendations
```

## Updated UI Mockup

```text
Chest X-ray Abnormality Detection Workstation
Clinical safety warning

Patient Information
Patient ID | Age | Sex | View | Demo case / Upload image

AI Findings
Primary Diagnosis | Calibrated Score | Uncertainty | Risk Badge
Short explanation of model decision

Imaging Review
Original X-ray            | AI Detection Result
Grad-CAM                  | Bounding Box Overlay

Localization
Finding | Confidence | x1 | y1 | x2 | y2 | Affected Lung Region | Area Ratio

Confidence Scores         | Probability Chart

Radiology Report
Findings                  | Impression              | Recommendations

Model Information         | Performance Dashboard
```

## Folder Structure

```text
configs/
  default.yaml
  model_card.yaml
data/
  demo_cases/
  processed/
  raw/
docs/
  ARCHITECTURE.md
scripts/
  generate_demo_cases.py
  test_pretrained.py
src/
  app.py
  xray_abnormal/
    analysis.py
    calibration.py
    data.py
    detectors.py
    gradcam.py
    infer.py
    localization.py
    model.py
    pretrained.py
    report.py
    train.py
    utils.py
```

## Suggested Libraries

- `torch`, `torchvision`: classification and training
- `torchxrayvision`: pretrained chest X-ray models
- `opencv-python`: heatmap and lesion proposal processing
- `streamlit`: clinical prototype UI
- `ultralytics`: optional YOLOv8 lesion detector
- `transformers`: optional DETR lesion detector
- `pydicom`, `dicom2nifti`: future DICOM ingestion
- `scikit-learn`: metrics, calibration validation

## Future Improvement Roadmap

1. Train a dedicated detector on VinDr-CXR, RSNA Pneumonia, or NIH bounding box subset.
2. Replace Grad-CAM proposal boxes with YOLOv8/DETR detections and evaluate mAP.
3. Fit calibration temperature on a held-out validation cohort.
4. Add external validation across multiple hospitals and scanners.
5. Add subgroup analysis by age, sex, view position, and acquisition protocol.
6. Add DICOM support with windowing, metadata extraction, and de-identification.
7. Add audit logging, model version pinning, and reproducibility reports.
8. Run radiologist reader study for clinical usability.
