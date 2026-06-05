# Chest X-ray 2D Abnormality Detection Workstation

Research-grade prototype cho phat hien bat thuong tren anh X-ray nguc 2D, kem Grad-CAM, lesion proposal bounding boxes, calibrated confidence, uncertainty estimation va radiology-style report.

## Chuc nang

- Train binary classification: `normal` vs `abnormal`
- Ho tro multi-label neu CSV co nhieu cot nhan
- Inference tren 1 anh X-ray 2D
- Grad-CAM heatmap overlay len anh goc
- Streamlit app de upload anh va xem ket qua
- Che do pretrained bang TorchXRayVision de chay thu ngay khi chua co checkpoint tu train rieng
- Lesion proposal bounding boxes tu Grad-CAM connected components
- YOLOv8 va DETR detector adapter de cam model localization that
- Confidence calibration, uncertainty, risk/severity level
- Structured clinical report: Findings, Impression, Recommendations
- Model card va performance dashboard

## Cau truc

```text
.
├── configs/
│   └── default.yaml
│   └── model_card.yaml
├── data/
│   ├── raw/
│   └── processed/
│   └── demo_cases/
├── docs/
│   └── ARCHITECTURE.md
├── outputs/
├── src/
│   ├── xray_abnormal/
│   │   ├── analysis.py
│   │   ├── calibration.py
│   │   ├── data.py
│   │   ├── detectors.py
│   │   ├── gradcam.py
│   │   ├── infer.py
│   │   ├── localization.py
│   │   ├── model.py
│   │   ├── pretrained.py
│   │   ├── report.py
│   │   ├── train.py
│   │   └── utils.py
│   └── app.py
├── scripts/
│   ├── generate_demo_cases.py
│   └── test_pretrained.py
├── requirements.txt
└── pyproject.toml
```

## Cai dat

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Dinh dang du lieu

Tao file CSV, vi du `data/processed/train.csv`:

```csv
image_path,abnormal
data/raw/img001.png,0
data/raw/img002.png,1
```

Voi multi-label:

```csv
image_path,atelectasis,cardiomegaly,effusion,pneumonia
data/raw/img001.png,0,1,0,0
data/raw/img002.png,1,0,1,0
```

Cap nhat `configs/default.yaml`:

```yaml
data:
  train_csv: data/processed/train.csv
  val_csv: data/processed/val.csv
  image_column: image_path
  label_columns: [abnormal]
```

## Train

```powershell
python -m xray_abnormal.train --config configs/default.yaml
```

Checkpoint tot nhat se duoc luu vao `outputs/checkpoints/best.pt`.

## Inference va tao heatmap

```powershell
python -m xray_abnormal.infer `
  --checkpoint outputs/checkpoints/best.pt `
  --image data/raw/example.png `
  --config configs/default.yaml `
  --output outputs/heatmap_example.png
```

## ConvNeXtV2 + RAD-DINO ensemble target

Workflow clinical muc tieu:

```text
Chest X-ray -> Disease Detection -> Explainability -> Clinical Evidence -> Clinical Reasoning -> Radiology Report
```

Ensemble dung weighted averaging:

```text
ensemble_probs = w * convnext_probs + (1 - w) * raddino_probs
```

De toi uu `w` tren validation AUROC, export CSV tu notebook voi cac cot:

- `y_<class>` cho ground truth 15 class VinBigData
- `convnext_<class>` cho ConvNeXtV2 probability
- `raddino_<class>` cho RAD-DINO probability

Sau do chay:

```powershell
$env:PYTHONPATH='src'
python scripts/optimize_ensemble_weight.py `
  --input outputs/validation_predictions.csv `
  --output outputs/ensemble_weight.yaml
```

Explainability target:

- ConvNeXtV2: EigenCAM
- RAD-DINO: Attention Rollout
- Demo hien tai: fallback CAM heatmap, khong xem la primary explainability khi viet bao cao khoa hoc.

## Chay app

```powershell
streamlit run src/app.py
```

Trong app, chon `Fallback pretrained demo` de dung model chest X-ray da train san. Chon `Local checkpoint` khi da wire checkpoint ConvNeXtV2/RAD-DINO rieng.

Neu khong co anh de upload, app co san muc `Demo cases` va se tu chay voi bo case mau trong `data/demo_cases`.

Tao lai bo 36 demo case:

```powershell
python scripts/generate_demo_cases.py
```

## Architecture va roadmap

Xem tai lieu chi tiet:

```text
docs/ARCHITECTURE.md
```

Detector hien tai mac dinh la Grad-CAM lesion proposal de app chay duoc ngay. Khi co checkpoint detection that, co the dung:

- `YOLOv8LesionDetector` trong `src/xray_abnormal/detectors.py`
- `DETRLesionDetector` trong `src/xray_abnormal/detectors.py`

## Ghi chu y khoa

Project nay la khung nghien cuu/thu nghiem, khong phai thiet bi chan doan y te. Ket qua can duoc bac si chuyen khoa xac nhan.
