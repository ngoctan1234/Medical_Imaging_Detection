# Conclusion - Clinical Chest X-ray AI Assistant

## 1. Mục tiêu project

Project này xây dựng một hệ thống hỗ trợ đọc ảnh X-ray ngực theo hướng explainable AI. Luồng xử lý được thiết kế gần với workflow của bác sĩ chẩn đoán hình ảnh:

```text
Chest X-ray image
-> Disease detection
-> Explainability heatmap / attention evidence
-> Clinical evidence extraction
-> Clinical reasoning
-> Structured radiology-style report
```

Mục tiêu chính không chỉ là dự đoán bệnh, mà còn giải thích vì sao model đưa ra dự đoán đó, vùng ảnh nào liên quan, bằng chứng lâm sàng là gì, và cần diễn giải thế nào trong báo cáo.

Project hiện có 2 tầng:

- Demo đang chạy: dùng pretrained TorchXRayVision để inference nhanh, có calibrated probability, fallback CAM heatmap, lesion proposal, clinical evidence mapping và structured report.
- Target research system: ConvNeXtV2 + RAD-DINO ensemble cho 15 class VinBigData, dùng EigenCAM và Attention Rollout làm explainability chính.

## 2. Trạng thái giao diện hiện tại

Giao diện Streamlit trong `src/app.py` hiện đã đồng bộ tốt với sườn demo/research prototype, và đã được chỉnh lại để ưu tiên vùng xem ảnh vì đây là phần quan trọng nhất với bác sĩ.

Các phần đã khớp với kiến trúc:

- Sidebar có upload ảnh, chọn demo case, thông tin patient/study và Advanced AI Protocol.
- Dashboard chính thể hiện workflow clinical: input image, AI analysis, ensemble target, clinical evidence và report.
- Vùng `Chest X-ray Analysis` là vùng chính, full-width, không còn bị summary ép nhỏ bên cạnh.
- Ảnh viewer dùng `object-fit: contain` để giữ toàn bộ phim X-ray, tránh crop ảnh.
- Viewer hiển thị song song ảnh gốc và ảnh được chọn để so sánh trực tiếp.
- Control `Image view` dùng segmented control native của Streamlit để đổi view trong cùng app:

```text
Original
Heatmap
Detection
Lateral
```

- `Heatmap` hiển thị fallback CAM overlay.
- `Detection` hiển thị detection/proposal overlay.
- `Lateral` hiện là placeholder vì demo case chưa có ảnh lateral riêng.
- Report Summary được đưa xuống dưới vùng ảnh, gọn thành các ô: Primary Finding, Confidence, Risk Level, Top Differential.
- UI hiển thị AI Findings, Explainability, AI Interpretation, Conclusion, Structured Report, Lesion Coordinates và Model Metrics.
- Model cards thể hiện rõ target ConvNeXtV2, RAD-DINO và Weighted Ensemble.

Các phần chưa hoàn toàn đồng bộ với target research system:

- App hiện chỉ chạy thật ở chế độ `Fallback pretrained demo`.
- Nhánh `Local checkpoint` đã có input checkpoint/config nhưng hiện dừng với thông báo ConvNeXtV2/RAD-DINO chưa được wire vào demo.
- ConvNeXtV2/RAD-DINO ensemble mới có module logic trong `src/xray_abnormal/ensemble.py`, chưa được gọi trong UI inference path.
- EigenCAM và Attention Rollout mới là định hướng target, chưa có implementation thật trong pipeline app.
- Các nút `Export PDF` và `Save Case` đang là UI placeholder, chưa có backend lưu file hoặc xuất PDF.
- Metrics trong model card là metadata demo/research, chưa phải kết quả validation được tính tự động từ run hiện tại.

Kết luận ngắn: giao diện hiện đã đồng bộ với sườn demo và đã được tối ưu theo hướng workstation đọc phim: ảnh lớn hơn, không crop, dễ chuyển Original/Heatmap/Detection/Lateral, summary nằm dưới ảnh. Phần chưa hoàn chỉnh vẫn là tầng model thật và các chức năng export/save.

## 3. Input của hệ thống

### 3.1. Input khi chạy app

Người dùng có thể đưa vào:

- Ảnh X-ray ngực dạng JPG/PNG/BMP qua Streamlit UI.
- Demo case có sẵn trong `data/demo_cases`.
- Các thông tin phụ như patient ID, age, sex, view position, model source, confidence threshold, temperature scaling, top findings, checkpoint path và config path.

File app chính:

```text
src/app.py
```

### 3.2. Input khi train model local

Module train hỗ trợ CSV gồm:

```csv
image_path,label_1,label_2,...
data/raw/img001.png,0,1,...
```

Trong cấu hình mặc định:

```text
configs/default.yaml
```

Các trường quan trọng:

- `data.train_csv`: đường dẫn file train CSV.
- `data.val_csv`: đường dẫn file validation CSV.
- `data.image_column`: tên cột chứa đường dẫn ảnh.
- `data.label_columns`: danh sách label cần học.

### 3.3. Input cho ensemble VinBigData

Target dataset là VinBigData Chest X-ray với 15 disease classes:

```text
Aortic enlargement
Atelectasis
Calcification
Cardiomegaly
Consolidation
ILD
Infiltration
Lung Opacity
Nodule/Mass
Other lesion
Pleural effusion
Pleural thickening
Pneumothorax
Pulmonary fibrosis
No finding
```

Để tối ưu ensemble weight, cần CSV validation prediction có các cột:

```text
y_<class>
convnext_<class>
raddino_<class>
```

Script xử lý:

```text
scripts/optimize_ensemble_weight.py
```

Output script:

```text
outputs/ensemble_weight.yaml
```

## 4. Output của hệ thống

Sau khi chạy inference ở chế độ fallback demo, hệ thống tạo ra:

- Bảng xác suất bệnh.
- Calibrated probability.
- Uncertainty score.
- Risk level.
- Severity level.
- Fallback CAM heatmap.
- Detection/proposal overlay từ heatmap.
- Clinical evidence.
- Clinical reasoning.
- Structured radiology-style report.

Dataclass gom kết quả phân tích:

```text
src/xray_abnormal/analysis.py
CaseAnalysis
```

Các field chính:

```text
predictions
heatmap
gradcam_overlay
detection_overlay
boxes
report
risk
uncertainty
primary_label
primary_confidence
explanation
```

Report cuối cùng là dictionary gồm:

```text
prediction
visual_evidence
clinical_evidence
clinical_reasoning
agreement
flag
findings
impression
recommendations
explainability
disclaimer
```

Module tạo report:

```text
src/xray_abnormal/report.py
```

## 5. Kiến trúc code

### 5.1. Streamlit UI

File:

```text
src/app.py
```

Vai trò:

- Tạo giao diện Radiology AI Workstation.
- Cho phép upload ảnh hoặc chọn demo case.
- Nhận metadata patient/study và tham số inference.
- Gọi inference pipeline fallback.
- Hiển thị ảnh gốc và view được chọn trong viewer lớn.
- Cho phép đổi view bằng `Image view`: Original, Heatmap, Detection, Lateral.
- Hiển thị primary finding, confidence, risk và top differential dưới vùng ảnh.
- Hiển thị Clinical Explainability Report và structured report.
- Hiển thị target model architecture để phân biệt demo fallback với research target.

### 5.2. Pipeline phân tích ảnh

File:

```text
src/xray_abnormal/analysis.py
```

Hàm chính:

```python
analyze_pretrained_case(model, image, device, threshold, temperature)
```

Luồng xử lý hiện tại:

```text
image
-> predict_pretrained_with_heatmap
-> calibrate_probabilities
-> uncertainty/risk/severity
-> GradCAMProposalDetector
-> draw_boxes
-> generate_report
-> CaseAnalysis
```

Đây là pipeline demo fallback đang được app gọi thật.

### 5.3. Pretrained fallback inference

File:

```text
src/xray_abnormal/pretrained.py
```

Vai trò:

- Load model pretrained từ TorchXRayVision.
- Preprocess ảnh X-ray về grayscale, normalize, center crop, resize.
- Chạy model để lấy disease probabilities.
- Tạo heatmap bằng Grad-CAM/fallback CAM.

Các hàm chính:

```python
load_torchxrayvision_model(...)
predict_pretrained_with_heatmap(...)
```

Lưu ý: fallback CAM dùng cho demo và minh họa visual evidence, không phải explainability chính của target research system.

### 5.4. Model local

Files:

```text
src/xray_abnormal/model.py
src/xray_abnormal/train.py
src/xray_abnormal/infer.py
```

Vai trò:

- Định nghĩa model classification local.
- Hỗ trợ training/inference với checkpoint tự train.
- `src/app.py` đã có UI nhập checkpoint path, nhưng nhánh ConvNeXtV2/RAD-DINO target chưa được wire vào inference runtime.

### 5.5. Calibration, uncertainty, risk

File:

```text
src/xray_abnormal/calibration.py
```

Vai trò:

- Temperature scaling cho probability.
- Tính binary entropy làm uncertainty score.
- Gán uncertainty label.
- Gán risk level.
- Gán severity level.

Logic chính:

```text
raw probability
-> calibrated probability
-> entropy uncertainty
-> risk/severity labels
```

### 5.6. Localization và proposal boxes

Files:

```text
src/xray_abnormal/localization.py
src/xray_abnormal/detectors.py
```

Vai trò:

- Tạo lesion proposal box từ heatmap.
- Map box sang vùng phổi tương đối.
- Vẽ bounding box lên ảnh.

Trong demo fallback, detector chính là:

```python
GradCAMProposalDetector
```

Các adapter detector có thể mở rộng:

```text
YOLOv8LesionDetector
DETRLesionDetector
```

### 5.7. Explainability

File fallback:

```text
src/xray_abnormal/gradcam.py
```

Vai trò:

- Cài đặt Grad-CAM/fallback CAM.
- Overlay heatmap lên ảnh gốc.

Target research:

- ConvNeXtV2 dùng EigenCAM.
- RAD-DINO dùng Attention Rollout.

Quy ước quan trọng:

```text
Grad-CAM/fallback CAM không dùng làm primary explainability trong báo cáo research,
trừ khi được yêu cầu rõ hoặc đang chạy demo fallback.
```

### 5.8. Clinical evidence mapping

File:

```text
src/xray_abnormal/clinical_evidence.py
```

Vai trò:

- Map disease label sang anatomical region.
- Tạo visual evidence.
- Tạo clinical evidence.
- Tạo clinical reasoning.
- Tạo agreement/safety flag.

Mục tiêu là giữ traceability:

```text
prediction -> visual evidence -> clinical evidence -> reasoning -> report
```

### 5.9. Ensemble ConvNeXtV2 + RAD-DINO

File:

```text
src/xray_abnormal/ensemble.py
```

Vai trò:

- Weighted averaging giữa ConvNeXtV2 và RAD-DINO.
- Tính macro AUROC.
- Grid search để tối ưu ensemble weight.
- Tính agreement score.
- Trả về top findings.

Công thức:

```text
ensemble_probs = w * convnext_probs + (1 - w) * raddino_probs
```

Agreement:

```text
agreement_score = 1 - mean(abs(convnext_probs - raddino_probs))
```

Label:

```text
High: score >= 0.90
Medium: score >= 0.75
Low: otherwise
```

Script tối ưu weight:

```text
scripts/optimize_ensemble_weight.py
```

Trạng thái hiện tại: ensemble logic đã có, nhưng app chưa gọi ConvNeXtV2/RAD-DINO runtime ensemble khi chạy inference.

### 5.10. Report generation

File:

```text
src/xray_abnormal/report.py
```

Vai trò:

- Chọn các finding vượt threshold.
- Lấy primary finding.
- Ghép confidence, risk, uncertainty.
- Thêm localization sentence nếu có box.
- Gọi clinical evidence mapping.
- Sinh Findings, Impression, Recommendations, Explainability, Disclaimer.

Nguyên tắc report:

- Không hallucinate bệnh ngoài prediction.
- Chỉ viết finding khi probability vượt threshold.
- Nếu không có finding vượt threshold, report nói rõ không có high-confidence abnormality.
- Luôn có disclaimer: AI chỉ hỗ trợ nghiên cứu, cần bác sĩ xác nhận.

## 6. Luồng chạy thực tế trong app hiện tại

Khi mở app:

```powershell
streamlit run src/app.py
```

Hoặc:

```powershell
.venv\Scripts\python.exe -m streamlit run src/app.py --server.port 8501
```

Luồng thực tế:

```text
User upload/select image
-> src/app.py nhận PIL image
-> load_torchxrayvision_model
-> analyze_pretrained_case
-> predict_pretrained_with_heatmap
-> calibrate probability
-> estimate uncertainty/risk/severity
-> propose lesion boxes from heatmap
-> generate clinical report
-> render dashboard UI
```

Output UI hiện tại gồm:

- Original Chest X-ray.
- Image view selector: Original, Heatmap, Detection, Lateral.
- AI heatmap/fallback CAM.
- Detection result/proposal overlay.
- Report Summary dưới vùng ảnh.
- Primary Finding.
- Confidence.
- Risk Level.
- Top Differential.
- AI Findings.
- Explainability block.
- AI Interpretation.
- Conclusion.
- Structured report, lesion coordinates và model-card metrics trong expander.

## 7. Luồng target research sau khi có model thật

Khi đã có checkpoint ConvNeXtV2 và RAD-DINO:

```text
Input X-ray
-> ConvNeXtV2 inference -> convnext_probs
-> RAD-DINO inference -> raddino_probs
-> Weighted ensemble -> ensemble_probs
-> Agreement score
-> ConvNeXtV2 EigenCAM
-> RAD-DINO Attention Rollout
-> Disease-to-anatomy clinical evidence
-> Structured report
```

Điểm cần wire thêm:

- Loader cho ConvNeXtV2 checkpoint.
- Loader cho RAD-DINO checkpoint.
- Preprocess đúng theo notebook/train pipeline.
- Inference trả về probability vector 15 class VinBigData.
- Runtime ensemble trong `src/app.py`.
- Agreement score đưa vào report.
- EigenCAM cho ConvNeXtV2.
- Attention Rollout cho RAD-DINO.
- Ảnh lateral thật nếu muốn hỗ trợ multi-view.
- Export/save report JSON/PDF nếu muốn dùng các nút trên UI.

## 8. Cấu hình quan trọng

### 8.1. Default config

```text
configs/default.yaml
```

Dùng cho training/inference local thông thường.

### 8.2. VinBigData target config

```text
configs/vinbig15.yaml
```

Chứa:

- 15 class labels.
- Ensemble default weights.
- Agreement thresholds.
- Explainability method names.
- Research disclaimer.

### 8.3. Model card

```text
configs/model_card.yaml
```

Mô tả:

- Tên model.
- Version.
- Target architecture.
- Dataset/task.
- Explainability target.
- Performance metadata.
- Safety warning.

## 9. Kỹ thuật code chính

Các kỹ thuật đang dùng trong project:

- PyTorch cho deep learning inference/training.
- TorchXRayVision cho pretrained fallback model.
- Streamlit cho web app.
- OpenCV cho resize, heatmap overlay, drawing.
- NumPy/Pandas cho xử lý prediction table.
- Scikit-learn `roc_auc_score` cho optimize ensemble AUROC.
- Temperature scaling để calibration.
- Binary entropy để uncertainty estimation.
- Connected components/heatmap proposal để tạo bounding box.
- Dataclass để gom output pipeline.
- YAML config để tách tham số khỏi code.

## 10. Cách chạy project

### 10.1. Cài môi trường

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 10.2. Chạy app

```powershell
streamlit run src/app.py
```

Nếu dùng virtual environment trực tiếp:

```powershell
.venv\Scripts\python.exe -m streamlit run src/app.py --server.port 8501
```

Mở:

```text
http://localhost:8501
```

### 10.3. Test pretrained demo

```powershell
.venv\Scripts\python.exe scripts\test_pretrained.py --image data\raw\pneumonia_xray_wikimedia.jpg --output outputs\pretrained_pneumonia_heatmap.png
```

### 10.4. Optimize ensemble weight

```powershell
$env:PYTHONPATH='src'
python scripts/optimize_ensemble_weight.py --input outputs/validation_predictions.csv --output outputs/ensemble_weight.yaml
```

## 11. Các giới hạn hiện tại

Project hiện vẫn là research prototype, chưa phải clinical product.

Các giới hạn chính:

- App hiện đang chạy fallback pretrained TorchXRayVision, chưa wire ConvNeXtV2/RAD-DINO checkpoint thật.
- Explainability trong demo là fallback CAM, chưa phải EigenCAM/Attention Rollout thật.
- Lesion boxes là proposal từ heatmap, không phải detector được train riêng.
- Calibration hiện dùng temperature scaling đơn giản.
- Ensemble logic đã có nhưng chưa chạy trong UI inference path.
- `Lateral` trong Image view là placeholder vì chưa có ảnh lateral thật.
- Report là rule-based clinical support, không thay thế radiologist report.
- Export PDF và Save Case đang là placeholder UI.
- Không được dùng kết quả như chẩn đoán y tế chính thức.

## 12. Hướng phát triển tiếp theo

Các bước nên làm tiếp:

1. Export checkpoint và preprocessing từ notebook ConvNeXtV2.
2. Implement `convnext_infer.py` để trả về 15-class probability.
3. Implement RAD-DINO inference wrapper.
4. Wire ensemble vào app local checkpoint mode.
5. Implement EigenCAM cho ConvNeXtV2.
6. Implement Attention Rollout cho RAD-DINO.
7. Đưa agreement score vào report runtime.
8. So sánh AUROC từng model và ensemble trên validation set.
9. Thêm ảnh lateral/multi-view nếu dataset hoặc input hỗ trợ.
10. Lưu report JSON/PDF cho từng case.
11. Thêm unit tests cho clinical evidence mapping và report generation.
12. Validate UI bằng screenshot nhiều viewport.

## 13. Kết luận ngắn

Project này là một Clinical Chest X-ray AI Assistant có cấu trúc theo hướng bác sĩ sử dụng:

```text
Ảnh X-ray
-> dự đoán bệnh
-> giải thích vùng ảnh
-> chuyển thành bằng chứng lâm sàng
-> reasoning
-> báo cáo có cấu trúc
```

Giao diện hiện đã đồng bộ tốt với sườn demo và đã được chỉnh để ưu tiên ảnh X-ray: viewer lớn, không crop phim, có selector đổi Original/Heatmap/Detection/Lateral, report summary nằm dưới vùng ảnh. Code hiện tại đủ cho demo fallback, report, evidence mapping, uncertainty, risk và ensemble utility.

Phần còn thiếu để thành research system hoàn chỉnh là nối model ConvNeXtV2/RAD-DINO thật từ notebook/checkpoint vào pipeline app, chạy ensemble trong UI, thay fallback CAM bằng EigenCAM/Attention Rollout, bổ sung ảnh lateral thật nếu cần, và triển khai các chức năng lưu/xuất report.

Disclaimer: This project is for research and educational support only. It does not replace professional medical diagnosis. Radiologist review is required.
