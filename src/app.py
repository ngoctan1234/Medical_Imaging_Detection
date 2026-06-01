from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image

from xray_abnormal.analysis import analyze_pretrained_case
from xray_abnormal.infer import load_model_for_inference
from xray_abnormal.pretrained import load_torchxrayvision_model
from xray_abnormal.utils import get_device


st.set_page_config(page_title="Radiology AI Workstation", layout="wide")

DEMO_MANIFEST = Path("data/demo_cases/manifest.csv")
MODEL_CARD = Path("configs/model_card.yaml")
FALLBACK_DEMO_CASES = pd.DataFrame(
    [
        {
            "case_id": "CASE-00",
            "title": "Pneumonia sample - Wikimedia",
            "path": "data/raw/pneumonia_xray_wikimedia.jpg",
            "source": "Wikimedia Commons",
            "note": "original sample",
        }
    ]
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fb; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1380px; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    .hero {
        border: 1px solid #d9e2ec; border-radius: 8px; padding: 1rem 1.15rem;
        background: linear-gradient(90deg, #ffffff 0%, #f4f8fb 100%);
        margin-bottom: 0.8rem;
    }
    .main-title { font-size: 1.72rem; font-weight: 780; letter-spacing: 0; margin-bottom: 0.2rem; color: #102a43; }
    .subtitle { color: #52606d; font-size: 0.96rem; margin-bottom: 0; }
    .safety {
        border: 1px solid #f2c98a; background: #fff9ef; color: #573b00;
        padding: 0.6rem 0.8rem; border-radius: 6px; margin: 0.6rem 0 0.8rem 0;
        font-size: 0.9rem;
    }
    .section-label {
        color: #334e68; font-size: 0.78rem; text-transform: uppercase;
        font-weight: 760; letter-spacing: 0.045rem; margin: 0.55rem 0 0.35rem 0;
    }
    .summary-strip {
        border: 1px solid #d9e2ec; border-radius: 8px; background: #ffffff;
        padding: 0.85rem 1rem; margin: 0.7rem 0 0.85rem 0;
    }
    .stat-card {
        border: 1px solid #d9e2ec; border-radius: 8px; background: #ffffff;
        padding: 0.85rem 0.95rem; min-height: 5.2rem;
    }
    .stat-label {
        color: #627d98; font-size: 0.76rem; font-weight: 760;
        text-transform: uppercase; letter-spacing: 0.04rem; margin-bottom: 0.32rem;
    }
    .stat-value {
        color: #102a43; font-size: 1.18rem; font-weight: 780; line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .stat-note { color: #829ab1; font-size: 0.82rem; margin-top: 0.22rem; }
    .report-box {
        border: 1px solid #d9e2ec; border-radius: 8px; background: #ffffff;
        padding: 0.9rem; min-height: 11rem;
    }
    .report-title { font-size: 0.92rem; font-weight: 760; color: #102a43; margin-bottom: 0.45rem; }
    .report-text { font-size: 0.94rem; color: #334e68; line-height: 1.48; }
    .study-line {
        color: #52606d; font-size: 0.9rem; margin-top: 0.4rem; margin-bottom: 0.4rem;
    }
    .badge {
        display: inline-block; border-radius: 999px; padding: 0.25rem 0.65rem;
        font-weight: 700; font-size: 0.82rem; margin-right: 0.35rem;
    }
    .badge-low { background: #e8f7ef; color: #10633d; }
    .badge-medium { background: #fff3d6; color: #8a5600; }
    .badge-high { background: #fde9e7; color: #b42318; }
    div[data-testid="stMetric"] {
        border: 1px solid #d9e2ec; border-radius: 8px; padding: 0.72rem 0.82rem; background: #ffffff;
    }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #102a43; }
    div[data-testid="stMetricLabel"] { color: #627d98; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid #d9e2ec; }
    .stTabs [data-baseweb="tab"] {
        height: 2.6rem; border-radius: 6px 6px 0 0; color: #334e68;
        padding: 0.5rem 0.9rem;
    }
    .stTabs [aria-selected="true"] { background: #ffffff; color: #102a43; border: 1px solid #d9e2ec; border-bottom: 1px solid #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_cases() -> pd.DataFrame:
    if DEMO_MANIFEST.exists():
        df = pd.read_csv(DEMO_MANIFEST)
    else:
        df = FALLBACK_DEMO_CASES.copy()
    return df[df["path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_model_card() -> dict:
    if not MODEL_CARD.exists():
        return {}
    with MODEL_CARD.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def get_pretrained_model(weights: str, device_name: str) -> torch.nn.Module:
    return load_torchxrayvision_model(weights, torch.device(device_name))


@st.cache_resource(show_spinner=False)
def get_checkpoint_model(checkpoint: str, config: str, device_name: str) -> tuple[torch.nn.Module, dict, list[str]]:
    return load_model_for_inference(checkpoint, config, torch.device(device_name))


def risk_badge(level: str) -> str:
    css = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "badge-medium")
    return f"<span class='badge {css}'>{level} Risk</span>"


def stat_card(label: str, value: str, note: str = "") -> str:
    note_html = f"<div class='stat-note'>{escape(note)}</div>" if note else ""
    return (
        "<div class='stat-card'>"
        f"<div class='stat-label'>{escape(label)}</div>"
        f"<div class='stat-value'>{escape(value)}</div>"
        f"{note_html}"
        "</div>"
    )


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["label", "calibrated_probability", "risk_level", "severity", "uncertainty"]
    out = df[cols].copy()
    out.columns = ["Finding", "Calibrated Confidence", "Risk", "Severity", "Uncertainty"]
    out["Calibrated Confidence"] = out["Calibrated Confidence"].map(lambda x: f"{x:.3f}")
    return out


model_card = load_model_card()
model_info = model_card.get("model", {})
performance = model_card.get("performance", {})
safety = model_card.get("safety", {})

st.markdown(
    """
    <div class='hero'>
      <div class='main-title'>Chest X-ray AI Workstation</div>
      <div class='subtitle'>Triage, multi-label findings, explainability, lesion proposal, calibrated confidence, and structured reporting.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='safety'>{safety.get('warning', 'Research use only. This system does not replace professional medical diagnosis.')}</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Study Setup")
    demo_cases = load_demo_cases()
    input_source = st.segmented_control("Image source", ["Demo cases", "Upload image"], default="Demo cases")
    if input_source == "Demo cases":
        if demo_cases.empty:
            st.error("No demo images found in data/demo_cases or data/raw.")
            st.stop()
        case_options = [f"{row.case_id} | {row.title}" for row in demo_cases.itertuples()]
        selected_label = st.selectbox("Case", case_options)
        selected_row = demo_cases.iloc[case_options.index(selected_label)]
        image_path = Path(str(selected_row["path"]))
        image = Image.open(image_path)
        case_note = f"{selected_row['source']} | {selected_row['note']}"
    else:
        uploaded = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is None:
            st.info("Upload a 2D chest X-ray image to run prediction and Grad-CAM.")
            st.stop()
        image = Image.open(uploaded)
        image_path = Path(uploaded.name)
        case_note = "User uploaded image"

    st.divider()
    patient_id = st.text_input("Patient ID", "DEMO-001")
    age = st.number_input("Age", min_value=1, max_value=110, value=54)
    sex = st.selectbox("Sex", ["Unknown", "Female", "Male"])
    view_position = st.selectbox("View", ["PA", "AP", "Lateral", "Unknown"])

    st.divider()
    st.header("AI Protocol")
    model_source = st.radio("Model source", ["TorchXRayVision pretrained", "Local checkpoint"], label_visibility="collapsed")
    xrv_weights = st.selectbox("Pretrained weights", ["densenet121-res224-all"])
    threshold = st.slider("Abnormal threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    temperature = st.slider("Calibration temperature", min_value=0.7, max_value=3.0, value=1.4, step=0.1)
    top_k = st.slider("Top findings", min_value=3, max_value=10, value=6, step=1)
    checkpoint_path = st.text_input("Checkpoint", "outputs/checkpoints/best.pt")
    config_path = st.text_input("Config fallback", "configs/default.yaml")

device = get_device("auto")
with st.spinner("Running classification, calibration, Grad-CAM, and lesion proposal..."):
    try:
        if model_source == "TorchXRayVision pretrained":
            model = get_pretrained_model(xrv_weights, str(device))
            analysis = analyze_pretrained_case(model, image, device, threshold, temperature=temperature)
        else:
            if not Path(checkpoint_path).exists():
                st.error(f"Checkpoint not found: {checkpoint_path}")
                st.stop()
            get_checkpoint_model(checkpoint_path, config_path, str(device))
            st.error("Advanced workstation view currently requires the TorchXRayVision pretrained model.")
            st.stop()
    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        st.stop()

if analysis.boxes:
    box_df = pd.DataFrame(
        [
            {
                "Finding": box.label,
                "Confidence": f"{box.confidence:.3f}",
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
                "Affected Lung Region": box.lung_region,
                "Area Ratio": f"{box.area_ratio:.3f}",
            }
            for box in analysis.boxes
        ]
    )
else:
    box_df = pd.DataFrame()

st.markdown(
    (
        "<div class='study-line'>"
        f"Study {escape(patient_id)} &middot; {age} years &middot; {escape(sex)} "
        f"&middot; {escape(view_position)} view &middot; {escape(case_note)}"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown("<div class='section-label'>AI Case Summary</div>", unsafe_allow_html=True)
metric_a, metric_b, metric_c, metric_d = st.columns([1.4, 1, 1, 1])
metric_a.markdown(stat_card("Primary diagnosis", analysis.primary_label, "Highest calibrated finding"), unsafe_allow_html=True)
metric_b.markdown(stat_card("Calibrated score", f"{analysis.primary_confidence:.3f}", "Temperature adjusted"), unsafe_allow_html=True)
metric_c.markdown(stat_card("Uncertainty", analysis.uncertainty, "Entropy estimate"), unsafe_allow_html=True)
metric_d.markdown(
    "<div class='stat-card'><div class='stat-label'>Risk level</div>"
    f"<div class='stat-value'>{risk_badge(analysis.risk)}</div>"
    "<div class='stat-note'>Triage indicator</div></div>",
    unsafe_allow_html=True,
)

overview_tab, imaging_tab, report_tab, model_tab = st.tabs(
    ["Overview", "Imaging", "Radiology Report", "Model & Metrics"]
)

with overview_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("<div class='section-label'>Original X-ray</div>", unsafe_allow_html=True)
        st.image(image, width="stretch")
    with right:
        st.markdown("<div class='section-label'>AI Detection Result</div>", unsafe_allow_html=True)
        st.image(analysis.detection_overlay, width="stretch")

    finding_col, chart_col = st.columns([1, 1])
    with finding_col:
        st.markdown("<div class='section-label'>Top Findings</div>", unsafe_allow_html=True)
        st.dataframe(format_prediction_table(analysis.predictions.head(top_k)), hide_index=True, width="stretch")
    with chart_col:
        st.markdown("<div class='section-label'>Calibrated Probability</div>", unsafe_allow_html=True)
        chart_df = analysis.predictions.head(top_k).set_index("label")[["calibrated_probability"]]
        st.bar_chart(chart_df, height=300)

    st.markdown("<div class='section-label'>Decision Rationale</div>", unsafe_allow_html=True)
    st.info(analysis.explanation)

with imaging_tab:
    img_a, img_b = st.columns(2)
    with img_a:
        st.markdown("<div class='section-label'>Grad-CAM</div>", unsafe_allow_html=True)
        st.image(analysis.gradcam_overlay, width="stretch")
    with img_b:
        st.markdown("<div class='section-label'>Bounding Box Overlay</div>", unsafe_allow_html=True)
        st.image(analysis.detection_overlay, width="stretch")

    st.markdown("<div class='section-label'>Lesion Coordinates</div>", unsafe_allow_html=True)
    if box_df.empty:
        st.warning("No compact lesion proposal was generated above the localization threshold.")
    else:
        st.dataframe(box_df, hide_index=True, width="stretch")

with report_tab:
    report_a, report_b, report_c = st.columns(3)
    with report_a:
        st.markdown(
            f"<div class='report-box'><div class='report-title'>Findings</div><div class='report-text'>{analysis.report['findings']}</div></div>",
            unsafe_allow_html=True,
        )
    with report_b:
        st.markdown(
            f"<div class='report-box'><div class='report-title'>Impression</div><div class='report-text'>{analysis.report['impression']}</div></div>",
            unsafe_allow_html=True,
        )
    with report_c:
        st.markdown(
            f"<div class='report-box'><div class='report-title'>Recommendations</div><div class='report-text'>{analysis.report['recommendations']}</div></div>",
            unsafe_allow_html=True,
        )

with model_tab:
    model_a, model_b = st.columns([1.15, 1])
    with model_a:
        st.markdown("<div class='section-label'>Model Card</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Model name", "Value": model_info.get("name", "N/A")},
                    {"Field": "Architecture", "Value": model_info.get("architecture", "N/A")},
                    {"Field": "Dataset used", "Value": model_info.get("dataset", "N/A")},
                    {"Field": "Input resolution", "Value": model_info.get("input_resolution", "N/A")},
                    {"Field": "Training date", "Value": model_info.get("training_date", "N/A")},
                    {"Field": "Version", "Value": model_info.get("version", "N/A")},
                    {"Field": "Detector backend", "Value": model_info.get("detector_backend", "N/A")},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with model_b:
        st.markdown("<div class='section-label'>Performance Dashboard</div>", unsafe_allow_html=True)
        perf_cols = st.columns(3)
        metric_names = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-score", "f1_score"),
            ("mAP", "map"),
            ("ROC-AUC", "roc_auc"),
        ]
        for idx, (label, key) in enumerate(metric_names):
            perf_cols[idx % 3].metric(label, f"{float(performance.get(key, 0.0)):.3f}")

    with st.expander("System Architecture and Roadmap", expanded=False):
        st.markdown(
            """
            **Architecture:** Streamlit UI -> analysis service -> classification model -> Grad-CAM -> localization proposal -> calibration/uncertainty -> report generator.

            **Detection upgrade path:** replace Grad-CAM proposal with YOLOv8 or DETR lesion detector trained on VinDr-CXR, RSNA Pneumonia, NIH bbox subset, or local annotations.

            **Research roadmap:** external validation, temperature calibration on validation split, subgroup analysis, reader study, DICOM support, PACS integration, structured audit logs.
            """
        )

st.caption("Research prototype only. Not for clinical diagnosis.")
