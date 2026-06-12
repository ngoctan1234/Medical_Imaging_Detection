from __future__ import annotations

import base64
from datetime import date
from io import BytesIO
from html import escape
from pathlib import Path

import torch
import pandas as pd
import streamlit as st
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
    html, body, .stApp, div[data-testid="stAppViewContainer"] {
        max-height: 120vh;
        overflow-y: auto;
    }
    .stApp { background: #eef5ff; color: #17233c; }
    .block-container { padding: 0 !important; max-width: none; }
    main .block-container, div[data-testid="stMainBlockContainer"] { padding: 0 !important; }
    header[data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
    .stDeployButton { display: none; }
    section[data-testid="stSidebar"] {
        background: #f8fbff; border-right: 1px solid #cfdaf0; box-shadow: 10px 0 28px rgba(17, 38, 74, 0.04);
        width: 205px !important;
        min-width: 205px !important;
        max-width: 205px !important;
        top: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container,
    div[data-testid="stSidebarContent"] {
        padding: 0.55rem 0.75rem 0.8rem 0.75rem !important;
        margin-top: -4.2rem !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.68rem;
        margin: -0.55rem -0.75rem 1.05rem -0.75rem;
        padding: 0.72rem 0.78rem;
        background: linear-gradient(180deg, #071d4d 0%, #061334 100%);
        color: white;
        min-height: 3.95rem;
    }
    .brand-mark {
        width: 2.75rem; height: 2.75rem; border-radius: 8px; display: grid; place-items: center;
        background: #061d52; color: #23d7ef; font-size: 1.25rem; font-weight: 900;
    }
    .lung-svg { width: 2rem; height: 2rem; display: block; }
    .lung-svg path, .lung-svg line { stroke: currentColor; }
    .brand-title { color: white; font-size: 1.16rem; font-weight: 900; line-height: 1.1; }
    .brand-subtitle { color: #0aa6c7; font-size: 0.7rem; font-weight: 750; margin-top: 0.15rem; }
    .side-nav { display: grid; gap: 0.6rem; margin-bottom: 1rem; }
    .side-nav-item {
        color: #123163; font-size: 0.82rem; font-weight: 860; display: flex; gap: 0.6rem; align-items: center;
        border: 1px solid #d7e4f4; border-radius: 6px; padding: 0.54rem 0.62rem; background: #ffffff;
    }
    .side-nav-item.primary { background: #1164ef; border-color: #1164ef; color: white; }
    .side-icon { width: 1.2rem; text-align: center; color: #60728c; }
    .case-select-spacer { margin-top: 0.8rem; }
    .side-detail { color: #2a3a54; font-size: 0.9rem; font-weight: 730; margin: 0.1rem 0 1.05rem 0; }
    .side-muted { color: #75869d; font-size: 0.78rem; font-weight: 720; margin-bottom: 0.3rem; }
    .sidebar-footer {
        border-top: 1px solid #dfe7f1; color: #8a9ab0; font-size: 0.76rem; padding-top: 0.75rem; margin-top: 1.45rem;
    }
    .xnet-dashboard {
        height: 100vh;
        max-height: 120vh;
        overflow: hidden;
        background: #eef5ff;
        display: grid;
        grid-template-rows: 3.55rem 3.0rem minmax(0, 1fr) 1.6rem;
        gap: 0.42rem;
        padding: 0;
        margin-top: -1rem;
    }
    .xnet-header {
        background: linear-gradient(90deg, #061c4d 0%, #071a43 58%, #071230 100%);
        color: white;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        padding: 0.58rem 0.92rem;
        box-shadow: 0 8px 18px rgba(4, 21, 62, 0.18);
    }
    .xnet-brand { display: flex; align-items: center; gap: 0.75rem; min-width: 0; }
    .xnet-logo { color: #1dd5f4; font-size: 1.7rem; line-height: 1; }
    .xnet-title { font-size: 1.1rem; font-weight: 900; letter-spacing: 0; }
    .xnet-subtitle { color: #20d7f2; font-size: 0.76rem; font-weight: 740; margin-top: 0.1rem; }
    .xnet-actions { display: flex; gap: 0.7rem; align-items: center; }
    .xnet-action {
        border: 1px solid rgba(255,255,255,0.62);
        border-radius: 6px;
        padding: 0.48rem 0.78rem;
        font-size: 0.72rem;
        font-weight: 850;
        color: white;
        background: rgba(255,255,255,0.03);
        white-space: nowrap;
    }
    .xnet-action .mini-icon {
        display: inline-grid;
        place-items: center;
        width: 1.05rem;
        height: 1.05rem;
        margin-right: 0.34rem;
        border: 1px solid currentColor;
        border-radius: 3px;
        font-size: 0.48rem;
        line-height: 1;
        vertical-align: -0.1rem;
    }
    .xnet-study-strip {
        margin: 0 0.65rem;
        border: 1px solid #d6e2f4;
        border-radius: 8px;
        background: white;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        align-items: center;
        box-shadow: 0 8px 20px rgba(17, 38, 74, 0.04);
        overflow: hidden;
    }
    .xnet-study-item {
        display: grid;
        grid-template-columns: 2.15rem minmax(0, 1fr);
        gap: 0.6rem;
        align-items: center;
        min-height: 2.72rem;
        padding: 0.38rem 0.7rem;
        border-right: 1px solid #dce6f5;
    }
    .xnet-study-item:last-child { border-right: 0; }
    .xnet-study-icon {
        width: 1.9rem;
        height: 1.9rem;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: #edf5ff;
        color: #0a54d6;
        font-weight: 900;
        font-size: 0.92rem;
    }
    .xnet-kicker { color: #102e70; font-size: 0.62rem; font-weight: 900; text-transform: uppercase; line-height: 1.1; }
    .xnet-field { color: #07152d; font-size: 0.78rem; font-weight: 850; margin-top: 0.12rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .xnet-content {
        min-height: 0;
        margin: 0 0.65rem;
        display: grid;
        grid-template-columns: minmax(0, 1.02fr) minmax(22rem, 0.98fr);
        grid-template-rows: 16.5rem 4.8rem minmax(0, 1fr);
        gap: 0.5rem;
    }
    .xnet-panel {
        border: 1px solid #d6e2f4;
        border-radius: 8px;
        background: #fff;
        box-shadow: 0 8px 20px rgba(17, 38, 74, 0.035);
        overflow: hidden;
    }
    .xnet-panel-title {
        height: 2rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0 0.72rem;
        color: #09255c;
        font-size: 0.78rem;
        font-weight: 900;
        border-bottom: 1px solid #e3ecfa;
        background: #fbfdff;
    }
    .xnet-image-panel { grid-row: 1 / 2; grid-column: 1 / 2; }
    .xnet-image-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        padding: 0.62rem;
        height: calc(100% - 2rem);
    }
    .xnet-image-frame {
        min-width: 0;
        background: #020814;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #0b1e4b;
        display: grid;
        grid-template-rows: 1.4rem minmax(0, 1fr);
    }
    .xnet-image-label {
        background: #061d52;
        color: white;
        font-size: 0.68rem;
        font-weight: 850;
        display: grid;
        place-items: center;
    }
    .xnet-image {
        width: 100%;
        height: 100%;
        min-height: 0;
        object-fit: contain;
        display: block;
        background: #020814;
    }
    .xnet-report-panel { grid-row: 1 / 2; grid-column: 2 / 3; }
    .xnet-report-grid {
        height: calc(100% - 2rem);
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
        padding: 0.72rem;
    }
    .xnet-report-card {
        border: 1px solid #dde7f7;
        border-radius: 8px;
        background: #fff;
        display: grid;
        grid-template-rows: 2.5rem auto auto;
        justify-items: center;
        align-content: center;
        text-align: center;
        padding: 0.5rem;
        min-width: 0;
    }
    .xnet-report-icon {
        width: 2.28rem;
        height: 2.28rem;
        border-radius: 999px;
        display: grid;
        place-items: center;
        color: white;
        background: #1b62db;
        font-size: 1rem;
        font-weight: 900;
    }
    .xnet-report-icon.teal { background: #0aa6a7; }
    .xnet-report-icon.orange { background: #ff7800; }
    .xnet-report-icon.blue { background: #2167da; }
    .xnet-card-label { color: #0b2b69; font-size: 0.68rem; font-weight: 900; margin-top: 0.3rem; }
    .xnet-card-value { color: #0a1d47; font-size: 1.02rem; font-weight: 950; margin-top: 0.45rem; overflow-wrap: anywhere; }
    .xnet-card-value.prob { color: #05a6b8; font-size: 1.58rem; }
    .xnet-card-value.risk { color: #ff7800; }
    .xnet-card-note { color: #8b9bb5; font-size: 0.62rem; font-weight: 760; margin-top: 0.2rem; }
    .xnet-diff-list { text-align: left; color: #233f72; font-size: 0.72rem; font-weight: 760; line-height: 1.65; margin-top: 0.35rem; }
    .xnet-findings-panel { grid-column: 2 / 3; grid-row: 2 / 3; padding-bottom: 0.55rem; }
    .xnet-chip-row { display: flex; flex-wrap: wrap; gap: 0.55rem; padding: 0.55rem 0.72rem 0; }
    .xnet-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 999px;
        background: #eef4ff;
        color: #153d88;
        padding: 0.35rem 0.66rem;
        font-size: 0.68rem;
        font-weight: 900;
        min-width: 6.6rem;
        justify-content: space-between;
    }
    .xnet-chip.primary { color: #d9233f; background: #fff0f3; }
    .xnet-chip-score { background: #dfe9fb; color: #173c88; border-radius: 999px; padding: 0.12rem 0.45rem; font-size: 0.63rem; }
    .xnet-lower-left { grid-column: 1 / 2; grid-row: 2 / 4; }
    .xnet-evidence {
        height: calc(100% - 2rem);
        display: grid;
        grid-template-columns: minmax(0, 1fr) 2rem;
        gap: 0.5rem;
        padding: 0.75rem;
    }
    .xnet-check-row { display: grid; grid-template-columns: 1.3rem minmax(0, 1fr); gap: 0.55rem; align-items: start; margin-bottom: 0.75rem; }
    .xnet-check { width: 1rem; height: 1rem; border-radius: 999px; display: grid; place-items: center; background: #10a8a8; color: white; font-size: 0.62rem; font-weight: 900; margin-top: 0.1rem; }
    .xnet-evidence-text { color: #24406f; font-size: 0.72rem; line-height: 1.45; font-weight: 700; }
    .xnet-mini-legend { display: grid; justify-items: center; align-content: center; color: #1e3562; font-size: 0.62rem; font-weight: 900; }
    .xnet-mini-bar { width: 0.72rem; height: 5.9rem; border-radius: 999px; margin: 0.25rem 0; background: linear-gradient(180deg, #ef1d25, #ffbf25, #68e04f, #1cc7d7, #1164f1); }
    .xnet-conclusion-panel { grid-column: 2 / 3; grid-row: 3 / 4; display: grid; grid-template-columns: 1fr 1.05fr; gap: 0.62rem; background: transparent; border: 0; box-shadow: none; }
    .xnet-conclusion-card, .xnet-notes-card { border: 1px solid #d6e2f4; border-radius: 8px; background: white; overflow: hidden; }
    .xnet-body { padding: 0.72rem; color: #24406f; font-size: 0.72rem; line-height: 1.5; font-weight: 700; }
    .xnet-notes-box {
        border: 1px solid #dce6f5;
        border-radius: 6px;
        background: #fbfdff;
        min-height: 4.2rem;
        padding: 0.55rem;
        color: #52698f;
        font-size: 0.68rem;
        line-height: 1.45;
    }
    .xnet-note-actions { display: flex; justify-content: space-between; margin-top: 0.55rem; }
    .xnet-note-btn { border: 1px solid #b9cff3; border-radius: 5px; color: #1a5fd0; font-size: 0.68rem; font-weight: 850; padding: 0.32rem 0.8rem; background: white; }
    .xnet-note-btn.primary { background: #1164ef; color: white; border-color: #1164ef; }
    .xnet-footer {
        margin: 0 0.65rem 0.35rem;
        border: 1px solid #d6e2f4;
        border-radius: 7px;
        background: #f9fcff;
        color: #1260d4;
        display: grid;
        place-items: center;
        font-size: 0.7rem;
        font-weight: 760;
    }
    .top-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 0.58rem 0.85rem;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.045); margin-bottom: 0.55rem;
        height: 6.1rem;
        overflow: hidden;
    }
    .dashboard-shell {
        width: 100%;
        display: grid;
        gap: 0.55rem;
        max-height: 120vh;
        overflow: hidden;
    }
    .workflow-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1rem 1.15rem;
        box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045);
    }
    .workflow-row {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 0.75rem;
    }
    .workflow-step {
        border: 1px solid #dce8f7; border-radius: 8px; background: #f7fbff;
        padding: 0.72rem 0.78rem; min-height: 4.6rem;
    }
    .workflow-step.active { border-color: #9fc4ff; background: #edf5ff; }
    .workflow-index { color: #1769e0; font-size: 0.72rem; font-weight: 900; text-transform: uppercase; }
    .workflow-title { color: #07152d; font-size: 0.86rem; font-weight: 880; line-height: 1.25; margin-top: 0.25rem; }
    .workflow-sub { color: #667a97; font-size: 0.74rem; font-weight: 680; line-height: 1.35; margin-top: 0.22rem; }
    .mode-banner {
        border: 1px solid #cfe0f8; border-radius: 8px; background: #f4f8ff;
        padding: 0.78rem 0.9rem; color: #314868; font-size: 0.88rem; font-weight: 680; line-height: 1.45;
        margin-top: 0.85rem;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: 1.16fr 1fr 1fr 1fr;
        gap: 1rem;
    }
    .model-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }
    .model-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1.05rem 1.15rem;
        box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045); min-height: 7.4rem;
    }
    .model-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
    .model-name { color: #07152d; font-size: 1rem; font-weight: 900; line-height: 1.2; }
    .model-role { color: #607289; font-size: 0.8rem; font-weight: 720; line-height: 1.35; margin-top: 0.35rem; }
    .status-pill {
        display: inline-flex; align-items: center; justify-content: center; white-space: nowrap;
        border-radius: 999px; padding: 0.34rem 0.62rem; font-size: 0.68rem; font-weight: 900; text-transform: uppercase;
        background: #e9f2ff; color: #1769e0;
    }
    .status-pill.pending { background: #fff5e8; color: #e97818; }
    .status-pill.ready { background: #eaf8ef; color: #258348; }
    .model-metric { color: #17233c; font-size: 0.86rem; font-weight: 780; margin-top: 0.75rem; line-height: 1.45; }
    .image-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 3.2rem minmax(0, 1fr);
        gap: 1rem;
        align-items: stretch;
    }
    .identity-grid {
        display: grid; grid-template-columns: 1fr 1.4fr auto; gap: 1rem; align-items: center;
    }
    .identity-block { display: flex; gap: 0.8rem; align-items: center; min-width: 0; }
    .identity-icon {
        width: 3.05rem; height: 3.05rem; border-radius: 999px; background: #e9f2ff; color: #1567f9;
        display: grid; place-items: center; font-size: 1.1rem; font-weight: 900; flex: 0 0 auto;
    }
    .eyebrow { color: #1970ef; font-size: 0.68rem; font-weight: 820; text-transform: uppercase; letter-spacing: 0.03rem; }
    .identity-title { color: #07152d; font-size: 1.02rem; font-weight: 900; line-height: 1.08; margin-top: 0.18rem; letter-spacing: 0; }
    .meta-line { color: #667a97; font-size: 0.78rem; margin-top: 0.28rem; overflow: hidden; text-overflow: ellipsis; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .top-actions { display: flex; gap: 0.55rem; align-items: center; justify-content: end; }
    .action-btn {
        border: 1px solid #d4e0ef; background: #ffffff; color: #253955; border-radius: 7px;
        padding: 0.58rem 0.75rem; font-size: 0.78rem; font-weight: 780; white-space: nowrap;
    }
    .action-btn.primary { background: #145eea; color: #ffffff; border-color: #145eea; }
    .kebab { color: #0f1d35; font-size: 1.4rem; font-weight: 900; padding-left: 0.45rem; }
    .section-label {
        color: #35506f; font-size: 0.72rem; text-transform: uppercase;
        font-weight: 820; letter-spacing: 0.045rem; margin: 0.25rem 0 0.42rem 0;
    }
    .stat-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1.25rem 1.25rem;
        min-height: 8.2rem; box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045);
        display: grid; grid-template-columns: 3.35rem minmax(0, 1fr); gap: 1rem; align-items: center;
        position: relative;
    }
    .stat-card * { word-break: normal; overflow-wrap: normal; }
    .stat-label { color: #607289; font-size: 0.83rem; font-weight: 790; margin-bottom: 0.45rem; white-space: nowrap; }
    .stat-value { color: #07152d; font-size: 1.28rem; font-weight: 900; line-height: 1.2; white-space: nowrap; }
    .stat-note { color: #72849a; font-size: 0.82rem; margin-top: 0.65rem; line-height: 1.35; max-width: 12rem; }
    .stat-icon {
        width: 3.35rem; height: 3.35rem; border-radius: 999px; background: #edf5ff; color: #1769e0;
        display: grid; place-items: center; font-size: 1.45rem; font-weight: 900;
    }
    .stat-mini {
        width: 2rem; height: 2rem; border-radius: 999px; background: #eaf3ff; color: #1769e0;
        display: grid; place-items: center; font-size: 0.9rem; font-weight: 900;
        position: absolute; right: 1.05rem; bottom: 1.05rem;
    }
    .stat-mini.warn { background: #fff3e7; color: #f08022; }
    .stat-icon.warn { background: #fff3e7; color: #f08022; }
    .image-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1.35rem;
        box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045); min-height: 100%;
    }
    .xray-img {
        display: block; width: 100%; aspect-ratio: 1 / 0.92; object-fit: cover;
        border-radius: 7px; border: 1px solid #e5edf6; background: #05070a;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem; box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] img {
        border-radius: 7px; border: 1px solid #e5edf6;
    }
    .image-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.95rem; }
    .expand-icon { color: #203653; font-weight: 900; font-size: 1rem; }
    .image-card img { border-radius: 7px; border: 1px solid #e5edf6; }
    .heat-legend {
        height: 15.2rem; width: 1rem; border-radius: 999px;
        background: linear-gradient(180deg, #e81d2a 0%, #ffb72c 24%, #6ee35f 48%, #1ec4d7 68%, #1164f1 100%);
        margin: 0.55rem auto;
    }
    .legend-card {
        display: grid;
        align-content: center;
        justify-items: center;
        min-height: 100%;
        padding-top: 3.2rem;
    }
    .legend-text { color: #415671; font-size: 0.82rem; text-align: center; font-weight: 820; white-space: nowrap; }
    .findings-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1.35rem 1.45rem;
        box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045); margin-top: 1.15rem;
    }
    .doctor-note {
        border: 1px solid #d8e8ff; border-radius: 10px; background: #f2f7ff; padding: 1.05rem 1.25rem;
        color: #2d4d78; font-size: 0.95rem; line-height: 1.55; margin-top: 1rem;
        box-shadow: 0 10px 24px rgba(22, 104, 242, 0.06);
    }
    .clinical-grid {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem; margin-top: 0.75rem;
    }
    .clinical-item {
        border: 1px solid #d8e8ff; border-radius: 8px; background: #ffffff; padding: 0.8rem 0.9rem;
    }
    .clinical-label { color: #1668f2; font-size: 0.72rem; font-weight: 860; text-transform: uppercase; letter-spacing: 0.04rem; }
    .clinical-text { color: #273b59; font-size: 0.9rem; font-weight: 680; line-height: 1.45; margin-top: 0.35rem; }
    .trace-card {
        border: 1px solid #dfe7f1; border-radius: 10px; background: #ffffff; padding: 1.25rem 1.35rem;
        box-shadow: 0 13px 34px rgba(16, 42, 67, 0.045); margin-top: 1.15rem;
    }
    .trace-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 0.75rem;
        margin-top: 0.8rem;
    }
    .trace-item {
        border: 1px solid #e0e9f5; border-radius: 8px; background: #fbfdff; padding: 0.82rem 0.9rem;
        min-height: 6.5rem;
    }
    .trace-label { color: #1668f2; font-size: 0.72rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.04rem; }
    .trace-text { color: #273b59; font-size: 0.86rem; font-weight: 680; line-height: 1.45; margin-top: 0.35rem; }
    .findings-row { display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap; margin-top: 0.9rem; }
    .finding-pill {
        display: inline-flex; align-items: center; gap: 0.55rem; border-radius: 999px; padding: 0.75rem 1.15rem;
        background: #f0f4fa; color: #516883; font-weight: 800; font-size: 0.93rem; min-width: 8.8rem; justify-content: center;
    }
    .finding-pill.primary { background: #fdebed; color: #dd2f33; }
    .finding-dot { width: 0.55rem; height: 0.55rem; border-radius: 999px; background: currentColor; }
    .finding-score { color: inherit; opacity: 0.95; margin-left: 0.4rem; }
    .view-all { margin-left: auto; color: #1668f2; font-weight: 850; font-size: 0.9rem; }
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
        border-radius: 6px; font-weight: 760;
    }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed #cbd9ec; border-radius: 8px; background: #f8fbff;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.9rem;
    }
    section[data-testid="stSidebar"] details {
        border: 1px solid #d7e4f4;
        border-radius: 6px;
        background: #fff;
        margin: 0.45rem 0 0.65rem;
    }
    section[data-testid="stSidebar"] details summary {
        min-height: 2.45rem;
        font-size: 0.8rem;
        font-weight: 780;
        color: #123163;
    }
    div[data-testid="stSegmentedControl"] {
        max-width: 31rem;
        margin-bottom: 0.45rem;
    }
    div[data-testid="stSegmentedControl"] label {
        display: none;
    }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .footer-note { color: #6c7d94; font-size: 0.9rem; text-align: center; margin-top: 1.65rem; }
    .cxr-stepper {
        border: 0;
        background: transparent;
        padding: 0;
        box-shadow: none;
        margin-top: 0.35rem;
    }
    .cxr-stepper-row {
        display: flex;
        flex-wrap: nowrap;
        gap: 0.35rem;
        align-items: center;
        overflow: hidden;
    }
    .cxr-rail-label { color: #607289; font-size: 0.62rem; font-weight: 900; text-transform: uppercase; }
    .cxr-step {
        display: inline-flex;
        gap: 0.25rem;
        align-items: center;
        min-height: 1.35rem;
        border: 1px solid #dce8f7;
        border-radius: 999px;
        background: #edf5ff;
        color: #203653;
        padding: 0.16rem 0.48rem;
        font-size: 0.64rem;
        font-weight: 850;
        white-space: nowrap;
    }
    .cxr-step-num { color: #1567f9; font-size: 0.58rem; font-weight: 950; }
    .analysis-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 1rem;
        align-items: stretch;
    }
    .analysis-card, .summary-card, .mini-card, .wide-card {
        border: 1px solid #dfe7f1;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 12px 28px rgba(16, 42, 67, 0.04);
    }
    .analysis-card { padding: 1.05rem 1.15rem 1.2rem; }
    .analysis-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e5edf6;
        padding-bottom: 0.65rem;
        margin-bottom: 1rem;
    }
    .analysis-title, .summary-title { color: #07152d; font-size: 0.86rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.03rem; }
    .analysis-tools {
        display: grid;
        gap: 0.5rem;
        align-content: center;
        justify-items: center;
        padding-top: 2.4rem;
    }
    .tool-btn {
        width: 2rem;
        height: 2rem;
        border-radius: 6px;
        border: 1px solid #d7e4f4;
        color: #4f6683;
        display: grid;
        place-items: center;
        font-weight: 900;
        background: #fbfdff;
    }
    .viewer-grid {
        display: grid;
        grid-template-columns: 2.25rem minmax(0, 1fr) minmax(0, 1fr) 3.2rem;
        gap: 1.25rem;
        align-items: stretch;
    }
    .viewer-block { min-width: 0; }
    .viewer-label {
        color: #203653;
        font-size: 0.82rem;
        font-weight: 900;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 0.7rem;
        min-height: 2.05rem;
        display: grid;
        place-items: center;
    }
    .viewer-img {
        width: 100%;
        height: clamp(21rem, 45vh, 31rem);
        object-fit: contain;
        background: #05070a;
        border: 1px solid #d7e4f4;
        border-radius: 8px;
        box-shadow: 0 14px 26px rgba(9, 20, 38, 0.12);
    }
    .viewer-legend {
        display: grid;
        align-content: center;
        justify-items: center;
        padding-top: 2.4rem;
    }
    .viewer-legend .heat-legend {
        height: clamp(18rem, 40vh, 28rem);
    }
    .summary-card { padding: 1rem 1.05rem; }
    .summary-copy {
        color: #334a68;
        font-size: 0.9rem;
        font-weight: 680;
        line-height: 1.55;
        padding: 0.8rem 0;
        border-bottom: 1px solid #e5edf6;
    }
    .summary-list {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }
    .summary-row {
        display: grid;
        grid-template-columns: 2.1rem minmax(0, 1fr);
        gap: 0.7rem;
        align-items: center;
        border: 1px solid #edf2f8;
        border-radius: 8px;
        padding: 0.72rem 0.8rem;
    }
    .summary-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        background: #edf5ff;
        color: #1567f9;
        display: grid;
        place-items: center;
        font-weight: 900;
    }
    .summary-icon.warn { background: #fff3e7; color: #f08022; }
    .summary-label { color: #667a97; font-size: 0.72rem; font-weight: 840; }
    .summary-value { color: #07152d; font-size: 0.86rem; font-weight: 900; margin-top: 0.12rem; }
    .diff-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 3rem;
        gap: 0.6rem;
        color: #273b59;
        font-size: 0.78rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }
    .card-grid-4 {
        display: grid;
        grid-template-columns: 1.15fr 1fr 1fr 1.15fr;
        gap: 1rem;
    }
    .mini-card { padding: 1rem 1.05rem; min-height: 8.4rem; }
    .mini-body {
        display: grid;
        grid-template-columns: 3rem minmax(0, 1fr);
        gap: 0.85rem;
        align-items: center;
        margin-top: 0.8rem;
    }
    .ring {
        width: 3.1rem;
        height: 3.1rem;
        border-radius: 999px;
        background: conic-gradient(#1567f9 0 250deg, #e6eef8 250deg 360deg);
        display: grid;
        place-items: center;
    }
    .ring::after {
        content: "";
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 999px;
        background: white;
    }
    .bottom-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 1rem;
    }
    .wide-card { padding: 1rem 1.05rem; }
    .explain-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 9rem 2.4rem;
        gap: 1rem;
        align-items: center;
        margin-top: 0.75rem;
    }
    .explain-thumb {
        width: 9rem;
        height: 7.6rem;
        object-fit: cover;
        border-radius: 7px;
        border: 1px solid #d7e4f4;
    }
    .interpret-grid {
        display: grid;
        grid-template-columns: 0.85fr 1.35fr;
        gap: 1rem;
    }
    .interpret-row {
        display: grid;
        grid-template-columns: 2rem minmax(0, 1fr);
        gap: 0.65rem;
        align-items: start;
        padding: 0.5rem 0;
    }
    .conclusion-box {
        background: #f6f9ff;
        border: 1px solid #dbe8fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.75rem;
    }
    .risk-scale {
        display: grid;
        grid-template-columns: auto 1fr auto;
        align-items: center;
        gap: 0.7rem;
        color: #667a97;
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 1rem;
    }
    .risk-bar {
        height: 0.45rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #e4efff 0%, #e4efff 42%, #f08022 42%, #f08022 62%, #fff0e2 62%, #fff0e2 100%);
    }
    @media (max-width: 1180px) {
        .workflow-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .model-grid, .metrics-grid, .trace-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .image-grid { grid-template-columns: 1fr; }
        .legend-card { display: none; }
        .identity-grid { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0.7rem; }
        .top-actions { justify-content: start; flex-wrap: wrap; }
        .analysis-layout, .bottom-grid, .interpret-grid { grid-template-columns: 1fr; }
        .viewer-grid { grid-template-columns: 1fr; }
        .viewer-img { height: clamp(19rem, 44vh, 30rem); }
        .analysis-tools, .viewer-legend { display: none; }
        .summary-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .card-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .cxr-stepper-row { grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.25rem; }
        .cxr-step::after { display: none; }
    }
    @media (max-width: 760px) {
        .identity-grid, .metrics-grid, .model-grid, .workflow-row, .clinical-grid, .trace-grid, .summary-list { grid-template-columns: 1fr; }
        .top-actions { justify-content: start; flex-wrap: wrap; }
        .card-grid-4, .cxr-stepper-row, .explain-layout { grid-template-columns: 1fr; }
        .viewer-img { height: clamp(16rem, 42vh, 24rem); }
        .top-card { height: auto; }
    }
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


def stat_card(label: str, value: str, note: str = "", icon: str = "+", warn: bool = False) -> str:
    note_html = f"<div class='stat-note'>{escape(note)}</div>" if note else ""
    icon_class = "stat-icon warn" if warn else "stat-icon"
    mini_class = "stat-mini warn" if warn else "stat-mini"
    return (
        "<div class='stat-card'>"
        f"<div class='{icon_class}'>{escape(icon)}</div>"
        "<div>"
        f"<div class='stat-label'>{escape(label)}</div>"
        f"<div class='stat-value'>{escape(value)}</div>"
        f"{note_html}</div>"
        f"<div class='{mini_class}'>{escape(icon)}</div>"
        "</div>"
    )


def workflow_html(active_target: bool) -> str:
    stages = [
        ("1", "X-ray", "Input image", "XR"),
        ("2", "Analysis", "Model inference", "AI"),
        ("3", "Ensemble", "Optimized AUROC-weight", "%"),
        ("4", "Evidence", "Guideline + clinical", "+"),
        ("5", "Report", "Findings", "R"),
    ]
    step_html = []
    for index, title, subtitle, icon in stages:
        step_html.append(
            f"<span class='cxr-step'><span class='cxr-step-num'>{escape(index)}</span>{escape(title)}</span>"
        )
    return (
        "<div class='cxr-stepper'>"
        "<div class='cxr-stepper-row'>"
        + "<span class='cxr-rail-label'>Workflow</span>"
        + "".join(step_html)
        + "</div>"
        "</div>"
    )


def model_stack_html(model_source: str, confidence: float, agreement: str) -> str:
    is_fallback = model_source == "Fallback pretrained demo"
    conv_status = "pending" if is_fallback else "ready"
    rad_status = "pending" if is_fallback else "ready"
    ensemble_status = "pending" if is_fallback else "ready"
    conv_text = "Target: EigenCAM evidence. Current app mode waits for local checkpoint." if is_fallback else "Active local checkpoint with EigenCAM target."
    rad_text = "Target: Attention Rollout evidence. Current app mode waits for local checkpoint." if is_fallback else "Active RAD-DINO branch with Attention Rollout target."
    ens_text = (
        "Fallback mode: agreement unavailable; single-model confidence displayed."
        if is_fallback
        else f"Weighted ensemble active; agreement label: {agreement}."
    )
    return (
        "<div class='model-grid'>"
        "<div class='model-card'><div class='model-head'>"
        "<div><div class='model-name'>ConvNeXtV2</div><div class='model-role'>CNN classifier for 15 VinBigData findings</div></div>"
        f"<span class='status-pill {conv_status}'>{'Target' if is_fallback else 'Active'}</span></div>"
        f"<div class='model-metric'>{escape(conv_text)}</div></div>"
        "<div class='model-card'><div class='model-head'>"
        "<div><div class='model-name'>RAD-DINO</div><div class='model-role'>Radiology foundation model branch</div></div>"
        f"<span class='status-pill {rad_status}'>{'Target' if is_fallback else 'Active'}</span></div>"
        f"<div class='model-metric'>{escape(rad_text)}</div></div>"
        "<div class='model-card'><div class='model-head'>"
        "<div><div class='model-name'>Weighted Ensemble</div><div class='model-role'>Probability fusion + reliability signal</div></div>"
        f"<span class='status-pill {ensemble_status}'>{'Fallback' if is_fallback else 'Active'}</span></div>"
        f"<div class='model-metric'>{escape(ens_text)}<br>Primary calibrated score: {confidence:.2f}</div></div>"
        "</div>"
    )


def finding_pills(df: pd.DataFrame, limit: int = 5) -> str:
    pills = []
    for idx, row in df.head(limit).reset_index(drop=True).iterrows():
        label = escape(str(row["label"]))
        score = float(row["calibrated_probability"])
        primary = " primary" if idx == 0 else ""
        dot = "<span class='finding-dot'></span>" if idx == 0 else ""
        pills.append(
            f"<span class='finding-pill{primary}'>{dot}<span>{label}</span>"
            f"<span class='finding-score'>{score:.2f}</span></span>"
        )
    pills.append(f"<span class='view-all'>View all ({len(df)})</span>")
    return "<div class='findings-row'>" + "".join(pills) + "</div>"


def differential_rows(df: pd.DataFrame, start: int = 1, limit: int = 3) -> str:
    rows = []
    for row in df.iloc[start : start + limit].itertuples():
        rows.append(
            "<div class='diff-row'>"
            f"<span>{escape(str(row.label))}</span>"
            f"<span>{float(row.calibrated_probability):.2f}</span>"
            "</div>"
        )
    if not rows:
        rows.append("<div class='diff-row'><span>No differential finding</span><span>-</span></div>")
    return "".join(rows)


def image_data_uri(image_obj: Image.Image | object) -> str:
    if isinstance(image_obj, Image.Image):
        pil_image = image_obj.convert("RGB")
    else:
        pil_image = Image.fromarray(image_obj).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_card(title: str, image_obj: Image.Image | object) -> str:
    return (
        "<div class='image-card'>"
        "<div class='image-title'>"
        f"<div class='section-label'>{escape(title)}</div>"
        "<div class='expand-icon'>&#8599;</div>"
        "</div>"
        f"<img class='xray-img' src='{image_data_uri(image_obj)}' alt='{escape(title)}' />"
        "</div>"
    )


def traceability_html(report: dict[str, str], primary_label: str, confidence: float, agreement: str) -> str:
    return (
        "<div class='trace-card'>"
        "<div class='section-label'>Prediction To Report Traceability</div>"
        "<div class='trace-grid'>"
        "<div class='trace-item'>"
        "<div class='trace-label'>Prediction</div>"
        f"<div class='trace-text'>{escape(primary_label)}<br>Calibrated probability: {confidence:.2f}</div>"
        "</div>"
        "<div class='trace-item'>"
        "<div class='trace-label'>Visual Evidence</div>"
        f"<div class='trace-text'>{escape(report.get('visual_evidence', 'Unavailable'))}</div>"
        "</div>"
        "<div class='trace-item'>"
        "<div class='trace-label'>Clinical Evidence</div>"
        f"<div class='trace-text'>{escape(report.get('clinical_evidence', 'Unavailable'))}</div>"
        "</div>"
        "<div class='trace-item'>"
        "<div class='trace-label'>Agreement / Safety</div>"
        f"<div class='trace-text'>{escape(agreement)}<br>{escape(report.get('flag', 'Unavailable'))}</div>"
        "</div>"
        "</div>"
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
patient_id = "DEMO 001"
age = 54
sex = "Unknown"
view_position = "PA"
lung_icon_svg = """
<svg class='lung-svg' viewBox='0 0 64 64' aria-hidden='true'>
  <path d='M31 7v18c-6-7-12-11-18-12-5 7-8 18-8 31 0 7 3 12 8 12 10 0 14-15 18-25' fill='none' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>
  <path d='M33 25V7m0 24c4 10 8 25 18 25 5 0 8-5 8-12 0-13-3-24-8-31-6 1-12 5-18 12' fill='none' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>
  <line x1='32' y1='7' x2='32' y2='4' stroke-width='3' stroke-linecap='round'/>
</svg>
"""

with st.sidebar:
    st.markdown(
        f"""
        <div class='sidebar-brand'>
          <div class='brand-mark'>{lung_icon_svg}</div>
          <div>
            <div class='brand-title'>XRayNet</div>
            <div class='brand-subtitle'>Clinical CXR AI Assistant</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-label'>Study</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='side-nav'>
          <div class='side-nav-item primary'><span class='side-icon'>UP</span><span>Upload Image</span></div>
          <div class='side-nav-item'><span class='side-icon'>DC</span><span>Demo Cases</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    demo_cases = load_demo_cases()
    source_name = "User upload"
    study_description = "Uploaded chest X-ray"
    with st.expander("Upload Image", expanded=False):
        uploaded = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
    if uploaded is not None:
        image = Image.open(uploaded)
        image_path = Path(uploaded.name)
        case_note = "User uploaded image"
        source_name = "Upload"
        study_description = uploaded.name
    else:
        if demo_cases.empty:
            st.error("No demo images found in data/demo_cases or data/raw.")
            st.stop()
        case_options = [f"{row.case_id}  {row.title}" for row in demo_cases.itertuples()]
        default_case_index = min(16, len(case_options) - 1)
        st.markdown("<div class='case-select-spacer'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Case Selection</div>", unsafe_allow_html=True)
        selected_label = st.selectbox("Demo case", case_options, index=default_case_index, label_visibility="collapsed")
        selected_row = demo_cases.iloc[case_options.index(selected_label)]
        image_path = Path(str(selected_row["path"]))
        image = Image.open(image_path)
        case_note = f"{selected_row['source']} | {selected_row['note']}"
        source_name = str(selected_row["source"])
        study_description = str(selected_row["note"])

    st.divider()
    st.markdown("<div class='section-label'>Patient Info</div>", unsafe_allow_html=True)
    st.markdown(
        (
            f"<div class='side-muted'>Patient ID</div><div class='side-detail'>{escape(patient_id)}</div>"
            f"<div class='side-muted'>Age</div><div class='side-detail'>{age} years</div>"
            f"<div class='side-muted'>Sex</div><div class='side-detail'>{escape(sex)}</div>"
            f"<div class='side-muted'>View Position</div><div class='side-detail'>{escape(view_position)}</div>"
            f"<div class='side-muted'>Source</div><div class='side-detail'>{escape(source_name)}</div>"
            f"<div class='side-muted'>Study Description</div><div class='side-detail'>{escape(study_description)}</div>"
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    with st.expander("Advanced AI Protocol", expanded=False):
        patient_id = st.text_input("Patient ID", patient_id)
        age = st.number_input("Age", min_value=1, max_value=110, value=age)
        sex = st.selectbox("Sex", ["Unknown", "Female", "Male"], index=0)
        view_position = st.segmented_control("View", ["PA", "AP", "Lateral"], default=view_position)
        model_source = st.radio("Model source", ["Fallback pretrained demo", "Local checkpoint"], label_visibility="collapsed")
        xrv_weights = st.selectbox("Pretrained weights", ["densenet121-res224-all"])
        threshold = st.slider("Abnormal threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        temperature = st.slider("Calibration temperature", min_value=0.7, max_value=3.0, value=1.4, step=0.1)
        top_k = st.slider("Top findings", min_value=3, max_value=10, value=5, step=1)
        checkpoint_path = st.text_input("Checkpoint", "outputs/checkpoints/best.pt")
        config_path = st.text_input("Config fallback", "configs/default.yaml")
        st.button("Run Analysis", type="primary", use_container_width=True)
    st.markdown("<div class='sidebar-footer'>AI for clinical support only.</div>", unsafe_allow_html=True)

device = get_device("auto")
with st.spinner("Running classification, calibration, fallback visual evidence, and clinical report generation..."):
    try:
        if model_source == "Fallback pretrained demo":
            model = get_pretrained_model(xrv_weights, str(device))
            analysis = analyze_pretrained_case(model, image, device, threshold, temperature=temperature)
        else:
            if not Path(checkpoint_path).exists():
                st.error(f"Checkpoint not found: {checkpoint_path}")
                st.stop()
            get_checkpoint_model(checkpoint_path, config_path, str(device))
            st.error("ConvNeXtV2/RAD-DINO local checkpoint inference is not wired into this demo yet.")
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

agreement_label = str(analysis.report.get("agreement", "Unavailable"))
is_target_mode = model_source != "Fallback pretrained demo"
if model_source == "Fallback pretrained demo":
    mode_note = (
        "Current execution mode: fallback pretrained TorchXRayVision demo. "
        "The clinical target architecture remains ConvNeXtV2 + RAD-DINO weighted ensemble with EigenCAM and Attention Rollout."
    )
    explainability_title = "2. Visual Evidence (Fallback CAM)"
else:
    mode_note = (
        "Current execution mode: local checkpoint target. ConvNeXtV2/RAD-DINO outputs should feed weighted ensemble, "
        "agreement score, EigenCAM, Attention Rollout, and grounded report generation."
    )
    explainability_title = "2. Explainability (EigenCAM / Attention Rollout)"

top_five = analysis.predictions.head(5).reset_index(drop=True)
chips_html = "".join(
    (
        f"<span class='xnet-chip {'primary' if idx == 0 else ''}'>"
        f"{escape(str(row['label']))}<span class='xnet-chip-score'>{float(row['calibrated_probability']):.2f}</span>"
        "</span>"
    )
    for idx, row in top_five.iterrows()
)
diff_html = "".join(
    f"<div>{idx + 1}. {escape(str(row['label']))}</div>"
    for idx, row in analysis.predictions.iloc[1:4].reset_index(drop=True).iterrows()
)
if not diff_html:
    diff_html = "<div>No differential finding</div>"

primary_label_html = escape(analysis.primary_label)
primary_confidence_html = f"{analysis.primary_confidence:.2f}"
risk_html = escape(analysis.risk)
study_date = date.today().isoformat()
visual_evidence = escape(str(analysis.report.get("visual_evidence", "Visual evidence unavailable.")))
clinical_evidence = escape(str(analysis.report.get("clinical_evidence", "Clinical evidence unavailable.")))
clinical_reasoning = escape(str(analysis.report.get("clinical_reasoning", "Radiologist review recommended.")))
impression = escape(str(analysis.report.get("impression", "AI impression unavailable.")))
note_text = (
    f"Review AI impression of {analysis.primary_label}. "
    "Correlation with clinical findings and follow-up recommended."
)

st.markdown(
    (
        "<div class='xnet-dashboard'>"
        "<div class='xnet-header'>"
        f"<div class='xnet-brand'><div class='xnet-logo'>{lung_icon_svg}</div><div>"
        "<div class='xnet-title'>XRayNet - System Result Output</div>"
        "<div class='xnet-subtitle'>Clinical Chest X-ray AI Assistant</div>"
        "</div></div>"
        "<div class='xnet-actions'>"
        "<div class='xnet-action'><span class='mini-icon'>PDF</span>Export PDF</div>"
        "<div class='xnet-action'><span class='mini-icon'>SV</span>Save Case</div>"
        "<div class='xnet-action'><span class='mini-icon'>ED</span>Radiologist Review / Edit</div>"
        "</div>"
        "</div>"
        "<div class='xnet-study-strip'>"
        "<div class='xnet-study-item'><div class='xnet-study-icon'>PT</div><div><div class='xnet-kicker'>Patient</div>"
        f"<div class='xnet-field'>{escape(patient_id)}</div></div></div>"
        "<div class='xnet-study-item'><div class='xnet-study-icon'>DT</div><div><div class='xnet-kicker'>Date</div>"
        f"<div class='xnet-field'>{study_date}</div></div></div>"
        "<div class='xnet-study-item'><div class='xnet-study-icon'>VW</div><div><div class='xnet-kicker'>View</div>"
        f"<div class='xnet-field'>{escape(view_position)}</div></div></div>"
        "<div class='xnet-study-item'><div class='xnet-study-icon'>RX</div><div><div class='xnet-kicker'>Study</div>"
        "<div class='xnet-field'>CHEST X-RAY</div></div></div>"
        "</div>"
        "<div class='xnet-content'>"
        "<div class='xnet-panel xnet-image-panel'>"
        "<div class='xnet-panel-title'>XR Chest X-ray Analysis</div>"
        "<div class='xnet-image-grid'>"
        "<div class='xnet-image-frame'><div class='xnet-image-label'>Original Chest X-ray (PA View)</div>"
        f"<img class='xnet-image' src='{image_data_uri(image)}' alt='Original chest X-ray' /></div>"
        "<div class='xnet-image-frame'><div class='xnet-image-label'>AI Heatmap Visual Evidence</div>"
        f"<img class='xnet-image' src='{image_data_uri(analysis.gradcam_overlay)}' alt='AI heatmap visual evidence' /></div>"
        "</div></div>"
        "<div class='xnet-panel xnet-report-panel'>"
        "<div class='xnet-panel-title'>RS Report Summary</div>"
        "<div class='xnet-report-grid'>"
        "<div class='xnet-report-card'><div class='xnet-report-icon'>XR</div><div class='xnet-card-label'>Primary Finding</div>"
        f"<div class='xnet-card-value'>{primary_label_html}</div></div>"
        "<div class='xnet-report-card'><div class='xnet-report-icon teal'>OK</div><div class='xnet-card-label'>Confidence</div>"
        f"<div class='xnet-card-value prob'>{primary_confidence_html}</div><div class='xnet-card-note'>Calibrated probability</div></div>"
        "<div class='xnet-report-card'><div class='xnet-report-icon orange'>!</div><div class='xnet-card-label'>Risk Level</div>"
        f"<div class='xnet-card-value risk'>{risk_html}</div><div class='xnet-card-note'>Triage indicator</div></div>"
        "<div class='xnet-report-card'><div class='xnet-report-icon blue'>TD</div><div class='xnet-card-label'>Top Differential</div>"
        f"<div class='xnet-diff-list'>{diff_html}</div></div>"
        "</div></div>"
        "<div class='xnet-panel xnet-findings-panel'>"
        "<div class='xnet-panel-title'>AI Findings (Top 5)</div>"
        f"<div class='xnet-chip-row'>{chips_html}</div>"
        "</div>"
        "<div class='xnet-panel xnet-lower-left'>"
        "<div class='xnet-panel-title'>Explainability</div>"
        "<div class='xnet-evidence'><div>"
        f"<div class='xnet-check-row'><span class='xnet-check'>OK</span><div class='xnet-evidence-text'>{visual_evidence}</div></div>"
        f"<div class='xnet-check-row'><span class='xnet-check'>OK</span><div class='xnet-evidence-text'>{clinical_evidence} {clinical_reasoning}</div></div>"
        "</div><div class='xnet-mini-legend'><div>High</div><div class='xnet-mini-bar'></div><div>Low</div></div>"
        "</div></div>"
        "<div class='xnet-panel xnet-conclusion-panel'>"
        "<div class='xnet-conclusion-card'><div class='xnet-panel-title'>Conclusion</div>"
        f"<div class='xnet-body'>{impression}<br><br>Findings should be reviewed by a radiologist for clinical correlation.</div></div>"
        "<div class='xnet-notes-card'><div class='xnet-panel-title'>Radiologist Review / Doctor Notes</div>"
        "<div class='xnet-body'><div class='xnet-notes-box'>"
        f"{escape(note_text)}"
        "</div><div class='xnet-note-actions'><span class='xnet-note-btn'>Reset</span><span class='xnet-note-btn primary'>Save Notes</span></div></div></div>"
        "</div>"
        "</div>"
        "<div class='xnet-footer'>For research support only. Radiologist review required.</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)
