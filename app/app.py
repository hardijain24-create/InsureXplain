"""
InsureXplain — Interpretable Insurance Claim Risk Engine
=========================================================
Premium dashboard redesign: clean, professional, no emojis, Plotly charts.

Deliverable 1 : SHAP (local + global) + LIME explanations, waterfall plot
Deliverable 2 : Claim Review Notice generator
Deliverable 3 : Global Regulatory Compliance Report (proxy audit, feature interactions)
Validation    : LIME vs SHAP on 3 representative records

Scope: Auto-insurance claim FRAUD RISK detection (fraud_reported Y/N).
       This is NOT an underwriting, coverage-denial, or premium-pricing engine.
"""

import os
import io
import base64
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    risk_category,
    plain_english_factor as _plain_english_factor,
    generate_adverse_action_notice,
    build_compliance_report_html,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureXplain — Claim Risk Intelligence",
    layout="wide",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>IX</text></svg>",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system — CSS variables + component styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

/* ---------- App shell ---------- */
.stApp { background: #F5F6F7; }
.block-container { padding: 1.6rem 2rem 3rem 2rem !important; max-width: 1280px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #1E2126 !important;
    border-right: 1px solid #2A2D31;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }
section[data-testid="stSidebar"] * { color: #C8CDD4 !important; }
section[data-testid="stSidebar"] .stMarkdown hr { border-color: #2A2D31 !important; }

/* File uploader in sidebar */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #1E2126 !important;
    border: 1.5px dashed #3A3D44 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #C8CDD4 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #2A2D31 !important;
    border: 1px solid #3A3D44 !important;
    color: #E6E9F5 !important;
}
section[data-testid="stSidebar"] small { color: #737780 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    background: #1E2126 !important;
    color: #C8CDD4 !important;
}

/* ---------- Section heading system ---------- */
.ix-page-header {
    padding: 1.4rem 0 1rem 0;
    border-bottom: 2px solid #E6E8EB;
    margin-bottom: 1.4rem;
}
.ix-page-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #17191C;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
}
.ix-page-subtitle {
    font-size: 0.88rem;
    color: #737780;
    margin: 0;
}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E6E8EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
div[data-testid="stMetricLabel"] p { color: #737780 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] { color: #17191C !important; font-weight: 700 !important; font-size: 1.7rem !important; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ---------- Cards / panels ---------- */
.ix-card {
    background: #FFFFFF;
    border: 1px solid #E6E8EB;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.ix-card-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #17191C;
    margin: 0 0 0.8rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Risk badges ---------- */
.ix-badge {
    display: inline-block;
    padding: 0.22rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.ix-badge-high   { background: #FEECEC; color: #B91C1C; border: 1px solid #FECACA; }
.ix-badge-medium { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.ix-badge-low    { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }

/* ---------- Notice preview ---------- */
.ix-notice {
    background: #FFFFFF;
    border: 1px solid #E6E8EB;
    border-left: 4px solid #9BCB45;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    font-size: 0.88rem;
    line-height: 1.75;
    white-space: pre-wrap;
    color: #17191C;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    max-height: 520px;
    overflow-y: auto;
}

/* ---------- Nav links in sidebar ---------- */
.ix-nav-label {
    display: block;
    font-size: 0.87rem;
    font-weight: 500;
    color: #9BA1A9;
    padding: 0.45rem 0.8rem;
    border-radius: 6px;
    margin: 0.1rem 0;
    cursor: pointer;
    text-decoration: none;
}
.ix-nav-active {
    background: #1E2126;
    color: #FFFFFF !important;
}
.ix-nav-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4A4F57;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 1rem 0.8rem 0.3rem 0.8rem;
    display: block;
}

/* ---------- Wordmark ---------- */
.ix-wordmark {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.01em;
}
.ix-wordmark span { color: #9BCB45 !important; }

/* ---------- Tables / dataframes ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid #E6E8EB;
    border-radius: 8px;
    overflow: hidden;
}

/* ---------- Buttons ---------- */
.stDownloadButton button, .stButton > button {
    background: #17191C !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 0.45rem 1.2rem !important;
    letter-spacing: 0.01em;
}
.stDownloadButton button:hover, .stButton > button:hover {
    background: #2A2D31 !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #FFFFFF;
    border: 1px solid #E6E8EB;
    border-radius: 8px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #737780;
    font-weight: 500;
    font-size: 0.87rem;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #17191C !important;
    color: #FFFFFF !important;
}

/* ---------- Expanders ---------- */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    color: #17191C !important;
}

/* ---------- Info/warning/success boxes ---------- */
div[data-testid="stAlertContainer"] { border-radius: 8px; font-size: 0.88rem; }

/* ---------- Sidebar radio nav ---------- */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.87rem !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants & colour palette
# ─────────────────────────────────────────────────────────────────────────────
ACCENT       = "#9BCB45"
ACCENT_DARK  = "#6C8D3B"
COL_PRIMARY  = "#17191C"
COL_SECONDARY= "#737780"
COL_BORDER   = "#E6E8EB"
COL_SURFACE  = "#FFFFFF"
COL_BG       = "#F5F6F7"
COL_RED      = "#DC2626"
COL_AMBER    = "#D97706"
COL_GREEN    = "#059669"
COL_CHART    = "#8CB8D8"

BADGE_CLASSES = {"High": "ix-badge-high", "Medium": "ix-badge-medium", "Low": "ix-badge-low"}

SECTIONS = [
    "Overview",
    "Claim Analysis",
    "Local Explainability",
    "Global Explainability",
    "Claim Review Notice",
    "Compliance & Fairness",
    "LIME vs SHAP Validation",
    "About",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: reusable UI components
# ─────────────────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="ix-page-header">'
        f'<div class="ix-page-title">{title}</div>'
        f'{"<div class=ix-page-subtitle>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    cls = BADGE_CLASSES.get(category, "ix-badge-medium")
    return f'<span class="ix-badge {cls}">{category} Risk</span>'


def card(content_fn, title: str = ""):
    st.markdown(
        f'<div class="ix-card">'
        f'{"<div class=ix-card-title>" + title + "</div>" if title else ""}',
        unsafe_allow_html=True,
    )
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def plotly_defaults(fig):
    """Apply consistent styling to every Plotly figure."""
    fig.update_layout(
        paper_bgcolor=COL_SURFACE,
        plot_bgcolor=COL_BG,
        font=dict(family="Inter, system-ui, sans-serif", color=COL_PRIMARY, size=12),
        margin=dict(l=16, r=16, t=36, b=16),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COL_BORDER, linecolor=COL_BORDER, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COL_BORDER, linecolor=COL_BORDER, zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    m   = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    fc  = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    cc  = joblib.load(os.path.join(MODEL_DIR, "categorical_columns.pkl"))
    return m, fc, cc


try:
    model, feature_columns, categorical_columns = load_model_artifacts()
except FileNotFoundError as _exc:
    st.error(
        "Model artifacts not found in models/. "
        "Run notebook 04_modeling_explainability.ipynb to generate them."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors notebook 03 exactly
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "incident_date" in df.columns and "policy_bind_date" in df.columns:
        df["incident_date"]    = pd.to_datetime(df["incident_date"],    errors="coerce")
        df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")
        df["policy_age_days"]  = (df["incident_date"] - df["policy_bind_date"]).dt.days
    if "months_as_customer" in df.columns:
        df["customer_tenure_years"] = df["months_as_customer"] / 12
        if "total_claim_amount" in df.columns:
            df["claim_amount_per_tenure_year"] = (
                df["total_claim_amount"] / df["customer_tenure_years"].replace(0, 0.1)
            )
    if "total_claim_amount" in df.columns:
        thr = df["total_claim_amount"].quantile(0.90)
        df["is_high_value_claim"] = (df["total_claim_amount"] >= thr).astype(int)
    for part in ["vehicle_claim", "injury_claim", "property_claim"]:
        if part in df.columns and "total_claim_amount" in df.columns:
            df[f"{part}_ratio"] = df[part] / df["total_claim_amount"].replace(0, np.nan)
    if all(c in df.columns for c in ["police_report_available", "witnesses", "property_damage"]):
        df["missing_doc_count"] = (
            df["police_report_available"].isin(["NO", "Unknown", "?"]).astype(int)
            + (df["witnesses"] == 0).astype(int)
            + df["property_damage"].isin(["NO", "Unknown", "?"]).astype(int)
        )
    if "incident_hour_of_the_day" in df.columns:
        df["is_odd_hour_incident"] = df["incident_hour_of_the_day"].apply(
            lambda h: 1 if pd.notnull(h) and (h <= 5 or h >= 22) else 0
        )
    return df


def prepare_for_model(df_raw: pd.DataFrame):
    """Clean, engineer, encode, align → returns (df_feat, X_aligned)."""
    df = df_raw.replace("?", np.nan)
    df_feat = engineer_features(df)
    drop_cols = [
        "fraud_reported", "fraud_flag", "incident_date", "policy_bind_date",
        "incident_location", "policy_number", "insured_zip", "_c39",
    ]
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors="ignore")
    cats = [c for c in categorical_columns if c in X.columns]
    X_enc = pd.get_dummies(X, columns=cats, drop_first=True)
    X_aligned = X_enc.reindex(columns=feature_columns, fill_value=0)
    return df_feat, X_aligned


# ─────────────────────────────────────────────────────────────────────────────
# SHAP caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def compute_shap_values(_X_values, _col_names, n_rows: int):
    """Compute SHAP values for first n_rows rows; cached by shape + hash."""
    expl = get_shap_explainer(model)
    X_df = pd.DataFrame(_X_values[:n_rows], columns=_col_names)
    raw  = expl.shap_values(X_df)
    if isinstance(raw, list):
        sv = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        sv = raw[:, :, 1]
    else:
        sv = raw
    ev = expl.expected_value
    base = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
    return sv, base


# ─────────────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_lime_explainer(_X_values, _col_names):
    try:
        from lime.lime_tabular import LimeTabularExplainer
        return LimeTabularExplainer(
            training_data=_X_values,
            feature_names=list(_col_names),
            class_names=["Genuine", "Fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
    except ImportError:
        return None


def get_lime_explanation(lime_exp, row_arr, num_features=10, num_samples=500):
    if lime_exp is None:
        return []
    try:
        exp = lime_exp.explain_instance(
            data_row=row_arr,
            predict_fn=model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )
        return exp.as_list(label=1)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Gemini / AI summary (optional)
# ─────────────────────────────────────────────────────────────────────────────
def generate_ai_summary(claim_summary: dict, top_factors: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    factors_text = "; ".join(
        [f"{f['feature']} ({f['impact']:+.3f})" for f in top_factors]
    )
    prompt = (
        "You are assisting an insurance investigator. Based ONLY on the structured claim data "
        "and model output below, write a short (2-3 sentence) plain-English explanation of why "
        "this claim received its risk score. Do not decide whether it is fraud — only explain "
        "the signals. Do not invent any facts not given below.\n\n"
        f"Risk probability: {claim_summary['risk_probability']:.0%}\n"
        f"Risk category: {claim_summary['risk_category']}\n"
        f"Most influential factors: {factors_text}\n"
        f"Claim amount: {claim_summary.get('total_claim_amount', 'N/A')}\n"
    )

    if not api_key:
        names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"This claim was classified as **{claim_summary['risk_category']} Risk** "
            f"({claim_summary['risk_probability']:.0%} probability) "
            f"primarily due to: {names}. "
            "(Provide a GEMINI_API_KEY environment variable to enable live AI summaries.)"
        )
    try:
        for mn in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{mn}:generateContent",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raise RuntimeError("All model endpoints failed")
    except Exception as exc:
        names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"[AI summary unavailable — {exc}] "
            f"Classified as {claim_summary['risk_category']} Risk "
            f"({claim_summary['risk_probability']:.0%}) due to: {names}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="ix-wordmark">Insure<span>X</span>plain</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.75rem;color:#4A4F57;margin-top:2px;margin-bottom:1.2rem;">'
        "Claim Risk Intelligence Platform</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Section navigation
    st.markdown('<span class="ix-nav-section">Navigation</span>', unsafe_allow_html=True)
    active_section = st.radio(
        "Section",
        SECTIONS,
        label_visibility="collapsed",
        key="nav_section",
    )

    st.markdown("---")
    st.markdown('<span class="ix-nav-section">Data</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload claims CSV", type=["csv"], label_visibility="visible")
    st.markdown(
        '<div style="font-size:0.75rem;color:#4A4F57;margin-top:0.5rem;">'
        "Use data/insurance_claims.csv to demo.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#4A4F57;line-height:1.5;">'
        "Model: Random Forest (300 trees)<br>"
        "Target: Fraud risk (fraud_reported)<br>"
        "Not an underwriting decision tool.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gate: require upload for all sections except About
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None and active_section != "About":
    page_header(
        active_section,
        "Upload a claims CSV from the sidebar to get started.",
    )
    st.info(
        "No file uploaded yet. Use the sidebar to upload a CSV file. "
        "You can use data/insurance_claims.csv from the repository to explore all features."
    )
    if active_section == "Overview":
        # Show static model summary cards even without data
        st.markdown("### Model at a glance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm", "Random Forest")
        c2.metric("Trees", "300")
        c3.metric("Max Depth", "8")
        c4.metric("Class Weighting", "Balanced")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Load and score data (cached per file content)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_score(file_bytes: bytes):
    import io as _io
    df_raw = pd.read_csv(_io.BytesIO(file_bytes))
    df_feat, X_aligned = prepare_for_model(df_raw)
    probs = model.predict_proba(X_aligned)[:, 1]
    df_feat["risk_probability"] = probs
    df_feat["risk_category"]    = probs.apply(risk_category) if hasattr(probs, "apply") \
                                  else pd.Series(probs).apply(risk_category).values
    return df_raw, df_feat, X_aligned


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Loading and scoring claims..."):
        df_raw, df_feat, X_aligned = load_and_score(file_bytes)

    # Compute SHAP (cached)
    N_SHAP = min(500, len(X_aligned))
    with st.spinner("Computing SHAP values..."):
        shap_vals, base_value = compute_shap_values(
            X_aligned.values, tuple(X_aligned.columns), N_SHAP
        )

    # Model metrics (only if labels present)
    model_metrics_dict = {}
    if "fraud_reported" in df_raw.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        y_true = df_raw["fraud_reported"].map({"Y": 1, "N": 0})
        y_pred = (df_feat["risk_probability"] >= 0.5).astype(int)
        model_metrics_dict = {
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
            "Recall":    f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
            "F1 Score":  f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
            "ROC-AUC":   f"{roc_auc_score(y_true, df_feat['risk_probability']):.3f}",
        }

    # Global SHAP importance
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": X_aligned.columns[:N_SHAP] if len(X_aligned.columns) <= N_SHAP
                      else X_aligned.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Overview
# ─────────────────────────────────────────────────────────────────────────────
if active_section == "Overview":
    page_header(
        "Overview",
        f"Dataset: {len(df_raw):,} claims analyzed — {datetime.date.today().strftime('%d %b %Y')}",
    )

    # Top metrics
    n_high   = int((df_feat["risk_category"] == "High").sum())
    n_medium = int((df_feat["risk_category"] == "Medium").sum())
    n_low    = int((df_feat["risk_category"] == "Low").sum())
    avg_risk = float(df_feat["risk_probability"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Claims Analyzed",   f"{len(df_raw):,}")
    c2.metric("High Risk",         f"{n_high:,}",   delta=f"{n_high/len(df_raw):.0%} of total")
    c3.metric("Medium Risk",       f"{n_medium:,}")
    c4.metric("Low Risk",          f"{n_low:,}")
    c5.metric("Avg Fraud Prob.",   f"{avg_risk:.1%}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Two-column layout: distribution chart + top features
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="ix-card">', unsafe_allow_html=True)
        st.markdown('<div class="ix-card-title">Fraud Risk Distribution</div>', unsafe_allow_html=True)
        bin_series = pd.cut(df_feat["risk_probability"], bins=20)
        hist_counts = bin_series.value_counts(sort=False)
        hist_data = pd.DataFrame({"bin": hist_counts.index, "count": hist_counts.values})
        hist_data["bin_mid"] = hist_data["bin"].apply(lambda x: x.mid)
        fig_dist = go.Figure()
        colors_hist = [
            COL_RED if m >= 0.6 else (COL_AMBER if m >= 0.3 else COL_GREEN)
            for m in hist_data["bin_mid"]
        ]
        fig_dist.add_trace(go.Bar(
            x=[f"{b.left:.2f}–{b.right:.2f}" for b in hist_data["bin"]],
            y=hist_data["count"],
            marker_color=colors_hist,
            hovertemplate="Range: %{x}<br>Claims: %{y}<extra></extra>",
        ))
        fig_dist.update_layout(
            title="", xaxis_title="Fraud Risk Probability",
            yaxis_title="Number of Claims", showlegend=False,
        )
        plotly_defaults(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="ix-card">', unsafe_allow_html=True)
        st.markdown('<div class="ix-card-title">Risk Category Breakdown</div>', unsafe_allow_html=True)
        cat_counts = df_feat["risk_category"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            hole=0.52,
            marker_colors=[COL_RED, COL_AMBER, COL_GREEN],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} claims (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
        plotly_defaults(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Model performance if available
        if model_metrics_dict:
            st.markdown('<div class="ix-card">', unsafe_allow_html=True)
            st.markdown('<div class="ix-card-title">Model Performance</div>', unsafe_allow_html=True)
            for k, v in model_metrics_dict.items():
                sc1, sc2 = st.columns([2, 1])
                sc1.markdown(f'<span style="font-size:0.85rem;color:{COL_SECONDARY}">{k}</span>', unsafe_allow_html=True)
                sc2.markdown(f'<span style="font-size:0.9rem;font-weight:700;color:{COL_PRIMARY}">{v}</span>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.75rem;color:#737780;margin-top:0.5rem;">'
                "Evaluated on uploaded labeled data. Threshold = 0.5.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Top features preview
    st.markdown('<div class="ix-card">', unsafe_allow_html=True)
    st.markdown('<div class="ix-card-title">Top Model Features (Mean |SHAP| across dataset)</div>', unsafe_allow_html=True)
    top10 = importance_df.head(10)
    fig_feat = go.Figure(go.Bar(
        x=top10["mean_abs_shap"][::-1],
        y=top10["feature"][::-1],
        orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_feat.update_layout(
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        showlegend=False,
        height=340,
    )
    plotly_defaults(fig_feat)
    st.plotly_chart(fig_feat, use_container_width=True)
    st.markdown(
        '<div style="font-size:0.78rem;color:#737780;">'
        "Mean absolute SHAP value across the first 500 uploaded claims. "
        "This reflects statistical association with model output — not proof of causality.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick actions
    st.markdown("### Quick Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("Analyze a Claim", use_container_width=True):
            st.session_state["nav_section"] = "Claim Analysis"
            st.rerun()
    with qa2:
        if st.button("View Explainability", use_container_width=True):
            st.session_state["nav_section"] = "Local Explainability"
            st.rerun()
    with qa3:
        if st.button("Generate Notice", use_container_width=True):
            st.session_state["nav_section"] = "Claim Review Notice"
            st.rerun()
    with qa4:
        if st.button("Compliance Report", use_container_width=True):
            st.session_state["nav_section"] = "Compliance & Fairness"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Claim Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "Claim Analysis":
    page_header(
        "Claim Analysis",
        "Browse all scored claims and inspect individual records.",
    )

    # Filter controls
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        risk_filter = st.multiselect(
            "Filter by risk category",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
    with fc2:
        sort_by = st.selectbox(
            "Sort by",
            ["risk_probability (desc)", "risk_probability (asc)", "total_claim_amount (desc)"],
        )

    ascending = "asc" in sort_by
    sort_col  = sort_by.split(" ")[0]
    if sort_col not in df_feat.columns:
        sort_col = "risk_probability"

    filtered = df_feat[df_feat["risk_category"].isin(risk_filter)].sort_values(
        sort_col, ascending=ascending
    )

    display_cols = [
        c for c in ["risk_probability", "risk_category", "total_claim_amount",
                    "incident_severity", "incident_type", "bodily_injuries", "witnesses"]
        if c in filtered.columns
    ]
    st.markdown(f"**{len(filtered):,} claims** match the current filter.")
    st.dataframe(
        filtered[display_cols]
        .style.format({"risk_probability": "{:.1%}"})
        .map(
            lambda v: f"color:{COL_RED};font-weight:700" if v == "High"
                      else (f"color:{COL_AMBER};font-weight:700" if v == "Medium"
                            else f"color:{COL_GREEN};font-weight:700"),
            subset=["risk_category"] if "risk_category" in display_cols else [],
        ),
        use_container_width=True,
        height=320,
    )

    # Individual claim inspector
    st.markdown("---")
    st.markdown("### Claim Detail")
    if len(filtered) == 0:
        st.info("No claims match the current filter.")
    else:
        selected_idx = st.selectbox(
            "Select claim",
            options=filtered.index.tolist(),
            format_func=lambda i: (
                f"Row {i} — {df_feat.loc[i,'risk_category']} Risk "
                f"({df_feat.loc[i,'risk_probability']:.0%})"
            ),
        )
        row = df_feat.loc[selected_idx]
        claim_id = (
            str(df_raw.loc[selected_idx, "policy_number"])
            if "policy_number" in df_raw.columns
            else f"CLAIM-{selected_idx:04d}"
        )

        # Store in session state for other sections
        pos = df_feat.index.get_loc(selected_idx)
        if pos < N_SHAP:
            claim_shap = shap_vals[pos]
            sv_df = (
                pd.DataFrame({"feature": X_aligned.columns, "impact": claim_shap})
                .sort_values("impact", key=abs, ascending=False)
            )
            shap_top3 = sv_df.head(3).to_dict("records")
        else:
            sv_df, shap_top3 = pd.DataFrame(), []

        st.session_state.update({
            "_selected_idx":  selected_idx,
            "_claim_id":      claim_id,
            "_shap_top3":     shap_top3,
            "_risk_prob":     float(row["risk_probability"]),
            "_risk_cat":      str(row["risk_category"]),
        })

        # Summary cards
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Fraud Risk Score", f"{row['risk_probability']:.1%}")
        d2.metric("Risk Category",    row["risk_category"])
        if "total_claim_amount" in row:
            d3.metric("Claim Amount", f"${row['total_claim_amount']:,.0f}")
        if "incident_severity" in row:
            d4.metric("Severity", str(row.get("incident_severity", "N/A")))

        st.markdown(
            f'<div style="margin:0.5rem 0 1rem 0">'
            f'{risk_badge(row["risk_category"])}'
            f'<span style="font-size:0.8rem;color:{COL_SECONDARY};margin-left:0.75rem;">'
            f"Claim reference: {claim_id}</span></div>",
            unsafe_allow_html=True,
        )

        # Attributes table
        attr_cols = [
            "incident_severity", "incident_type", "incident_hour_of_the_day",
            "bodily_injuries", "witnesses", "police_report_available",
            "property_damage", "number_of_vehicles_involved",
        ]
        avail = {c: df_raw.loc[selected_idx, c] for c in attr_cols if c in df_raw.columns}
        if avail:
            st.dataframe(
                pd.DataFrame(list(avail.items()), columns=["Attribute", "Value"]),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

        st.caption(
            "This risk score is a screening signal generated by a statistical model. "
            "It is not proof of fraud. A human claims investigator must review this case."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Local Explainability
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "Local Explainability":
    page_header(
        "Local Explainability",
        "SHAP and LIME explanations for an individual claim — why did the model give this score?",
    )

    if "_selected_idx" not in st.session_state:
        st.info("Select a claim in the Claim Analysis section first.")
        st.stop()

    selected_idx = st.session_state["_selected_idx"]
    claim_id     = st.session_state["_claim_id"]
    risk_prob    = st.session_state["_risk_prob"]
    risk_cat     = st.session_state["_risk_cat"]
    pos          = df_feat.index.get_loc(selected_idx)

    st.markdown(
        f"Explaining claim **{claim_id}** — "
        f"{risk_badge(risk_cat)} {risk_prob:.1%} predicted fraud probability",
        unsafe_allow_html=True,
    )

    # Tabs
    tab_shap, tab_lime, tab_compare = st.tabs(["SHAP Explanation", "LIME Explanation", "Compare Methods"])

    if pos >= N_SHAP:
        st.warning("SHAP values not available for this row (beyond the first 500 scored).")
        st.stop()

    claim_shap = shap_vals[pos]
    sv_df = (
        pd.DataFrame({"feature": X_aligned.columns, "impact": claim_shap})
        .sort_values("impact", key=abs, ascending=False)
    )
    shap_top10 = sv_df.head(10)
    shap_top3  = sv_df.head(3).to_dict("records")
    st.session_state["_shap_top3"] = shap_top3

    with tab_shap:
        st.markdown(
            "SHAP (SHapley Additive exPlanations) uses exact Shapley values from the Random Forest. "
            "Positive values increase the fraud-risk estimate; negative values decrease it. "
            "The base value is the model's average predicted probability across training data.",
        )

        # Plotly bar chart
        colors = [COL_RED if v > 0 else COL_CHART for v in shap_top10["impact"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_top10["impact"][::-1],
            y=[_plain_english_factor(f) for f in shap_top10["feature"]][::-1],
            orientation="h",
            marker_color=colors[::-1],
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        ))
        fig_shap.add_vline(x=0, line_color=COL_PRIMARY, line_width=1)
        fig_shap.update_layout(
            title=f"Top 10 SHAP Contributions — base value {base_value:.3f}",
            xaxis_title="SHAP value (impact on fraud-risk prediction)",
            yaxis_title="",
            showlegend=False,
            height=380,
        )
        plotly_defaults(fig_shap)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Waterfall
        st.markdown("**SHAP Waterfall** — how each feature moves the prediction from the base value")
        try:
            shap_exp = shap.Explanation(
                values=claim_shap,
                base_values=base_value,
                data=X_aligned.values[pos],
                feature_names=list(X_aligned.columns),
            )
            fig_wf, _ = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_exp, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close(fig_wf)
        except Exception as _e:
            st.info(f"Waterfall plot unavailable ({_e}). Showing ranked SHAP bar chart as substitute.")
            fig_wf2 = go.Figure(go.Bar(
                x=sv_df.head(12)["impact"],
                y=[_plain_english_factor(f) for f in sv_df.head(12)["feature"]],
                orientation="h",
                marker_color=[COL_RED if v > 0 else COL_CHART for v in sv_df.head(12)["impact"]],
            ))
            fig_wf2.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            plotly_defaults(fig_wf2)
            st.plotly_chart(fig_wf2, use_container_width=True)

        # Feature table
        with st.expander("Full SHAP contribution table"):
            display_sv = sv_df.copy()
            display_sv["plain_name"] = display_sv["feature"].apply(_plain_english_factor)
            display_sv["direction"]  = display_sv["impact"].apply(
                lambda v: "Increases risk" if v > 0 else "Decreases risk"
            )
            st.dataframe(
                display_sv[["plain_name", "impact", "direction"]]
                .rename(columns={"plain_name": "Feature", "impact": "SHAP Value", "direction": "Direction"})
                .head(30),
                use_container_width=True,
                hide_index=True,
            )

        # AI summary
        with st.expander("AI-generated investigator summary (optional)"):
            if st.button("Generate summary"):
                with st.spinner("Generating..."):
                    summary = generate_ai_summary(
                        {"risk_probability": risk_prob, "risk_category": risk_cat,
                         "total_claim_amount": df_feat.loc[selected_idx].get("total_claim_amount", "N/A")},
                        sv_df.head(6).to_dict("records"),
                    )
                st.info(summary)
                st.caption(
                    "This is an AI-generated explanation of model output only. "
                    "It is not a fraud determination — human review is required."
                )

    with tab_lime:
        st.markdown(
            "LIME (Local Interpretable Model-agnostic Explanations) fits a simple linear model "
            "around this prediction by perturbing the input (500 samples, seed=42). "
            "LIME describes features as interval conditions, e.g. 'amount > 50,000'.",
        )

        with st.spinner("Computing LIME explanation (~15 seconds)..."):
            lime_exp_obj = build_lime_explainer(X_aligned.values, tuple(X_aligned.columns))
            lime_raw     = get_lime_explanation(lime_exp_obj, X_aligned.values[pos])
            lime_top3    = lime_raw[:3] if lime_raw else []

        st.session_state["_lime_top3"] = lime_top3

        if lime_raw:
            lime_df = pd.DataFrame(lime_raw, columns=["feature_cond", "contribution"])
            colors_lime = [COL_RED if v > 0 else COL_CHART for v in lime_df["contribution"]]
            fig_lime = go.Figure(go.Bar(
                x=lime_df["contribution"][::-1],
                y=lime_df["feature_cond"][::-1],
                orientation="h",
                marker_color=colors_lime[::-1],
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ))
            fig_lime.add_vline(x=0, line_color=COL_PRIMARY, line_width=1)
            fig_lime.update_layout(
                title="Top LIME Contributions (class: Fraud)",
                xaxis_title="LIME contribution (positive = increases fraud probability)",
                yaxis_title="",
                showlegend=False,
                height=380,
            )
            plotly_defaults(fig_lime)
            st.plotly_chart(fig_lime, use_container_width=True)

            st.dataframe(
                lime_df.rename(columns={"feature_cond": "Condition", "contribution": "LIME Value"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "LIME explanation could not be computed for this claim. "
                "Ensure the lime package is installed (pip install lime)."
            )

    with tab_compare:
        st.markdown(
            "Direct comparison of SHAP and LIME for this claim. "
            "Note: SHAP provides exact Shapley values on exact feature values; "
            "LIME provides linear-approximation contributions on discretized conditions. "
            "Their numerical magnitudes are **not directly comparable**.",
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Top 3 SHAP Factors**")
            if shap_top3:
                st.dataframe(
                    pd.DataFrame([{
                        "Feature":    _plain_english_factor(f["feature"]),
                        "SHAP Value": f"{f['impact']:+.4f}",
                        "Direction":  "Increases risk" if f["impact"] > 0 else "Decreases risk",
                    } for f in shap_top3]),
                    use_container_width=True, hide_index=True,
                )
        with cc2:
            st.markdown("**Top 3 LIME Factors**")
            lime_top3_comp = st.session_state.get("_lime_top3", [])
            if lime_top3_comp:
                st.dataframe(
                    pd.DataFrame([{
                        "Condition":   fd,
                        "LIME Value":  f"{c:+.4f}",
                        "Direction":   "Increases risk" if c > 0 else "Decreases risk",
                    } for fd, c in lime_top3_comp]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("LIME result not available.")

        if shap_top3 and lime_top3_comp:
            shap_dir = shap_top3[0]["impact"] > 0
            lime_dir = lime_top3_comp[0][1] > 0
            if shap_dir == lime_dir:
                st.success(
                    "SHAP and LIME agree on the direction of the primary contributing factor. "
                    "This increases confidence in the local explanation for this claim."
                )
            else:
                st.warning(
                    "SHAP and LIME disagree on the direction of the top factor. "
                    "This may reflect LIME's local linear approximation not capturing "
                    "the model's non-linear decision boundary near this point. "
                    "Human review is especially recommended."
                )
            st.caption(
                "Disagreement between SHAP and LIME is expected to some degree and does not "
                "indicate an error. SHAP is exact for tree models; LIME is an approximation. "
                "Neither method proves causation."
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Global Explainability
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "Global Explainability":
    page_header(
        "Global Explainability",
        f"Dataset-level model interpretation across {N_SHAP} claims. "
        "Global importance does not establish causality.",
    )

    st.info(
        "Global explainability shows how the model behaves across the uploaded dataset — "
        "not for any single claim. "
        "Compare with Local Explainability for individual-level explanations."
    )

    # SHAP bar chart
    st.markdown("#### Feature Importance — Mean |SHAP| Value")
    fig_global = go.Figure(go.Bar(
        x=importance_df["mean_abs_shap"][::-1],
        y=[_plain_english_factor(f) for f in importance_df["feature"]][::-1],
        orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_global.update_layout(
        xaxis_title="Mean absolute SHAP value",
        yaxis_title="",
        showlegend=False,
        height=480,
    )
    plotly_defaults(fig_global)
    st.plotly_chart(fig_global, use_container_width=True)
    st.caption(
        "Rankings reflect average influence on the model's fraud-risk prediction across "
        f"the first {N_SHAP} uploaded claims. This is a statistical association measure."
    )

    # SHAP beeswarm
    st.markdown("#### SHAP Summary Plot (Beeswarm)")
    st.markdown(
        "Each dot represents one claim. The x-axis shows the SHAP value (impact on prediction). "
        "Colour shows the feature value (red = high, blue = low)."
    )
    try:
        n_bg = min(200, N_SHAP)
        fig_bee, ax_bee = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals[:n_bg],
            X_aligned.iloc[:n_bg],
            max_display=15,
            show=False,
            plot_size=None,
        )
        plt.title(f"Global SHAP Summary — top 15 features ({n_bg} claims)", fontsize=12, pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        global_shap_img_b64 = base64.b64encode(buf.read()).decode()
        st.pyplot(fig_bee)
        plt.close(fig_bee)
    except Exception as _bee_err:
        st.warning(f"Beeswarm plot could not be rendered: {_bee_err}")
        global_shap_img_b64 = None

    # Ranked table
    with st.expander("Full feature importance table"):
        disp_df = importance_df.copy()
        disp_df["plain_name"] = disp_df["feature"].apply(_plain_english_factor)
        disp_df["rank"] = range(1, len(disp_df) + 1)
        st.dataframe(
            disp_df[["rank", "plain_name", "mean_abs_shap"]]
            .rename(columns={"rank": "#", "plain_name": "Feature", "mean_abs_shap": "Mean |SHAP|"}),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Claim Review Notice
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "Claim Review Notice":
    page_header(
        "Claim Review Notice",
        "Plain-language explanation of the model's fraud-risk estimate for a selected claim.",
    )

    st.info(
        "This notice explains the model's fraud-risk estimate in plain language. "
        "It is a decision-support document, not a fraud finding, policy decision, "
        "or coverage determination. A human claims investigator must review this case."
    )

    if "_shap_top3" not in st.session_state:
        st.warning(
            "No claim selected. Go to Claim Analysis, select a claim, "
            "then visit Local Explainability — the notice will use SHAP + LIME from that session."
        )
        st.stop()

    notice_shap3  = st.session_state.get("_shap_top3", [])
    notice_lime3  = st.session_state.get("_lime_top3", [])
    notice_prob   = st.session_state.get("_risk_prob", 0.0)
    notice_cat    = st.session_state.get("_risk_cat", "Unknown")
    notice_id     = st.session_state.get("_claim_id", "UNKNOWN")

    # Factor tables
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("**Top 3 SHAP Factors** — primary ranking by |SHAP| value")
        if notice_shap3:
            st.dataframe(
                pd.DataFrame([{
                    "Feature":    _plain_english_factor(f["feature"]),
                    "SHAP Value": f"{f['impact']:+.4f}",
                    "Direction":  "Increases risk" if f["impact"] > 0 else "Decreases risk",
                } for f in notice_shap3]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("SHAP factors not available. Select a claim in Claim Analysis.")
    with col_n2:
        st.markdown("**Top 3 LIME Factors** — supplementary local surrogate")
        if notice_lime3:
            st.dataframe(
                pd.DataFrame([{
                    "Condition":  fd,
                    "LIME Value": f"{c:+.4f}",
                    "Direction":  "Increases risk" if c > 0 else "Decreases risk",
                } for fd, c in notice_lime3]),
                use_container_width=True, hide_index=True,
            )
            if notice_shap3:
                s_dir = notice_shap3[0]["impact"] > 0
                l_dir = notice_lime3[0][1] > 0
                if s_dir != l_dir:
                    st.warning(
                        "SHAP and LIME disagree on the direction of the top factor. "
                        "Human review is especially important."
                    )
                else:
                    st.success("SHAP and LIME agree on the primary factor direction.")
        else:
            st.info(
                "LIME factors not available. Visit Local Explainability to compute them."
            )

    # Notice preview
    st.markdown("---")
    st.markdown("**Notice Preview**")
    notice_text = generate_adverse_action_notice(
        claim_id=notice_id,
        risk_prob=notice_prob,
        risk_cat=notice_cat,
        shap_top3=notice_shap3,
        lime_top3=notice_lime3,
    )
    st.markdown(
        f'<div class="ix-notice">{notice_text}</div>',
        unsafe_allow_html=True,
    )

    # Download
    st.download_button(
        label="Download Notice as TXT",
        data=notice_text,
        file_name=f"claim_review_notice_{notice_id}.txt",
        mime="text/plain",
        key="dl_notice_txt",
    )
    st.caption(
        "This notice is generated from model output only. It is not a legally compliant "
        "adverse-action notice and must not be used in any real insurance decision workflow."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Compliance & Fairness
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "Compliance & Fairness":
    page_header(
        "Compliance & Fairness",
        "Feature interactions, potential proxy variables, and fairness analysis.",
    )

    st.info(
        "This section provides a model explainability and fairness audit based on the uploaded "
        "dataset. It does not constitute a formal regulatory compliance certification. "
        "All findings are observational — correlation is not causation."
    )

    cf_tab1, cf_tab2, cf_tab3, cf_tab4 = st.tabs([
        "Feature Interactions",
        "Proxy & Fairness Audit",
        "Dataset & Model Context",
        "Download Report",
    ])

    with cf_tab1:
        st.markdown("#### Feature Correlation Matrix")
        key_num = [c for c in [
            "total_claim_amount", "months_as_customer", "age", "policy_annual_premium",
            "umbrella_limit", "capital-gains", "capital-loss", "policy_age_days",
            "vehicle_claim_ratio", "injury_claim_ratio", "property_claim_ratio",
            "missing_doc_count", "bodily_injuries", "witnesses",
            "number_of_vehicles_involved", "policy_deductable",
        ] if c in X_aligned.columns]

        if len(key_num) >= 4:
            corr = X_aligned[key_num].corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                labels={"color": "Pearson r"},
            )
            fig_corr.update_traces(textfont_size=9)
            fig_corr.update_layout(
                title="Pearson Correlation Matrix — numeric features",
                coloraxis_colorbar_title="r",
                height=520,
            )
            plotly_defaults(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)

            pairs = [
                (c1, c2, float(corr.loc[c1, c2]))
                for i, c1 in enumerate(key_num)
                for j, c2 in enumerate(key_num) if j > i
            ]
            pairs_df = (
                pd.DataFrame(pairs, columns=["Feature A", "Feature B", "Pearson r"])
                .assign(abs_r=lambda d: d["Pearson r"].abs())
                .sort_values("abs_r", ascending=False)
                .drop(columns="abs_r")
                .head(10)
                .reset_index(drop=True)
            )
            st.markdown("**Strongest feature associations (top 10 by |r|)**")
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
            interaction_text = pairs_df.to_string(index=False)

            st.caption(
                "These are statistical associations, not causal relationships. "
                "A correlation between two features does not imply either causes fraud."
            )
        else:
            interaction_text = "Insufficient numeric features for correlation analysis."
            st.info("Not enough numeric features for correlation analysis.")

        st.markdown("#### Partial Dependence: Total Claim Amount vs Fraud Risk")
        if "total_claim_amount" in X_aligned.columns:
            tca_range = np.linspace(
                X_aligned["total_claim_amount"].quantile(0.05),
                X_aligned["total_claim_amount"].quantile(0.95),
                30,
            )
            bg = X_aligned.sample(min(100, len(X_aligned)), random_state=42)
            pdp_means = []
            for v in tca_range:
                bg2 = bg.copy()
                bg2["total_claim_amount"] = v
                pdp_means.append(model.predict_proba(bg2)[:, 1].mean())

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(
                x=tca_range, y=pdp_means,
                mode="lines",
                line=dict(color=ACCENT_DARK, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(155,203,69,0.1)",
                hovertemplate="Amount: $%{x:,.0f}<br>Avg Fraud Prob: %{y:.1%}<extra></extra>",
            ))
            fig_pdp.update_layout(
                title="Partial Dependence Plot — Total Claim Amount",
                xaxis_title="Total Claim Amount ($)",
                yaxis_title="Average Predicted Fraud Probability",
                yaxis_tickformat=".0%",
                showlegend=False,
            )
            plotly_defaults(fig_pdp)
            st.plotly_chart(fig_pdp, use_container_width=True)
            st.caption(
                "The PDP shows the model's average output as claim amount varies, "
                "holding other features at observed values. Statistical association — not causation."
            )
        else:
            st.info("total_claim_amount not in feature set.")

    with cf_tab2:
        st.markdown("#### Protected Attributes in Dataset")
        st.markdown("""
| Attribute | Type | Notes |
|---|---|---|
| `insured_sex` | Protected attribute | Gender — used directly by the model |
| `insured_education_level` | Potential proxy | May correlate with socioeconomic status |
| `insured_occupation` | Potential proxy | May correlate with socioeconomic class / ethnicity |
| `insured_hobbies` | Potential proxy | May correlate with lifestyle / demographic groups |
| `incident_state`, `incident_city` | Geographic proxy | May correlate with race / ethnicity |
        """)

        st.markdown("#### Mean Fraud Risk Score by Sex Category")
        sex_cols = [c for c in X_aligned.columns if c.startswith("insured_sex_")]
        if sex_cols and "risk_probability" in df_feat.columns:
            sex_data = {}
            for sc in sex_cols:
                mask = X_aligned[sc] == 1
                if mask.sum() > 0:
                    sex_data[sc.replace("insured_sex_", "")] = (
                        df_feat.loc[mask.values, "risk_probability"].mean()
                    )
            ref_mask = (X_aligned[sex_cols] == 0).all(axis=1)
            if ref_mask.sum() > 0:
                sex_data["(reference group)"] = df_feat.loc[ref_mask.values, "risk_probability"].mean()

            sex_df = pd.DataFrame(
                list(sex_data.items()), columns=["Sex Category", "Mean Fraud Risk Score"]
            )
            fig_sex = go.Figure(go.Bar(
                x=sex_df["Sex Category"],
                y=sex_df["Mean Fraud Risk Score"],
                marker_color=ACCENT,
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ))
            fig_sex.update_layout(
                yaxis_tickformat=".1%",
                yaxis_title="Mean Fraud Risk Score",
                showlegend=False,
            )
            plotly_defaults(fig_sex)
            st.plotly_chart(fig_sex, use_container_width=True)
            st.dataframe(sex_df, use_container_width=True, hide_index=True)

            if len(sex_df) >= 2:
                diff = sex_df["Mean Fraud Risk Score"].max() - sex_df["Mean Fraud Risk Score"].min()
                if diff > 0.05:
                    st.warning(
                        f"The mean fraud-risk score differs by {diff:.3f} across sex categories. "
                        "This may indicate differential model impact. "
                        "Further statistical testing and domain expert review is required."
                    )
                else:
                    st.success(
                        f"Mean fraud-risk scores are close across sex categories (max diff: {diff:.3f}). "
                        "No obvious differential impact detected, though full fairness analysis "
                        "requires ground-truth labels and formal statistical tests."
                    )
        else:
            st.info("insured_sex column not found or not one-hot encoded — sex audit not available.")

        st.markdown("#### Education Level Analysis")
        edu_cols = [c for c in X_aligned.columns if c.startswith("insured_education_level_")]
        if edu_cols and "risk_probability" in df_feat.columns:
            edu_data = {}
            for ec in edu_cols:
                mask = X_aligned[ec] == 1
                if mask.sum() > 0:
                    edu_data[ec.replace("insured_education_level_", "")] = (
                        df_feat.loc[mask.values, "risk_probability"].mean()
                    )
            ref_mask = (X_aligned[edu_cols] == 0).all(axis=1)
            if ref_mask.sum() > 0:
                edu_data["(reference)"] = df_feat.loc[ref_mask.values, "risk_probability"].mean()
            edu_df = pd.DataFrame(
                list(edu_data.items()), columns=["Education Level", "Mean Fraud Risk Score"]
            )
            st.dataframe(edu_df, use_container_width=True, hide_index=True)

        st.markdown("""
**Fairness analysis limitations:**
- This dataset does not contain race, ethnicity, or national origin attributes.
  Group-level fairness metrics (demographic parity, equalised odds) for these protected
  characteristics **cannot be computed** with the available data.
- Additional data collection and third-party auditing is required before deployment
  in a real insurance context.
- Proxy discrimination is difficult to detect statistically without ground-truth demographic
  labels — legal and domain expert review is essential.
        """)

        proxy_audit_text = """
Protected / proxy attributes in dataset:
  - insured_sex: directly used as a model feature (protected attribute)
  - insured_education_level: potential socioeconomic proxy
  - insured_occupation: potential socioeconomic/ethnicity proxy
  - insured_hobbies: lifestyle/demographic proxy
  - incident_state/city: geographic proxies for demographic groups

Findings:
  - The model uses insured_sex directly. Legal in some jurisdictions for
    insurance, prohibited in others. Legal review is required.
  - No race, ethnicity, or national origin fields exist — group fairness metrics
    cannot be computed for these protected classes.
  - Statistical difference in mean risk scores by sex was computed (see dashboard).
  - Education and occupation proxies warrant further investigation.

Recommendation:
  - Consult legal counsel on jurisdiction-specific rules for sex in insurance ML.
  - Conduct adversarial proxy audit using external demographic data.
  - Implement monitoring for outcome disparities by protected attributes.
  - Consider removing insured_sex if not legally permissible.
"""

    with cf_tab3:
        st.markdown("#### Dataset Context")
        col_dc1, col_dc2 = st.columns(2)
        with col_dc1:
            st.metric("Total Claims", f"{len(df_raw):,}")
            st.metric("Raw Features", df_raw.shape[1])
            st.metric("Model Features (encoded)", len(feature_columns))
        with col_dc2:
            if "fraud_reported" in df_raw.columns:
                vc = df_raw["fraud_reported"].value_counts()
                st.metric("Fraudulent (Y)", f"{vc.get('Y', 0)} ({vc.get('Y',0)/len(df_raw):.1%})")
                st.metric("Genuine (N)",    f"{vc.get('N', 0)} ({vc.get('N',0)/len(df_raw):.1%})")
            else:
                st.info("Upload labeled data to see class distribution.")

        st.markdown("#### Model Details")
        st.dataframe(
            pd.DataFrame({
                "Parameter":    ["Algorithm", "n_estimators", "max_depth", "class_weight",
                                 "Evaluation", "Explainability"],
                "Value":        ["Random Forest Classifier", "300", "8", "balanced",
                                 "5-fold CV + 20% held-out test set",
                                 "SHAP TreeExplainer + LIME LimeTabularExplainer"],
            }),
            use_container_width=True, hide_index=True,
        )

        if model_metrics_dict:
            st.markdown("#### Performance Metrics (uploaded labeled data)")
            st.dataframe(
                pd.DataFrame(list(model_metrics_dict.items()), columns=["Metric", "Value"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Accuracy is not the primary metric due to class imbalance. "
                "Precision, Recall, F1, and ROC-AUC are reported instead."
            )

    with cf_tab4:
        st.markdown("#### Download Compliance Report (HTML)")
        st.markdown(
            "The report includes: dataset context, model details, performance metrics, "
            "global SHAP summary, feature interactions, proxy audit, and limitations."
        )

        fraud_dist_dict = (
            df_raw["fraud_reported"].value_counts().to_dict()
            if "fraud_reported" in df_raw.columns
            else {"Y": "N/A", "N": "N/A"}
        )
        interaction_t = locals().get("interaction_text", "N/A")
        proxy_t       = locals().get("proxy_audit_text", proxy_audit_text if "proxy_audit_text" in dir() else "")

        shap_global_summary = (
            "Top 5 globally important features (mean |SHAP|):\n"
            + "\n".join([
                f"  {i+1}. {r['feature']} — {r['mean_abs_shap']:.4f}"
                for i, r in importance_df.head(5).iterrows()
            ])
        )
        g_img = globals().get("global_shap_img_b64", None)

        with st.spinner("Building compliance report..."):
            report_html = build_compliance_report_html(
                dataset_shape=df_raw.shape,
                fraud_dist=fraud_dist_dict,
                model_metrics=model_metrics_dict,
                shap_global_text=shap_global_summary,
                proxy_audit_text=proxy_audit_text,
                interaction_text=interaction_text if "interaction_text" in dir() else "N/A",
                global_shap_img_b64=g_img,
            )

        st.download_button(
            label="Download Compliance Report (HTML)",
            data=report_html,
            file_name=f"insurexplain_compliance_{datetime.date.today()}.html",
            mime="text/html",
            key="dl_compliance_html",
        )
        st.caption(
            "Open the downloaded HTML file in any browser. "
            "This report is for academic demonstration — not a regulatory compliance certificate."
        )

        # CSV download here too
        st.markdown("#### Download Scored Claims (CSV)")
        out_cols = [c for c in df_raw.columns if c in df_feat.columns] + [
            "risk_probability", "risk_category"
        ]
        csv_buf = io.StringIO()
        df_feat[out_cols].to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Scored Claims (CSV)",
            data=csv_buf.getvalue(),
            file_name="scored_claims.csv",
            mime="text/csv",
            key="dl_scored_csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: LIME vs SHAP Validation
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "LIME vs SHAP Validation":
    page_header(
        "LIME vs SHAP Validation",
        "Three representative claims compared: High, Medium, and Low predicted risk.",
    )

    st.markdown(
        "Both SHAP and LIME are computed from the actual model on actual claim records. "
        "LIME random seed = 42 for reproducibility. "
        "LIME expresses features as interval conditions (e.g. 'amount > 50,000') rather than "
        "exact feature names — so feature-name matching with SHAP is approximate."
    )

    # Select 3 representative records
    high_r = df_feat[df_feat["risk_category"] == "High"].sort_values("risk_probability", ascending=False)
    med_r  = df_feat[df_feat["risk_category"] == "Medium"].sort_values("risk_probability")
    low_r  = df_feat[df_feat["risk_category"] == "Low"].sort_values("risk_probability")

    val_indices, val_labels = [], []
    if len(high_r) > 0:
        val_indices.append(high_r.index[0]);          val_labels.append("High Risk")
    if len(med_r)  > 0:
        val_indices.append(med_r.index[len(med_r)//2]); val_labels.append("Medium Risk")
    if len(low_r)  > 0:
        val_indices.append(low_r.index[-1]);           val_labels.append("Low Risk")

    if len(val_indices) < 3:
        st.warning("Not enough records in each risk category for the full validation study.")
        st.stop()

    with st.spinner("Computing SHAP and LIME for 3 representative records..."):
        lime_val = build_lime_explainer(X_aligned.values, tuple(X_aligned.columns))
        val_results = []
        for idx, label in zip(val_indices, val_labels):
            pos    = df_feat.index.get_loc(idx)
            prob   = float(df_feat.loc[idx, "risk_probability"])
            cat    = str(df_feat.loc[idx, "risk_category"])

            if pos < N_SHAP:
                sv     = shap_vals[pos]
                sv_df2 = (
                    pd.DataFrame({"feature": X_aligned.columns, "shap": sv})
                    .sort_values("shap", key=abs, ascending=False)
                    .head(3)
                )
                shap3 = [(r["feature"], r["shap"]) for _, r in sv_df2.iterrows()]
            else:
                shap3 = []

            lime3 = get_lime_explanation(lime_val, X_aligned.values[pos])[:3]

            # Agreement
            if shap3 and lime3:
                s_dir = shap3[0][1] > 0
                l_dir = lime3[0][1]  > 0
                top_in_lime = any(shap3[0][0] in ld for ld, _ in lime3)
                agr = "Full" if (s_dir == l_dir and top_in_lime) else \
                      ("Direction" if s_dir == l_dir else "None")
            else:
                agr = "N/A"

            val_results.append({
                "label": label, "idx": idx, "prob": prob, "cat": cat,
                "shap3": shap3, "lime3": lime3, "agreement": agr,
            })

    # Summary comparison table
    summ_rows = []
    for vr in val_results:
        summ_rows.append({
            "Record":            vr["label"],
            "Fraud Probability": f"{vr['prob']:.1%}",
            "Risk Category":     vr["cat"],
            "SHAP Top Feature":  _plain_english_factor(vr["shap3"][0][0]) if vr["shap3"] else "N/A",
            "LIME Top Condition":vr["lime3"][0][0] if vr["lime3"] else "N/A",
            "Agreement":         vr["agreement"],
        })
    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame(summ_rows), use_container_width=True, hide_index=True)

    # Detail cards
    st.markdown("---")
    for vr in val_results:
        with st.expander(f"{vr['label']} — {vr['prob']:.1%} fraud probability", expanded=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability", f"{vr['prob']:.1%}")
            m2.metric("Risk Category",     vr["cat"])
            m3.metric("SHAP/LIME Agreement", vr["agreement"])

            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("**Top 3 SHAP Contributors**")
                if vr["shap3"]:
                    st.dataframe(
                        pd.DataFrame([{
                            "Feature":    _plain_english_factor(f),
                            "SHAP Value": f"{v:+.4f}",
                            "Direction":  "Up" if v > 0 else "Down",
                        } for f, v in vr["shap3"]]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("SHAP not computed for this row.")
            with vc2:
                st.markdown("**Top 3 LIME Contributors**")
                if vr["lime3"]:
                    st.dataframe(
                        pd.DataFrame([{
                            "Condition":  fd,
                            "LIME Value": f"{c:+.4f}",
                            "Direction":  "Up" if c > 0 else "Down",
                        } for fd, c in vr["lime3"]]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("LIME not available.")

            # Agreement commentary
            if vr["agreement"] == "None":
                st.warning(
                    f"Disagreement detected for {vr['label']}: SHAP and LIME indicate opposite "
                    "directions for the top factor. Possible causes: (1) LIME's local linear "
                    "surrogate may not capture the RF's non-linear boundary; (2) feature "
                    "dependencies are not well-captured by LIME's independent perturbation; "
                    "(3) the decision boundary near this point is complex. Neither method proves "
                    "causation. Human review is recommended."
                )
            elif vr["agreement"] == "Direction":
                st.info(
                    f"Partial agreement for {vr['label']}: SHAP and LIME agree on direction "
                    "for the top factor but may disagree on exact feature ranking. "
                    "LIME expresses conditions rather than raw feature names, "
                    "making direct ranking comparison approximate."
                )
            elif vr["agreement"] == "Full":
                st.success(
                    f"Good agreement for {vr['label']}: SHAP and LIME broadly agree on the top "
                    "contributing factor and its direction. This increases confidence in the "
                    "local explanation for this claim."
                )

    st.markdown("""
---
**Validation Notes:**
- LIME random state is fixed at 42 for reproducibility.
- LIME uses 500 perturbation samples — increasing this improves stability but increases compute time.
- LIME describes features as interval conditions — direct name matching with SHAP is approximate.
- Disagreements are expected to some degree: SHAP is exact for tree models; LIME is an approximation.
- Neither method proves causation.
- To regenerate: re-upload the claims CSV and re-visit this section.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: About
# ─────────────────────────────────────────────────────────────────────────────
elif active_section == "About":
    page_header(
        "About InsureXplain",
        "Academic micro-project — Trustworthy and Responsible AI.",
    )

    st.markdown("""
### What This Application Does

InsureXplain is an academic demonstration of interpretable machine learning applied to
auto-insurance claim fraud risk detection.

The application:
- Scores uploaded insurance claims with a trained Random Forest model
- Provides local explanations for individual claim predictions (SHAP + LIME)
- Provides global model explanations across the dataset (SHAP summary)
- Generates plain-language claim-review notices describing the model's estimate
- Produces a compliance and fairness audit report

### What This Application Does NOT Do

This application does **not**:
- Make final fraud determinations
- Deny claims or revoke coverage
- Set or change insurance premiums
- Constitute a legally compliant adverse-action notice
- Guarantee regulatory compliance
- Prove that the model is unbiased

All model outputs are **decision-support signals** that require human review before any action is taken.

### Model Details

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training data | 1,000 synthetic/semi-public auto-insurance claims |
| Prediction target | `fraud_reported` — binary (Y = fraud, N = genuine) |
| Features (raw) | 39 input variables |
| Features (encoded) | ~152 after engineering and one-hot encoding |
| Explainability | SHAP TreeExplainer + LIME LimeTabularExplainer |
| Class imbalance handling | `class_weight='balanced'` |

### Prediction vs Causation

A high SHAP value for a feature means that feature contributed to a higher model output
for this specific claim. It does **not** mean:
- The feature caused fraud
- The claimant committed fraud
- The claim should be denied

Statistical associations in historical data may reflect systemic patterns rather than
genuine causal mechanisms. Domain expert review is essential.

### Fraud Risk vs Fraud Finding

The model estimates a **statistical fraud-risk probability**. A high risk score means
the model identified patterns similar to historically fraudulent claims — it is a
screening signal, not a finding of fraud. Only a trained investigator can determine
whether a claim is actually fraudulent.

### Known Limitations

- Dataset is synthetic/semi-public and may not reflect real-world fraud patterns
- Race, ethnicity, and national origin attributes are absent — group fairness for
  these classes cannot be assessed
- `insured_sex` is used directly as a model feature — legal review is required before
  any deployment
- SHAP and LIME are explanations of the model's behaviour, not explanations of reality
- Model was trained on historical data — performance may degrade if claim patterns shift
- This is a college project — it should not be used for any real insurance decision

### Academic Context

**Course:** Trustworthy and Responsible AI

**Project:** InsureXplain — Interpretable Underwriting Engine

**Repository:** github.com/hardijain24-create/InsureXplain

**Model training:** See notebooks/04_modeling_explainability.ipynb

---
*Academic use only. Not for production deployment.*
    """)
