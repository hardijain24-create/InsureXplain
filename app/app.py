"""
InsureXplain: Interpretable Underwriting Engine – Streamlit App
================================================================
Deliverable 1: SHAP (local + global) + LIME local explanations, waterfall plot
Deliverable 2: Adverse Action / Claim-Review Notice generator
Deliverable 3: Global Regulatory Compliance Report (proxy audit, feature interactions)
Validation:    LIME vs SHAP on 3 representative records

Existing functionality preserved:
  - CSV upload, data quality overview, fraud-risk scoring
  - Model performance metrics (when ground-truth labels present)
  - SHAP bar chart for individual claims
  - AI-generated investigator summary (Gemini, optional)
  - CSV download of scored results
"""

import os
import io
import json
import datetime
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests

# Pure utility functions (no Streamlit dep — also used by tests)
from utils import (
    risk_category,
    plain_english_factor as _plain_english_factor,
    generate_adverse_action_notice,
    build_compliance_report_html,
)

# ---------------------------------------------------------------------------
# Page setup + visual theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InsureXplain – Claim Risk Detection",
    layout="wide",
    page_icon="🔍"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* App background */
.stApp { background-color: #F7F8FC; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #101C36;
}
section[data-testid="stSidebar"] * { color: #E6E9F5 !important; }
section[data-testid="stSidebar"] .stFileUploader label { color: #E6E9F5 !important; }

section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    color: #1B2A4A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileData"] * {
    color: #1B2A4A !important;
}
section[data-testid="stSidebar"] small { color: #6B7280 !important; }

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #1B2A4A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background-color: #EFF2FA !important;
    border: 1px solid #D6DCEF !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span {
    color: #1B2A4A !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button svg {
    fill: #1B2A4A !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] div[data-testid="stFileUploaderFile"] > div {
    background-color: #1B2A4A !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    color: #E6E9F5 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: #16233F !important;
    border: 1px dashed #3A4A70 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #E6E9F5 !important;
}
section[data-testid="stSidebar"] small { color: #9AA6C4 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #101C36 0%, #1B2A4A 60%, #24365F 100%);
    padding: 2.2rem 2.4rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.3rem;
    color: #FFFFFF;
    margin: 0 0 0.4rem 0;
}
.hero p {
    color: #AEB8D6;
    font-size: 1.02rem;
    margin: 0;
    max-width: 760px;
}
.hero .eyebrow {
    color: #6C8CFF;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Section headers */
h2 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: #1B2A4A !important;
    font-size: 1.5rem !important;
    margin-top: 1.8rem !important;
}
h3 { color: #1B2A4A !important; font-weight: 600 !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E7EAF3;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(27,42,74,0.06);
}
div[data-testid="stMetricLabel"] { color: #6B7280 !important; }
div[data-testid="stMetricValue"] { color: #1B2A4A !important; font-weight: 700 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #E7EAF3;
    border-radius: 10px;
    overflow: hidden;
}

/* Buttons */
.stDownloadButton button, .stButton button {
    background-color: #2E5BFF !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
}
.stDownloadButton button:hover, .stButton button:hover { background-color: #1E4BE0 !important; }

/* Info / success / warning boxes */
div[data-testid="stAlertContainer"] { border-radius: 10px; }

/* Risk badge pills */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.risk-high { background-color: #FDECEC; color: #C13B44; }
.risk-medium { background-color: #FFF3DC; color: #B7791F; }
.risk-low { background-color: #E7F6EE; color: #1E8A57; }

/* Notice card */
.notice-box {
    background: #fff;
    border: 1px solid #D6DCEF;
    border-left: 5px solid #2E5BFF;
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    font-size: 0.97rem;
    line-height: 1.7;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
    color: #1B2A4A;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="eyebrow">InsureXplain · Trustworthy & Responsible AI — Micro-Project</div>
    <h1>🔍 Insurance Claim Risk Detection</h1>
    <p>Upload a claims CSV to score fraud risk, get local SHAP + LIME explanations,
    generate an Adverse Action Notice, and download a Compliance Audit report.</p>
</div>
""", unsafe_allow_html=True)

RISK_BADGE_CLASS = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}


def risk_badge_html(category):
    cls = RISK_BADGE_CLASS.get(category, "risk-medium")
    return f'<span class="risk-badge {cls}">{category} risk</span>'


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Load model + metadata (cached so it only loads once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    categorical_columns = joblib.load(os.path.join(MODEL_DIR, "categorical_columns.pkl"))
    return model, feature_columns, categorical_columns


try:
    model, feature_columns, categorical_columns = load_model_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found in the `models/` folder. "
        "Run notebook 04_modeling_explainability.ipynb all the way to the end first — "
        "it saves fraud_model.pkl, feature_columns.pkl, and categorical_columns.pkl."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Feature engineering — mirrors notebook 03 exactly
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    if 'incident_date' in df.columns and 'policy_bind_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    if 'months_as_customer' in df.columns:
        df['customer_tenure_years'] = df['months_as_customer'] / 12
        if 'total_claim_amount' in df.columns:
            df['claim_amount_per_tenure_year'] = df['total_claim_amount'] / \
                df['customer_tenure_years'].replace(0, 0.1)

    if 'total_claim_amount' in df.columns:
        threshold = df['total_claim_amount'].quantile(0.90)
        df['is_high_value_claim'] = (df['total_claim_amount'] >= threshold).astype(int)

    for part, total_col in [('vehicle_claim', 'total_claim_amount'),
                             ('injury_claim', 'total_claim_amount'),
                             ('property_claim', 'total_claim_amount')]:
        if part in df.columns and total_col in df.columns:
            df[f'{part}_ratio'] = df[part] / df[total_col].replace(0, np.nan)

    if all(c in df.columns for c in ['police_report_available', 'witnesses', 'property_damage']):
        df['missing_doc_count'] = (
            (df['police_report_available'].isin(['NO', 'Unknown', '?'])).astype(int) +
            (df['witnesses'] == 0).astype(int) +
            (df['property_damage'].isin(['NO', 'Unknown', '?'])).astype(int)
        )

    if 'incident_hour_of_the_day' in df.columns:
        df['is_odd_hour_incident'] = df['incident_hour_of_the_day'].apply(
            lambda h: 1 if pd.notnull(h) and (h <= 5 or h >= 22) else 0
        )

    return df


def prepare_for_model(df_raw):
    """Clean, engineer, encode, and align columns to exactly match what the model expects."""
    df = df_raw.replace('?', np.nan)
    df_feat = engineer_features(df)

    drop_cols = ['fraud_reported', 'fraud_flag', 'incident_date', 'policy_bind_date',
                 'incident_location', 'policy_number', 'insured_zip', '_c39']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors='ignore')

    cat_cols_present = [c for c in categorical_columns if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols_present, drop_first=True)

    # Align to exact columns model was trained on
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)

    return df_feat, X_aligned


# ---------------------------------------------------------------------------
# LLM explanation (Gemini, optional)
# ---------------------------------------------------------------------------
def generate_ai_explanation(claim_summary: dict, top_factors: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""

    factors_text = "; ".join([f"{f['feature']} (impact: {f['impact']:+.3f})" for f in top_factors])

    prompt = (
        "You are assisting an insurance investigator. Based ONLY on the structured claim data "
        "and model output below, write a short (2-3 sentence) plain-English explanation of why "
        "this claim received its risk score. Do not decide whether it is fraud - only explain the "
        "signals. Do not invent any facts not given below.\n\n"
        f"Risk probability: {claim_summary['risk_probability']:.0%}\n"
        f"Risk category: {claim_summary['risk_category']}\n"
        f"Most influential factors (feature, SHAP impact): {factors_text}\n"
        f"Claim amount: {claim_summary.get('total_claim_amount', 'N/A')}\n"
        f"Incident severity: {claim_summary.get('incident_severity', 'N/A')}\n"
    )

    if not api_key:
        factor_names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"This claim was classified as **{claim_summary['risk_category']} risk** "
            f"({claim_summary['risk_probability']:.0%} probability) primarily due to: {factor_names}. "
            f"(Set the GEMINI_API_KEY environment variable to enable live AI-generated explanations "
            f"via Google Gemini — currently showing a rule-based summary.)"
        )

    try:
        last_error = None
        for model_name in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]:
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=15,
                )
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                last_error = f"{resp.status_code} for model '{model_name}': {resp.text[:200]}"
            except Exception as inner_e:
                last_error = str(inner_e)
        raise RuntimeError(last_error)
    except Exception as e:
        factor_names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"[AI service unavailable, showing rule-based summary — {e}] "
            f"This claim was classified as {claim_summary['risk_category']} risk "
            f"({claim_summary['risk_probability']:.0%} probability) primarily due to: {factor_names}."
        )


# ---------------------------------------------------------------------------
# LIME helper — genuine local explanation using the actual model + preprocessing
# ---------------------------------------------------------------------------
@st.cache_resource
def build_lime_explainer(_X_aligned_values, _feature_names):
    """
    Build a LIME tabular explainer on the aligned, encoded feature matrix.
    _X_aligned_values: numpy array used as the background distribution.
    _feature_names: list of feature names.
    Random state is fixed for reproducibility.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            training_data=_X_aligned_values,
            feature_names=list(_feature_names),
            class_names=["Genuine", "Fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        return explainer
    except ImportError:
        return None


def get_lime_explanation(lime_explainer, model, row_array, num_features=10, num_samples=500):
    """
    Generate LIME local explanation for a single row.
    Returns a sorted list of (feature_name, contribution) tuples for class=1 (Fraud).
    """
    try:
        exp = lime_explainer.explain_instance(
            data_row=row_array,
            predict_fn=model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )
        # exp.as_list(label=1) gives (feature_description, contribution) for Fraud class
        raw = exp.as_list(label=1)
        return raw
    except Exception as e:
        return []


# generate_adverse_action_notice and _plain_english_factor imported from utils.py
# build_compliance_report_html imported from utils.py


# ---------------------------------------------------------------------------
# Sidebar: upload
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Claims Data")
uploaded_file = st.sidebar.file_uploader("Upload a claims CSV", type=["csv"])

# Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### 📑 Sections")
st.sidebar.markdown("""
- 📋 Data Quality  
- 🎯 Scoring  
- 📊 Model Performance  
- 🔬 Individual Explanation (SHAP + LIME)  
- 🌍 Global Explainability  
- 📄 Adverse Action Notice  
- 🏛️ Compliance Audit & Report  
- 🔁 LIME vs SHAP Validation  
- ⬇️ Download Results
""")

if uploaded_file is None:
    st.info("👈 Upload a CSV to get started. You can use `data/insurance_claims.csv` "
            "(the original dataset) to try the app out.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ---------------------------------------------------------------------------
# Data quality section
# ---------------------------------------------------------------------------
st.header("📋 Data Quality Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total claims", len(df_raw))
col2.metric("Columns", df_raw.shape[1])
missing_pct = (df_raw.replace('?', np.nan).isnull().sum().sum() /
               (df_raw.shape[0] * df_raw.shape[1]) * 100)
col3.metric("Missing values", f"{missing_pct:.1f}%")
col4.metric("Duplicate rows", int(df_raw.duplicated().sum()))

with st.expander("View raw data sample"):
    st.dataframe(df_raw.head(20))

# ---------------------------------------------------------------------------
# Score all claims
# ---------------------------------------------------------------------------
st.header("🎯 Scoring Claims")

with st.spinner("Scoring claims and computing SHAP values (may take ~30s for 1000 rows)..."):
    df_feat, X_aligned = prepare_for_model(df_raw)
    probs = model.predict_proba(X_aligned)[:, 1]
    df_feat['risk_probability'] = probs
    df_feat['risk_category'] = df_feat['risk_probability'].apply(risk_category)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_aligned)
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_vals = raw_shap[:, :, 1]
    else:
        shap_vals = raw_shap

    # Build SHAP Explanation object for waterfall plot
    try:
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[1]
        else:
            base_value = expected_value
    except Exception:
        base_value = 0.0

st.success(f"✅ Scored {len(df_feat)} claims.")

# ---------------------------------------------------------------------------
# Model performance (only if ground-truth labels are present)
# ---------------------------------------------------------------------------
model_metrics_dict = {}
if 'fraud_reported' in df_raw.columns:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y_true = df_raw['fraud_reported'].map({'Y': 1, 'N': 0})
    y_pred = (df_feat['risk_probability'] >= 0.5).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, df_feat['risk_probability'])
    model_metrics_dict = {
        "Precision": f"{prec:.3f}",
        "Recall": f"{rec:.3f}",
        "F1 Score": f"{f1:.3f}",
        "ROC-AUC": f"{auc:.3f}",
    }

    st.header("📊 Model Performance (ground-truth labels found in upload)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Precision", f"{prec:.2f}")
    p2.metric("Recall", f"{rec:.2f}")
    p3.metric("F1 Score", f"{f1:.2f}")
    p4.metric("ROC-AUC", f"{auc:.2f}")

# ---------------------------------------------------------------------------
# Filter high-risk claims
# ---------------------------------------------------------------------------
st.header("🚩 Review Claims")
risk_filter = st.multiselect(
    "Filter by risk category", options=["High", "Medium", "Low"], default=["High", "Medium"]
)
filtered = df_feat[df_feat['risk_category'].isin(risk_filter)].sort_values(
    'risk_probability', ascending=False
)

display_cols = [c for c in ['risk_probability', 'risk_category', 'total_claim_amount',
                             'incident_severity', 'incident_type'] if c in filtered.columns]


def _style_risk(val):
    colors = {"High": "#C13B44", "Medium": "#B7791F", "Low": "#1E8A57"}
    return f"color: {colors.get(val, '#1B2A4A')}; font-weight: 700;"


styled_table = filtered[display_cols].style.format({'risk_probability': '{:.1%}'})
if 'risk_category' in display_cols:
    styled_table = styled_table.map(_style_risk, subset=['risk_category'])

st.dataframe(styled_table, use_container_width=True)

# ---------------------------------------------------------------------------
# Individual Claim Explanation: SHAP + LIME  (Deliverable 1)
# ---------------------------------------------------------------------------
st.header("🔬 Individual Claim Explanation: SHAP vs LIME")
st.caption(
    "**Local analysis**: explains one specific claim — why did the model give it this score? "
    "SHAP uses exact Shapley values from the actual tree model. "
    "LIME creates a local linear approximation by perturbing the input."
)

if len(filtered) == 0:
    st.warning("No claims match the current filter.")
else:
    selected_idx = st.selectbox(
        "Select a claim (shown by row number)",
        options=filtered.index.tolist(),
        format_func=lambda i: f"Row {i} — {df_feat.loc[i, 'risk_category']} risk "
                               f"({df_feat.loc[i, 'risk_probability']:.0%})"
    )

    row = df_feat.loc[selected_idx]
    position_in_full = df_feat.index.get_loc(selected_idx)

    # Claim identifier (use policy_number if available)
    if 'policy_number' in df_raw.columns:
        claim_id = str(df_raw.loc[selected_idx, 'policy_number'])
    else:
        claim_id = f"CLAIM-{selected_idx:04d}"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Fraud Risk Probability", f"{row['risk_probability']:.1%}")
        st.markdown(
            f'<div style="margin: -0.3rem 0 1.1rem 0;">{risk_badge_html(row["risk_category"])}</div>',
            unsafe_allow_html=True
        )
        if 'total_claim_amount' in row:
            st.metric("Claim Amount", f"${row['total_claim_amount']:,.0f}")
        st.metric("Claim ID", claim_id)

    with c2:
        st.subheader("Key claim attributes")
        attr_cols = ['incident_severity', 'incident_type', 'incident_hour_of_the_day',
                     'bodily_injuries', 'witnesses', 'police_report_available',
                     'property_damage', 'number_of_vehicles_involved']
        available_attrs = {c: df_raw.loc[selected_idx, c]
                           for c in attr_cols if c in df_raw.columns}
        attr_df = pd.DataFrame(list(available_attrs.items()), columns=['Attribute', 'Value'])
        st.dataframe(attr_df, use_container_width=True, hide_index=True)

    # SHAP values for this row
    claim_shap = shap_vals[position_in_full]
    contrib_df = pd.DataFrame({
        'feature': X_aligned.columns,
        'impact': claim_shap
    }).sort_values('impact', key=abs, ascending=False)

    shap_top_n = contrib_df.head(10)
    shap_top3 = contrib_df.head(3).to_dict('records')

    # LIME explanation
    with st.spinner("Computing LIME explanation (~10–20 seconds)..."):
        lime_exp = build_lime_explainer(X_aligned.values, list(X_aligned.columns))
        if lime_exp is not None:
            lime_raw = get_lime_explanation(
                lime_exp, model,
                X_aligned.values[position_in_full],
                num_features=10, num_samples=500
            )
            lime_top3 = lime_raw[:3] if lime_raw else []
        else:
            lime_raw = []
            lime_top3 = []
            st.warning("LIME library not available. Install it via: `pip install lime`")

    # ── Side-by-side SHAP bar + LIME bar ──
    col_shap, col_lime = st.columns(2)

    with col_shap:
        st.subheader("SHAP — Local Explanation")
        st.caption("Exact Shapley values from the Random Forest model")
        fig_shap, ax_shap = plt.subplots(figsize=(5.5, 4.5))
        colors_shap = ['#C44E52' if v > 0 else '#4C72B0' for v in shap_top_n['impact']]
        ax_shap.barh(shap_top_n['feature'], shap_top_n['impact'], color=colors_shap)
        ax_shap.set_xlabel("SHAP impact (red = increases fraud risk)")
        ax_shap.set_title("Top-10 SHAP Contributions", fontsize=11)
        ax_shap.invert_yaxis()
        ax_shap.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    with col_lime:
        st.subheader("LIME — Local Explanation")
        st.caption("Local linear surrogate approximation (500 perturbation samples)")
        if lime_raw:
            lime_df = pd.DataFrame(lime_raw, columns=['feature_desc', 'contribution'])
            fig_lime, ax_lime = plt.subplots(figsize=(5.5, 4.5))
            colors_lime = ['#C44E52' if v > 0 else '#4C72B0' for v in lime_df['contribution']]
            ax_lime.barh(lime_df['feature_desc'], lime_df['contribution'], color=colors_lime)
            ax_lime.set_xlabel("LIME contribution (red = increases fraud risk)")
            ax_lime.set_title("Top-10 LIME Contributions", fontsize=11)
            ax_lime.invert_yaxis()
            ax_lime.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig_lime)
            plt.close(fig_lime)
        else:
            st.info("LIME explanation not available for this claim.")

    # ── SHAP Waterfall Plot ──
    st.subheader("📊 SHAP Waterfall Plot")
    st.caption(
        "Shows how each feature's SHAP value moves the prediction from the base value "
        f"(model average: {base_value:.3f}) to the final predicted probability."
    )
    try:
        shap_explanation = shap.Explanation(
            values=claim_shap,
            base_values=base_value,
            data=X_aligned.values[position_in_full],
            feature_names=list(X_aligned.columns)
        )
        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.close(fig_wf)
    except Exception as e:
        st.warning(f"Waterfall plot could not be rendered: {e}")
        # Fallback: manual waterfall-style bar
        contrib_sorted = contrib_df.head(12).sort_values('impact')
        fig_fallback, ax_fallback = plt.subplots(figsize=(8, 5))
        ax_fallback.barh(
            contrib_sorted['feature'],
            contrib_sorted['impact'],
            color=['#C44E52' if v > 0 else '#4C72B0' for v in contrib_sorted['impact']]
        )
        ax_fallback.set_xlabel(f"SHAP value (base = {base_value:.3f})")
        ax_fallback.set_title("SHAP Contributions (top 12 by magnitude)")
        ax_fallback.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_fallback)
        plt.close(fig_fallback)

    # ── AI-generated investigator summary ──
    st.subheader("🤖 AI-generated investigator summary")
    top_factors = contrib_df.head(6).to_dict('records')
    claim_summary = {
        'risk_probability': row['risk_probability'],
        'risk_category': row['risk_category'],
        'total_claim_amount': row.get('total_claim_amount', 'N/A'),
        'incident_severity': row.get('incident_severity', 'N/A'),
    }
    with st.spinner("Generating explanation..."):
        explanation = generate_ai_explanation(claim_summary, top_factors)
    st.info(explanation)
    st.caption(
        "⚠️ This explanation is AI-generated from model output only. "
        "It is NOT a fraud determination — a human investigator makes that call."
    )

    # Store for later sections
    st.session_state['_selected_claim_id'] = claim_id
    st.session_state['_shap_top3'] = shap_top3
    st.session_state['_lime_top3'] = lime_top3
    st.session_state['_risk_prob'] = row['risk_probability']
    st.session_state['_risk_cat'] = row['risk_category']

# ---------------------------------------------------------------------------
# Global Explainability — SHAP Summary Plot (Deliverable 1 + 3A)
# ---------------------------------------------------------------------------
st.header("🌍 Global Explainability — SHAP Summary")
st.caption(
    "**Global analysis**: shows overall model behaviour across the entire uploaded dataset, "
    "not any single claim. Reveals which features matter most on average and in which direction."
)

with st.expander("📈 Global SHAP Summary Plot (beeswarm)", expanded=True):
    try:
        n_bg = min(200, len(X_aligned))
        fig_summary, ax_summary = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals[:n_bg],
            X_aligned.iloc[:n_bg],
            max_display=15,
            show=False,
            plot_size=None
        )
        plt.title("Global SHAP Summary (top 15 features)", fontsize=13, pad=12)
        plt.tight_layout()

        # Capture as bytes for compliance report
        buf_summary = io.BytesIO()
        plt.savefig(buf_summary, format='png', dpi=120, bbox_inches='tight')
        buf_summary.seek(0)
        import base64
        global_shap_img_b64 = base64.b64encode(buf_summary.read()).decode()

        st.pyplot(fig_summary)
        plt.close(fig_summary)
    except Exception as e:
        st.warning(f"Could not render SHAP summary plot: {e}")
        global_shap_img_b64 = None

with st.expander("📊 Global SHAP Feature Importance (bar chart)"):
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_aligned.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).head(20)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    ax_imp.barh(importance_df['feature'][::-1],
                importance_df['mean_abs_shap'][::-1],
                color='#2E5BFF')
    ax_imp.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax_imp.set_title("Top 20 Features by Global SHAP Importance")
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)

    st.caption("Features ranked by mean absolute SHAP value across all uploaded claims.")
    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True, hide_index=True)

# Global SHAP text summary for compliance report
top5_features = importance_df.head(5)['feature'].tolist()
shap_global_text = (
    f"Top 5 globally important features (by mean |SHAP|):\n"
    + "\n".join([
        f"  {i+1}. {row['feature']} — mean |SHAP|: {row['mean_abs_shap']:.4f}"
        for i, (_, row) in enumerate(importance_df.head(5).iterrows())
    ])
    + "\n\nInterpretation:\n"
    + "- High SHAP magnitude = strong average influence on the fraud prediction.\n"
    + "- The beeswarm plot (above) shows BOTH the magnitude and direction of each feature's\n"
    + "  influence: red dots = high feature value, blue dots = low feature value.\n"
    + "- Features with both high positive and negative SHAP values have non-linear effects.\n"
    + "\nPotentially concerning features (possible demographic proxies):\n"
    + "  - insured_sex_* : gender is a protected attribute in many jurisdictions.\n"
    + "  - insured_education_level_* : may correlate with socioeconomic status.\n"
    + "  - insured_occupation_* : may correlate with income/class.\n"
    + "  - insured_hobbies_* : may correlate with lifestyle/demographic factors.\n"
    + "  - incident_city_*, incident_state_* : geographic features may proxy for ethnicity.\n"
    + "\nThese features warrant further proxy analysis (see Section 7)."
)

# ---------------------------------------------------------------------------
# Adverse Action Notice Generator  (Deliverable 2)
# ---------------------------------------------------------------------------
st.header("📄 Adverse Action Notice / Decision Explanation")
st.info(
    "ℹ️ This section generates a customer-facing **Claim Review Notice** explaining the model's "
    "risk estimate for the claim selected in the Individual Explanation section above. "
    "It is a decision-support notice — not a fraud finding or a policy/coverage decision."
)

if 'fraud_reported' not in df_raw.columns or True:
    st.caption(
        "**Target clarification**: The underlying model detects fraud risk in auto-insurance claims "
        "(`fraud_reported`). This notice describes that fraud-risk estimate only — it does not "
        "represent an underwriting rejection, coverage denial, or premium change."
    )

if '_shap_top3' not in st.session_state:
    st.warning("⬆️ First select a claim in the Individual Explanation section above.")
else:
    notice_shap3 = st.session_state['_shap_top3']
    notice_lime3 = st.session_state['_lime_top3']
    notice_prob = st.session_state['_risk_prob']
    notice_cat = st.session_state['_risk_cat']
    notice_id = st.session_state['_selected_claim_id']

    st.subheader(f"Notice for Claim: {notice_id}")

    # Display SHAP top-3 in a readable table
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown("**Top 3 SHAP Factors** (primary ranking)")
        shap_display = pd.DataFrame([{
            'Feature': _plain_english_factor(f['feature']),
            'Direction': '⬆️ Increases risk' if f['impact'] > 0 else '⬇️ Decreases risk',
            'SHAP Value': f"{f['impact']:+.4f}"
        } for f in notice_shap3])
        st.dataframe(shap_display, use_container_width=True, hide_index=True)

    with col_n2:
        st.markdown("**Top 3 LIME Factors** (supplementary)")
        if notice_lime3:
            lime_display = pd.DataFrame([{
                'Feature': fd,
                'Direction': '⬆️ Increases risk' if c > 0 else '⬇️ Decreases risk',
                'LIME Value': f"{c:+.4f}"
            } for fd, c in notice_lime3])
            st.dataframe(lime_display, use_container_width=True, hide_index=True)

            # Check agreement
            if notice_shap3 and notice_lime3:
                shap_dir = notice_shap3[0]['impact'] > 0
                lime_dir = notice_lime3[0][1] > 0
                if shap_dir != lime_dir:
                    st.warning(
                        "⚠️ SHAP and LIME disagree on the direction of the top factor. "
                        "This may reflect LIME's local linear approximation differing from "
                        "the tree model's non-linear decision boundaries. "
                        "Human review is especially recommended."
                    )
                else:
                    st.success("✅ SHAP and LIME agree on the primary factor's direction.")
        else:
            st.info("LIME explanation not available.")

    with st.spinner("Generating notice..."):
        notice_text = generate_adverse_action_notice(
            claim_id=notice_id,
            risk_prob=notice_prob,
            risk_cat=notice_cat,
            shap_top3=notice_shap3,
            lime_top3=notice_lime3,
        )

    st.markdown(f'<div class="notice-box">{notice_text}</div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Notice as TXT",
        data=notice_text,
        file_name=f"claim_review_notice_{notice_id}.txt",
        mime="text/plain",
        key="download_notice_txt"
    )

# ---------------------------------------------------------------------------
# Compliance Audit & Report (Deliverable 3)
# ---------------------------------------------------------------------------
st.header("🏛️ Compliance Audit & Regulatory Report")

# --- 3B: Feature Interactions ---
st.subheader("B. Feature Interactions")
with st.expander("Feature Correlation Heatmap (numeric features)", expanded=False):
    numeric_cols = X_aligned.select_dtypes(include=[np.number]).columns.tolist()
    # Focus on non-dummy numeric columns from original dataset
    key_numeric = [c for c in [
        'total_claim_amount', 'months_as_customer', 'age', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss', 'policy_age_days',
        'vehicle_claim_ratio', 'injury_claim_ratio', 'property_claim_ratio',
        'missing_doc_count', 'bodily_injuries', 'witnesses',
        'number_of_vehicles_involved', 'policy_deductable'
    ] if c in X_aligned.columns]

    if len(key_numeric) >= 4:
        corr_matrix = X_aligned[key_numeric].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        import matplotlib.colors as mcolors
        im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_corr.set_xticks(range(len(key_numeric)))
        ax_corr.set_yticks(range(len(key_numeric)))
        ax_corr.set_xticklabels([c.replace('_', '\n') for c in key_numeric], fontsize=7, rotation=45, ha='right')
        ax_corr.set_yticklabels([c.replace('_', ' ') for c in key_numeric], fontsize=8)
        plt.colorbar(im, ax=ax_corr, shrink=0.8)
        ax_corr.set_title("Feature Correlation Matrix (Pearson r)", fontsize=12)
        # Annotate cells
        for i in range(len(key_numeric)):
            for j in range(len(key_numeric)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.3:
                    ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center',
                                 fontsize=6, color='white' if abs(val) > 0.6 else 'black')
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

        # Top correlated pairs
        pairs = []
        for i, c1 in enumerate(key_numeric):
            for j, c2 in enumerate(key_numeric):
                if j > i:
                    pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
        pairs_df = pd.DataFrame(pairs, columns=['Feature A', 'Feature B', 'Pearson r'])
        pairs_df = pairs_df.reindex(pairs_df['Pearson r'].abs().sort_values(ascending=False).index)
        st.markdown("**Strongest feature associations** (top 10 by |r|):")
        st.dataframe(pairs_df.head(10).reset_index(drop=True), use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ These are statistical associations, not causal relationships. "
            "A high correlation between two features does not mean one causes the other, "
            "nor that either causes fraud."
        )
        interaction_text = pairs_df.head(10).to_string(index=False)
    else:
        st.info("Not enough numeric features for correlation analysis.")
        interaction_text = "Insufficient numeric features for correlation analysis."

with st.expander("Partial Dependence: Total Claim Amount vs Fraud Risk", expanded=False):
    if 'total_claim_amount' in X_aligned.columns:
        tca_idx = list(X_aligned.columns).index('total_claim_amount')
        tca_range = np.linspace(
            X_aligned['total_claim_amount'].quantile(0.05),
            X_aligned['total_claim_amount'].quantile(0.95),
            30
        )
        pdp_means = []
        bg_sample = X_aligned.sample(min(100, len(X_aligned)), random_state=42)
        for v in tca_range:
            bg_mod = bg_sample.copy()
            bg_mod['total_claim_amount'] = v
            pdp_means.append(model.predict_proba(bg_mod)[:, 1].mean())

        fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
        ax_pdp.plot(tca_range, pdp_means, color='#2E5BFF', linewidth=2)
        ax_pdp.fill_between(tca_range, pdp_means, alpha=0.1, color='#2E5BFF')
        ax_pdp.set_xlabel("Total Claim Amount ($)")
        ax_pdp.set_ylabel("Average Predicted Fraud Probability")
        ax_pdp.set_title("Partial Dependence Plot: Total Claim Amount")
        ax_pdp.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.tight_layout()
        st.pyplot(fig_pdp)
        plt.close(fig_pdp)
        st.caption(
            "The PDP shows how the model's average fraud probability changes "
            "as claim amount varies, holding all other features at their observed values. "
            "This is an association — not causation."
        )
    else:
        st.info("total_claim_amount not found in feature set.")

# --- 3C: Proxy & Fairness Audit ---
st.subheader("C. Discrimination and Proxy Audit")

with st.expander("Proxy Audit Results", expanded=True):
    st.markdown("""
**Protected attributes in dataset:**
- `insured_sex` — directly a protected attribute (gender) under insurance anti-discrimination law
- `insured_education_level` — may correlate with socioeconomic status
- `insured_occupation` — may correlate with socioeconomic class and indirectly with ethnicity
- `insured_hobbies` — may correlate with lifestyle/demographic groups
- `incident_state`, `incident_city` — geographic proxies that may correlate with race/ethnicity

**What the model includes:**
All of the above are included as categorical features (one-hot encoded). 
This means the model _does_ use gender (`insured_sex`) directly.
""")

    # Correlation between insured_sex dummy and outcome
    sex_cols = [c for c in X_aligned.columns if c.startswith('insured_sex_')]
    if sex_cols and 'risk_probability' in df_feat.columns:
        st.markdown("**Fraud-risk probability by sex category:**")
        sex_data = {}
        for sc in sex_cols:
            mask = X_aligned[sc] == 1
            if mask.sum() > 0:
                sex_data[sc.replace('insured_sex_', '')] = df_feat.loc[mask.values, 'risk_probability'].mean()
        # baseline (reference group, drop_first=True means one category is implicit)
        mask_ref = (X_aligned[sex_cols] == 0).all(axis=1)
        if mask_ref.sum() > 0:
            sex_data['(reference group)'] = df_feat.loc[mask_ref.values, 'risk_probability'].mean()

        sex_df = pd.DataFrame(list(sex_data.items()), columns=['Sex Category', 'Mean Fraud Risk Score'])
        st.dataframe(sex_df, use_container_width=True, hide_index=True)

        if len(sex_df) >= 2:
            diff = sex_df['Mean Fraud Risk Score'].max() - sex_df['Mean Fraud Risk Score'].min()
            if diff > 0.05:
                st.warning(
                    f"⚠️ The mean fraud-risk score differs by {diff:.3f} across sex categories. "
                    "This may indicate the model treats different groups differently. "
                    "Further statistical testing and domain expert review is required. "
                    "Correlation ≠ discrimination — but this warrants investigation."
                )
            else:
                st.success(
                    f"✅ Mean fraud-risk scores are close across sex categories (max diff: {diff:.3f}). "
                    "No obvious differential impact detected, but full fairness analysis "
                    "requires ground-truth labels and statistical tests."
                )
    else:
        st.info("insured_sex column not found or not one-hot encoded — cannot perform sex-based audit.")

    # Education level analysis
    edu_cols = [c for c in X_aligned.columns if c.startswith('insured_education_level_')]
    if edu_cols and 'risk_probability' in df_feat.columns:
        st.markdown("**Mean fraud-risk score by education level:**")
        edu_data = {}
        for ec in edu_cols:
            mask = X_aligned[ec] == 1
            if mask.sum() > 0:
                edu_data[ec.replace('insured_education_level_', '')] = \
                    df_feat.loc[mask.values, 'risk_probability'].mean()
        mask_ref = (X_aligned[edu_cols] == 0).all(axis=1)
        if mask_ref.sum() > 0:
            edu_data['(reference)'] = df_feat.loc[mask_ref.values, 'risk_probability'].mean()
        edu_df = pd.DataFrame(list(edu_data.items()),
                               columns=['Education Level', 'Mean Fraud Risk Score'])
        st.dataframe(edu_df, use_container_width=True, hide_index=True)

    st.markdown("""
**Fairness analysis limitations:**
- This dataset does not contain race, ethnicity, or national origin attributes.
  Group-level fairness metrics (demographic parity, equalised odds) for these protected 
  characteristics **cannot be computed** from the available data.
- Additional data collection and third-party auditing would be required before any 
  deployment in a real insurance context.
- Proxy discrimination is difficult to detect statistically without ground-truth 
  demographic labels; legal and domain expert review is essential.
""")

proxy_audit_text = """
Protected / proxy attributes in dataset:
  - insured_sex: directly used as a model feature (protected attribute)
  - insured_education_level: potential socioeconomic proxy
  - insured_occupation: potential socioeconomic/ethnicity proxy
  - insured_hobbies: lifestyle/demographic proxy
  - incident_state/city: geographic proxies for demographic groups

Findings:
  - The model uses insured_sex directly. This is legal in some jurisdictions for
    insurance but prohibited in others. Legal review is required.
  - No race, ethnicity, or national origin fields exist — group fairness metrics
    cannot be computed for these protected classes.
  - Statistical difference in mean risk scores by sex was computed (see app dashboard).
  - Education and occupation proxies warrant further investigation.

Recommendation:
  - Consult legal counsel on jurisdiction-specific rules for use of sex in insurance ML.
  - Conduct an adversarial proxy audit using external demographic data.
  - Implement monitoring for outcome disparities by available protected attributes.
  - Consider removing or regularising insured_sex if not legally permissible.
"""

# --- Compliance Report Download ---
st.subheader("D. Download Compliance Report")

fraud_dist_dict = {}
if 'fraud_reported' in df_raw.columns:
    fraud_dist_dict = df_raw['fraud_reported'].value_counts().to_dict()
else:
    fraud_dist_dict = {'Y': 'N/A', 'N': 'N/A'}

with st.spinner("Building compliance report..."):
    report_html = build_compliance_report_html(
        dataset_shape=df_raw.shape,
        fraud_dist=fraud_dist_dict,
        model_metrics=model_metrics_dict,
        shap_global_text=shap_global_text,
        proxy_audit_text=proxy_audit_text,
        interaction_text=interaction_text if 'interaction_text' in locals() else "N/A",
        global_shap_img_b64=global_shap_img_b64 if 'global_shap_img_b64' in locals() else None,
    )

st.download_button(
    label="⬇️ Download Compliance Report (HTML)",
    data=report_html,
    file_name=f"insurexplain_compliance_report_{datetime.date.today()}.html",
    mime="text/html",
    key="download_compliance_html"
)
st.caption("Open the downloaded HTML file in any browser to view the full formatted report.")

# ---------------------------------------------------------------------------
# LIME vs SHAP Validation Study (3 representative records)
# ---------------------------------------------------------------------------
st.header("🔁 LIME vs SHAP Validation Study")
st.caption(
    "Three representative records are evaluated: one High-risk, one Medium-risk, one Low-risk claim. "
    "For each, we compare top-3 SHAP contributors vs top-3 LIME contributors "
    "and discuss any material disagreements. Random seed = 42 for reproducibility."
)

with st.expander("View Validation Study (3 representative records)", expanded=False):
    # Select 3 representative records: highest high, median medium, lowest low
    high_risk = df_feat[df_feat['risk_category'] == 'High'].sort_values(
        'risk_probability', ascending=False)
    med_risk = df_feat[df_feat['risk_category'] == 'Medium'].sort_values('risk_probability')
    low_risk = df_feat[df_feat['risk_category'] == 'Low'].sort_values('risk_probability')

    val_indices = []
    val_labels = []
    if len(high_risk) > 0:
        val_indices.append(high_risk.index[0])
        val_labels.append("High Risk (highest probability)")
    if len(med_risk) > 0:
        mid = len(med_risk) // 2
        val_indices.append(med_risk.index[mid])
        val_labels.append("Medium Risk (median probability)")
    if len(low_risk) > 0:
        val_indices.append(low_risk.index[-1])
        val_labels.append("Low Risk (lowest probability)")

    if len(val_indices) < 3:
        st.info("Not enough records in each risk category for the full validation study.")
    else:
        lime_val_exp = build_lime_explainer(X_aligned.values, list(X_aligned.columns))
        validation_results = []

        for idx, label in zip(val_indices, val_labels):
            pos = df_feat.index.get_loc(idx)
            prob = df_feat.loc[idx, 'risk_probability']
            cat = df_feat.loc[idx, 'risk_category']

            # SHAP
            claim_sv = shap_vals[pos]
            sv_df = pd.DataFrame({'feature': X_aligned.columns, 'shap': claim_sv})
            sv_df = sv_df.reindex(sv_df['shap'].abs().sort_values(ascending=False).index).head(3)
            shap3 = [(r['feature'], r['shap']) for _, r in sv_df.iterrows()]

            # LIME
            if lime_val_exp is not None:
                lime3_raw = get_lime_explanation(
                    lime_val_exp, model,
                    X_aligned.values[pos],
                    num_features=10, num_samples=500
                )[:3]
            else:
                lime3_raw = []

            # Agreement analysis
            if shap3 and lime3_raw:
                shap_top_feat = shap3[0][0]
                lime_top_desc = lime3_raw[0][0]
                shap_dir = shap3[0][1] > 0
                lime_dir = lime3_raw[0][1] > 0
                dir_agree = shap_dir == lime_dir
                # Check if top feature names overlap (LIME uses conditions like "feat > X")
                top_feat_in_lime = any(shap_top_feat in ld for ld, _ in lime3_raw)
                agreement = "Full" if (dir_agree and top_feat_in_lime) else \
                            ("Direction" if dir_agree else "None")
            else:
                agreement = "N/A"

            validation_results.append({
                'label': label,
                'idx': idx,
                'prob': prob,
                'cat': cat,
                'shap3': shap3,
                'lime3': lime3_raw,
                'agreement': agreement
            })

        # Display
        for vr in validation_results:
            st.markdown(f"---\n### {vr['label']}")
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Predicted Probability", f"{vr['prob']:.1%}")
            vc2.metric("Risk Category", vr['cat'])
            vc3.metric("LIME/SHAP Agreement", vr['agreement'])

            col_vs, col_vl = st.columns(2)
            with col_vs:
                st.markdown("**Top 3 SHAP Contributors**")
                shap_val_df = pd.DataFrame([{
                    'Feature': _plain_english_factor(f),
                    'SHAP Value': f"{v:+.4f}",
                    'Direction': '⬆️' if v > 0 else '⬇️'
                } for f, v in vr['shap3']])
                st.dataframe(shap_val_df, use_container_width=True, hide_index=True)

            with col_vl:
                st.markdown("**Top 3 LIME Contributors**")
                if vr['lime3']:
                    lime_val_df = pd.DataFrame([{
                        'Feature (condition)': fd,
                        'LIME Value': f"{c:+.4f}",
                        'Direction': '⬆️' if c > 0 else '⬇️'
                    } for fd, c in vr['lime3']])
                    st.dataframe(lime_val_df, use_container_width=True, hide_index=True)
                else:
                    st.info("LIME unavailable.")

            # Commentary on agreement/disagreement
            if vr['agreement'] == "None":
                st.warning(
                    f"⚠️ **Disagreement detected** for {vr['label']}: SHAP and LIME indicate "
                    "opposite directions for the top factor. "
                    "Possible causes: (1) LIME's local linear surrogate may not capture "
                    "the RF's non-linear boundary; (2) feature dependencies or interactions "
                    "are not well-captured by LIME's independent perturbation sampling; "
                    "(3) the decision boundary near this point may be complex. "
                    "This does not necessarily indicate an error — both methods have known limitations. "
                    "Human review and domain expertise are required."
                )
            elif vr['agreement'] == "Direction":
                st.info(
                    f"ℹ️ **Partial agreement** for {vr['label']}: SHAP and LIME agree on direction "
                    "for the top factor but may disagree on exact feature ranking. "
                    "LIME expresses conditions (e.g. 'amount > 50000') rather than raw feature names, "
                    "so ranking comparison is approximate."
                )
            else:
                st.success(
                    f"✅ **Good agreement** for {vr['label']}: SHAP and LIME broadly agree on the "
                    "top contributing factor and its direction. This increases confidence in the "
                    "local explanation for this claim."
                )

        st.markdown("""
---
**Validation Study Notes:**
- LIME random seed is fixed at 42 for reproducibility (`random_state=42`).
- LIME uses 500 perturbation samples — increasing this improves stability but increases compute time.
- LIME describes features as interval conditions (e.g. `feature > threshold`) rather than exact names.
  This makes direct feature-name comparison with SHAP approximate.
- Disagreements between SHAP and LIME are expected to some degree:
  SHAP is exact for tree models; LIME is a local linear approximation.
  Neither method proves causation.
- To regenerate results: re-upload `data/insurance_claims.csv` and re-expand this section.
""")

# ---------------------------------------------------------------------------
# Download scored results
# ---------------------------------------------------------------------------
st.header("⬇️ Download Results")
output_cols = [c for c in df_raw.columns if c in df_feat.columns] + ['risk_probability', 'risk_category']
csv_buffer = io.StringIO()
df_feat[output_cols].to_csv(csv_buffer, index=False)
st.download_button(
    "Download scored claims as CSV",
    data=csv_buffer.getvalue(),
    file_name="scored_claims.csv",
    mime="text/csv"
)
