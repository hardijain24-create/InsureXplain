# 🔍 InsureXplain: Interpretable Underwriting Engine for Life/Motor Insurance Claims

**AI-powered fraud risk scoring for auto-insurance claims — with SHAP + LIME explainability,
Adverse Action Notices, and a Regulatory Compliance Report.**

> **College Micro-Project**: Trustworthy and Responsible AI  
> **Author**: Deepak  
> **Target**: Fraud detection in auto-insurance claims (`fraud_reported` = Y/N)

**🚀 [Live App](https://insurance-claim-risk-ap7ujahjub93vknofkwmsh.streamlit.app/) &nbsp;·&nbsp; 📦 [Repo](https://github.com/deedpak/insurance-claim-risk)**

---

## ⚠️ Important: Scope Clarification

This project detects **fraud risk** in auto-insurance claims. It does **not** make underwriting
eligibility decisions, issue coverage denials, or change premiums. All model outputs are
decision-support estimates that require human review before any action is taken.

---

## What It Does

| Step | Description |
|---|---|
| 🧹 **Clean & Understand** | 1,000 real auto-insurance claims, 39 fields — missing values, duplicates, and data leakage handled (Notebook 01) |
| 📊 **Explore & Engineer** | 7 visual analyses + 9 engineered features (Notebooks 02–03) |
| 🤖 **Model & Explain** | Baseline → Logistic Regression → Random Forest, evaluated with imbalance-aware metrics, explained per-claim with SHAP + LIME (Notebook 04) |
| 🔬 **Local Explanations** | SHAP waterfall + LIME surrogate for any individual claim, side-by-side |
| 🌍 **Global Explainability** | SHAP beeswarm summary plot across all uploaded claims |
| 📄 **Adverse Action Notice** | Customer-facing plain-English claim-review notice with top-3 SHAP factors |
| 🏛️ **Compliance Report** | Downloadable HTML report: proxy audit, feature interactions, fairness analysis |
| 🔁 **Validation Study** | LIME vs SHAP comparison across 3 representative records |

---

## Project Structure

```
insurance-claim-risk/
├── data/
│   ├── insurance_claims.csv              # original dataset (1,000 claims, 39 fields)
│   ├── insurance_claims_cleaned.csv      # after notebook 1
│   └── insurance_claims_features.csv     # after notebook 3
├── notebooks/
│   ├── 01_data_understanding_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_explainability.ipynb  # trains model, saves artifacts
├── models/
│   ├── fraud_model.pkl                   # trained Random Forest (300 trees, max_depth=8)
│   ├── feature_columns.pkl               # exact column order (152 features)
│   └── categorical_columns.pkl           # 17 categorical columns for one-hot encoding
├── app/
│   └── app.py                            # Streamlit app (all 3 deliverables)
├── tests/
│   └── test_deliverables.py              # pytest unit tests for all deliverables
├── requirements.txt
└── README.md
```

---

## Quick Start (Windows)

### 1. Clone and set up

```powershell
git clone https://github.com/deedpak/insurance-claim-risk.git
cd insurance-claim-risk
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
streamlit run app/app.py
```

Open `http://localhost:8501` and upload `data/insurance_claims.csv`.

### 4. (Optional) Set Gemini API key for live AI summaries

```powershell
setx GEMINI_API_KEY "your-key-here"   # Windows — persists across sessions
```

Free key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)  
Without a key the app falls back to a rule-based summary — all other features work fully.

### 5. (Optional) Regenerate the model from scratch

```powershell
jupyter notebook
```

Run notebooks 01 → 02 → 03 → 04 in order. Notebook 04 saves the model artifacts to `models/`.

---

## Three Deliverables — Where to Find Them

### Deliverable 1: Model Training Pipeline + Automated Explanation Engine

| Feature | Location |
|---|---|
| Model training pipeline | `notebooks/04_modeling_explainability.ipynb` |
| SHAP local bar chart | App → "Individual Claim Explanation" section |
| SHAP waterfall plot | App → "Individual Claim Explanation" → waterfall |
| LIME local explanation | App → "Individual Claim Explanation" → LIME panel |
| SHAP vs LIME side-by-side | App → "Individual Claim Explanation" |
| Global SHAP beeswarm | App → "Global Explainability" section |
| Global SHAP importance bar | App → "Global Explainability" (expandable) |

### Deliverable 2: Adverse Action Notice Generator

| Feature | Location |
|---|---|
| Notice section | App → "Adverse Action Notice / Decision Explanation" |
| Top-3 SHAP factors (primary ranking) | Displayed in notice section table |
| Top-3 LIME factors (supplementary) | Displayed alongside SHAP |
| SHAP vs LIME agreement check | Shown with ✅ / ⚠️ indicator |
| Plain-English notice with direction | Notice box (scrollable) |
| TXT download button | "Download Notice as TXT" button |

### Deliverable 3: Global Regulatory Compliance Report

| Feature | Location |
|---|---|
| Proxy audit (A + C) | App → "Compliance Audit" → "Proxy Audit Results" |
| Feature interactions (B) | App → "Compliance Audit" → correlation heatmap + PDP |
| Fairness analysis by sex | App → "Compliance Audit" → sex-based mean risk table |
| HTML report download | App → "Compliance Audit" → "Download Compliance Report" |
| LIME vs SHAP validation (3 records) | App → "LIME vs SHAP Validation Study" |

---

## How to Demonstrate Each Feature

1. **Start app**: `streamlit run app/app.py`
2. **Upload**: `data/insurance_claims.csv` via the sidebar
3. **Scoring**: Watch the "Scoring Claims" section compute fraud risk for all 1,000 claims
4. **Model metrics**: Since the file contains `fraud_reported` labels, live Precision/Recall/F1/AUC appear
5. **Individual explanation**:
   - Select any High-risk claim from the dropdown
   - See SHAP bar chart + LIME bar chart side-by-side
   - See SHAP waterfall plot
6. **Adverse Action Notice**:
   - Scroll to "Adverse Action Notice" section
   - Top-3 SHAP + LIME factors shown in tables
   - Notice text generated automatically
   - Click "Download Notice as TXT"
7. **Compliance Report**:
   - Scroll to "Compliance Audit"
   - View feature correlation heatmap
   - View partial dependence plot for claim amount
   - View sex-based fairness analysis
   - Click "Download Compliance Report (HTML)"
8. **Validation Study**:
   - Expand "LIME vs SHAP Validation Study"
   - See 3 representative records with full comparison

---

## Model Performance

**Final model**: `RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced')`

| Metric | Value (upload labeled CSV to see live) |
|---|---|
| Precision | See app / Notebook 04 |
| Recall | See app / Notebook 04 |
| F1 Score | See app / Notebook 04 |
| ROC-AUC | See app / Notebook 04 |

> **Why not accuracy?** ~75% of claims are genuine. A naive "always predict genuine" model would
> score 75% accuracy while catching zero fraud. We use Precision, Recall, F1, and ROC-AUC instead.

---

## Running Tests

```powershell
.venv\Scripts\activate
pip install pytest
python -m pytest tests/ -v
```

Tests cover:
- Model loading and artifact integrity
- Prediction shape and probability validity
- SHAP value computation and top-3 extraction
- LIME explainer build, single-record explanation, reproducibility
- Adverse action notice generation (content, disclaimer, date)
- Compliance report HTML generation (structure, metrics, proxy text)
- Full end-to-end pipeline smoke test

---

## Limitations

- **Dataset**: Synthetic/semi-public dataset. May not reflect real-world fraud patterns.
- **Protected attributes**: `insured_sex` is used directly as a model feature. Legal review required for any real deployment.
- **Race/ethnicity/nationality**: Absent from dataset — group fairness metrics for these classes cannot be computed.
- **LIME vs SHAP disagreement**: Expected to some degree. SHAP is exact for tree models; LIME is an approximation.
- **Not for production**: This is a college demonstration. Do not use for real insurance decisions.
- **Causal inference**: Correlations found in SHAP and the proxy audit do not establish causation.

---

## Viva Preparation Guide

### SHAP (SHapley Additive exPlanations)
- **What**: Assigns each feature an exact contribution value based on game theory (Shapley values)
- **How (tree models)**: `shap.TreeExplainer` computes exact values by traversing all decision paths
- **Local vs Global**: Local = one prediction; Global = average |SHAP| across all predictions
- **Waterfall plot**: Shows how SHAP values accumulate from the base value (model average) to the final prediction
- **Why it's trustworthy**: Satisfies efficiency, symmetry, dummy, and linearity axioms — mathematically rigorous

### LIME (Local Interpretable Model-agnostic Explanations)
- **What**: Fits a simple linear model around a single prediction by perturbing the input
- **How**: Generates `n_samples` (here: 500) perturbed records → gets model predictions → trains linear regression
- **Limitation**: Only locally valid; may disagree with SHAP due to linear approximation of non-linear boundaries
- **Reproducibility**: Fixed `random_state=42` ensures identical results each run

### Global vs Local Explainability
- **Global**: Explains overall model behaviour (SHAP beeswarm, mean |SHAP|, feature importance)
- **Local**: Explains one specific prediction (SHAP waterfall, LIME bar chart)
- **When to use what**: Global for model audits/compliance; Local for customer explanations/notices

### Proxy Discrimination
- **Definition**: A variable that is not a protected attribute but correlates with one
- **Examples here**: `insured_occupation` (→ socioeconomic class → ethnicity proxy), `incident_city/state` (→ geographic racial proxy)
- **What we can detect**: Statistical differences in mean risk scores across `insured_sex` categories
- **What we cannot detect**: Race/ethnicity discrimination (data absent); causal discrimination

### Feature Interactions
- **Correlation ≠ causation**: Two features correlated with fraud doesn't mean either causes it
- **PDP**: Partial Dependence Plot — shows model's average prediction as one feature varies, holding others fixed
- **SHAP interaction values**: Can be computed for tree models (`TreeExplainer(model).shap_interaction_values(X)`) but are computationally expensive at scale

### Adverse Action Notice Requirements
- Must be understandable to a non-technical customer
- Must state the model's estimate, not a definitive finding
- Must identify top contributing factors
- Must require human review before any adverse decision
- Must have a unique reference number and date

---

## Key Design Decisions

- **No data leakage**: `policy_number`, `insured_zip`, `incident_location` dropped before modelling
- **Class imbalance**: `class_weight='balanced'` — keeps training distribution honest vs naive oversampling
- **LIME sampling**: 500 perturbation samples with `random_state=42` — reproducible and computationally practical
- **SHAP base value**: `TreeExplainer.expected_value` — the model's average prediction on training data
- **Top-3 factor selection**: Ranked by `|SHAP value|` (absolute magnitude) — documented, consistent, based on actual contributions not raw feature values

---

## Tech Stack

**Python** · **pandas** / **NumPy** — data wrangling  
**scikit-learn** — modelling (Logistic Regression, Random Forest)  
**SHAP** — per-prediction + global explainability  
**LIME** — local surrogate explanations  
**Streamlit** — interactive web app  
**Google Gemini API** — optional AI investigator summaries (fallback: rule-based)  
**fpdf2** — PDF generation support

---

Built by **Deepak** · InsureXplain College Micro-Project · Trustworthy and Responsible AI
