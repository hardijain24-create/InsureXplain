"""
app/utils.py
============
Pure utility functions shared between app.py and tests.
These functions have NO Streamlit dependency and can be imported freely.
"""

import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Risk category helper
# ---------------------------------------------------------------------------

def risk_category(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Plain-English feature name conversion
# ---------------------------------------------------------------------------

def plain_english_factor(feature_name: str) -> str:
    """Convert a model feature name to a plain-English description."""
    lookup = {
        "total_claim_amount": "Total claimed amount",
        "policy_age_days": "Age of policy at time of incident",
        "customer_tenure_years": "Length of customer relationship (years)",
        "claim_amount_per_tenure_year": "Claim amount relative to customer tenure",
        "is_high_value_claim": "Claim flagged as high-value (top 10%)",
        "vehicle_claim_ratio": "Proportion of claim attributed to vehicle damage",
        "injury_claim_ratio": "Proportion of claim attributed to bodily injury",
        "property_claim_ratio": "Proportion of claim attributed to property damage",
        "missing_doc_count": "Number of missing documents (police report, witnesses, damage report)",
        "is_odd_hour_incident": "Incident occurred during unusual hours (10pm–5am)",
        "bodily_injuries": "Number of bodily injuries reported",
        "witnesses": "Number of witnesses",
        "number_of_vehicles_involved": "Number of vehicles involved",
        "policy_annual_premium": "Annual policy premium",
        "policy_deductable": "Policy deductible amount",
        "umbrella_limit": "Umbrella coverage limit",
        "capital-gains": "Capital gains reported",
        "capital-loss": "Capital losses reported",
        "incident_hour_of_the_day": "Hour of day when incident occurred",
        "months_as_customer": "Months as a customer",
        "age": "Policyholder age",
        "auto_year": "Vehicle year",
    }
    for k, v in lookup.items():
        if feature_name.startswith(k):
            suffix = feature_name[len(k):]
            if suffix:
                return f"{v} — category: {suffix.lstrip('_')}"
            return v
    return feature_name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Adverse Action Notice generator
# ---------------------------------------------------------------------------

def generate_adverse_action_notice(
    claim_id, risk_prob, risk_cat, shap_top3, lime_top3, claim_date=None
):
    """
    Generate a plain-English adverse action / claim-review notice.

    Parameters
    ----------
    claim_id : str
        Unique claim identifier.
    risk_prob : float
        Model's predicted fraud probability [0, 1].
    risk_cat : str
        Risk category — 'High', 'Medium', or 'Low'.
    shap_top3 : list of dict
        Top-3 SHAP factors, each with keys 'feature' and 'impact'.
    lime_top3 : list of (str, float) tuples
        Top-3 LIME factors as (feature_description, contribution) pairs.
    claim_date : str, optional
        Date string for the notice. Defaults to today.

    Returns
    -------
    str
        Plain-text notice.
    """
    today = datetime.date.today().strftime("%d %B %Y")
    ref = f"REF-{claim_id}-{datetime.date.today().strftime('%Y%m%d')}"

    direction_map = {True: "increased", False: "decreased"}

    shap_lines = []
    for i, f in enumerate(shap_top3, 1):
        direction = direction_map[f['impact'] > 0]
        plain = plain_english_factor(f['feature'])
        shap_lines.append(
            f"  {i}. {plain}\n"
            f"     → This factor {direction} the model's estimated fraud-risk score "
            f"(SHAP contribution: {f['impact']:+.4f})."
        )

    lime_lines = []
    for j, (feat_desc, contrib) in enumerate(lime_top3, 1):
        direction = direction_map[contrib > 0]
        lime_lines.append(
            f"  {j}. {feat_desc}\n"
            f"     → Contribution: {contrib:+.4f} ({direction} fraud likelihood per LIME surrogate)."
        )

    # Check SHAP vs LIME agreement on top factor direction
    agreement_note = ""
    if shap_top3 and lime_top3:
        shap_dir = shap_top3[0]['impact'] > 0
        lime_dir = lime_top3[0][1] > 0
        if shap_dir == lime_dir:
            agreement_note = (
                "Note: Both SHAP and LIME agree on the direction of the primary contributing factor."
            )
        else:
            agreement_note = (
                "Note: SHAP and LIME disagree on the direction of the top factor. "
                "This may reflect LIME's local linear approximation differing from the "
                "tree model's actual non-linear boundaries. Human review is especially important here."
            )

    if risk_cat == 'High':
        risk_sentence = 'There are significant signals that warrant careful human investigation.'
    elif risk_cat == 'Medium':
        risk_sentence = 'There are moderate signals that may warrant additional scrutiny.'
    else:
        risk_sentence = 'The model detected limited fraud signals for this claim.'

    nl = "\n"
    notice = f"""
================================================================================
              CLAIM FRAUD RISK REVIEW — DECISION SUPPORT NOTICE
================================================================================
Reference Number : {ref}
Date of Assessment: {today}
Assessment Type  : Automated Model Estimate (Decision-Support Only)

IMPORTANT: This notice is generated by a machine-learning decision-support tool.
It is NOT a final fraud determination and does NOT constitute an adverse decision.
A human claims investigator must review this case before any action is taken.

────────────────────────────────────────────────────────────────────────────────
CLAIM OVERVIEW
────────────────────────────────────────────────────────────────────────────────
Claim Identifier   : {claim_id}
Model Prediction   : {risk_cat} Fraud Risk
Estimated Probability: {risk_prob:.1%}
  (Interpreted as: {risk_sentence})

────────────────────────────────────────────────────────────────────────────────
TOP THREE CONTRIBUTING FACTORS  (ranked by SHAP contribution magnitude)
────────────────────────────────────────────────────────────────────────────────
The following three features had the largest influence on the model's estimate
for this specific claim. A positive contribution means the feature increased the
model's fraud-risk estimate; a negative contribution means it decreased it.

{nl.join(shap_lines) if shap_lines else '  No SHAP factors available.'}

────────────────────────────────────────────────────────────────────────────────
SUPPLEMENTARY LIME EXPLANATION  (local surrogate model)
────────────────────────────────────────────────────────────────────────────────
LIME creates a simple local approximation of the model around this claim.
The top three factors from LIME are:

{nl.join(lime_lines) if lime_lines else '  LIME explanation not available for this record.'}

{agreement_note}

────────────────────────────────────────────────────────────────────────────────
WHAT THIS NOTICE MEANS
────────────────────────────────────────────────────────────────────────────────
• The model's output is a statistical risk estimate, not proof of fraud.
• The model was trained on historical patterns and may not reflect all relevant
  factors specific to this claim.
• This output must be treated as one input among many in a human review process.
• No adverse decision (e.g. claim denial, investigation referral, penalty) should
  be based solely on this automated score.
• If you believe this assessment is incorrect, you have the right to request a
  human review by contacting your claims handler.

────────────────────────────────────────────────────────────────────────────────
DISCLAIMER
────────────────────────────────────────────────────────────────────────────────
This is a demonstration notice generated by the InsureXplain college micro-project
(Trustworthy & Responsible AI). It is NOT a legally compliant adverse action notice
and must not be used in any real insurance decision workflow.

================================================================================
""".strip()

    return notice


# ---------------------------------------------------------------------------
# Compliance report HTML builder
# ---------------------------------------------------------------------------

def build_compliance_report_html(
    dataset_shape, fraud_dist, model_metrics,
    shap_global_text, proxy_audit_text, interaction_text,
    global_shap_img_b64=None
):
    """Build an HTML compliance report string."""
    today = datetime.date.today().strftime("%d %B %Y")

    metrics_rows = ""
    if model_metrics:
        for k, v in model_metrics.items():
            metrics_rows += f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
    else:
        metrics_rows = (
            "<tr><td colspan='2'><em>Upload labeled data (with fraud_reported column) "
            "to see live metrics.</em></td></tr>"
        )

    img_section = ""
    if global_shap_img_b64:
        img_section = f"""
        <h3>Global SHAP Feature Importance Plot</h3>
        <img src="data:image/png;base64,{global_shap_img_b64}"
             style="max-width:100%;border:1px solid #ddd;border-radius:8px;"
             alt="SHAP Summary Plot"/>
        """

    n_fraud = fraud_dist.get('Y', 0)
    n_total = dataset_shape[0]
    n_genuine = fraud_dist.get('N', 0)
    fraud_pct = (n_fraud / n_total * 100) if isinstance(n_fraud, int) and n_total > 0 else 0
    genuine_pct = (n_genuine / n_total * 100) if isinstance(n_genuine, int) and n_total > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>InsureXplain — Regulatory Compliance Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 40px auto;
         color: #1B2A4A; background: #F7F8FC; padding: 0 20px 60px; }}
  h1 {{ color: #101C36; border-bottom: 3px solid #2E5BFF; padding-bottom: 10px; }}
  h2 {{ color: #1B2A4A; border-left: 4px solid #2E5BFF; padding-left: 12px; margin-top: 40px; }}
  h3 {{ color: #2E5BFF; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #101C36; color: #fff; padding: 8px 12px; text-align: left; }}
  td {{ border: 1px solid #D6DCEF; padding: 8px 12px; }}
  tr:nth-child(even) {{ background: #EFF2FA; }}
  .warning {{ background: #FFF3DC; border-left: 4px solid #B7791F; padding: 12px 16px;
              border-radius: 6px; margin: 16px 0; }}
  .info {{ background: #E7F6EE; border-left: 4px solid #1E8A57; padding: 12px 16px;
           border-radius: 6px; margin: 16px 0; }}
  .danger {{ background: #FDECEC; border-left: 4px solid #C13B44; padding: 12px 16px;
             border-radius: 6px; margin: 16px 0; }}
  pre {{ background: #101C36; color: #AEB8D6; padding: 16px; border-radius: 8px;
         overflow-x: auto; font-size: 0.9rem; white-space: pre-wrap; }}
  footer {{ margin-top: 60px; color: #6B7280; font-size: 0.85rem;
            border-top: 1px solid #D6DCEF; padding-top: 16px; }}
</style>
</head>
<body>
<h1>🔍 InsureXplain — Regulatory Compliance Report</h1>
<p><strong>Generated:</strong> {today} &nbsp;|&nbsp;
   <strong>Project:</strong> InsureXplain: Interpretable Underwriting Engine &nbsp;|&nbsp;
   <strong>Course:</strong> Trustworthy and Responsible AI (Micro-Project)</p>

<div class="warning">
⚠️ <strong>Disclaimer:</strong> This is an academic demonstration report. It does not constitute
a formal regulatory compliance audit. No insurance decisions should be based on this output.
</div>

<h2>1. Project Overview</h2>
<p><strong>Prediction target:</strong> Binary fraud detection — whether an auto-insurance claim is
fraudulent (<code>fraud_reported = Y</code>) or genuine (<code>fraud_reported = N</code>).</p>
<p><strong>Scope:</strong> This model is a <em>decision-support tool</em>. It produces a risk
probability to assist human investigators; it does not make final fraud determinations and is not
used for underwriting eligibility, coverage denial, or premium setting.</p>

<h2>2. Dataset Description</h2>
<table>
  <tr><th>Attribute</th><th>Value</th></tr>
  <tr><td>Total records</td><td>{n_total:,}</td></tr>
  <tr><td>Raw features</td><td>{dataset_shape[1]}</td></tr>
  <tr><td>Fraudulent claims</td><td>{n_fraud} ({fraud_pct:.1f}%)</td></tr>
  <tr><td>Genuine claims</td><td>{n_genuine} ({genuine_pct:.1f}%)</td></tr>
  <tr><td>Source domain</td><td>Auto-insurance claims (synthetic/public dataset)</td></tr>
</table>
<p><strong>Dataset limitations:</strong> The dataset is a publicly available synthetic/semi-synthetic
dataset. It may not reflect real-world fraud patterns. Class imbalance (~25% fraud) exists and is
handled via <code>class_weight='balanced'</code>.</p>

<h2>3. Model Details</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Algorithm</td><td>Random Forest Classifier</td></tr>
  <tr><td>n_estimators</td><td>300</td></tr>
  <tr><td>max_depth</td><td>8</td></tr>
  <tr><td>class_weight</td><td>balanced</td></tr>
  <tr><td>Evaluation</td><td>5-fold cross-validation + held-out 20% test set</td></tr>
  <tr><td>Explainability</td><td>SHAP TreeExplainer + LIME LimeTabularExplainer</td></tr>
</table>

<h2>4. Model Performance Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {metrics_rows}
</table>
<p><em>Note: Accuracy is not the primary metric due to class imbalance. Precision, Recall, F1,
and ROC-AUC are reported instead.</em></p>

<h2>5. Global Explainability (SHAP)</h2>
{img_section}
<pre>{shap_global_text}</pre>

<h2>6. Feature Interactions</h2>
<pre>{interaction_text}</pre>
<div class="info">
ℹ️ <strong>Important:</strong> Feature associations shown here are statistical correlations,
not causal relationships. A feature correlated with fraud in historical data does not necessarily
cause fraud. Causal inference would require domain expertise and controlled study designs.
</div>

<h2>7. Proxy &amp; Fairness Audit</h2>
<pre>{proxy_audit_text}</pre>
<div class="warning">
⚠️ <strong>Proxy Discrimination Risk:</strong> Even if protected attributes are excluded from the
model, correlated features may act as proxies. The findings above identify potential proxies that
should be reviewed by domain experts and legal counsel before any deployment.
</div>
<div class="danger">
🚫 <strong>Limitation:</strong> This dataset does not contain ground-truth demographic group labels
sufficient for a full group fairness analysis (e.g. demographic parity, equalised odds).
Additional data collection and domain expert review is required before any real-world deployment.
</div>

<h2>8. Limitations and Recommended Safeguards</h2>
<ul>
  <li>Human review is required before any adverse action is taken based on model output.</li>
  <li>The model should be retrained periodically on fresh data to avoid concept drift.</li>
  <li>SHAP and LIME explanations are approximations; they should be validated on held-out data.</li>
  <li>Proxy variable analysis should be expanded with a diverse demographic dataset.</li>
  <li>Consider differential privacy or federated learning if real PII is ever used.</li>
  <li>Implement model monitoring in production to detect distribution shift.</li>
  <li>Conduct periodic third-party audits of model fairness and accuracy.</li>
</ul>

<h2>9. Conclusion</h2>
<p>The InsureXplain prototype demonstrates interpretable, responsible AI applied to insurance fraud
detection. SHAP and LIME provide complementary local and global explanations. The proxy audit
identifies features warranting further legal review. The model's outputs are decision-support
signals only and should always be combined with human judgment.</p>

<footer>
  InsureXplain — College Micro-Project &nbsp;|&nbsp; Trustworthy and Responsible AI &nbsp;|&nbsp;
  Generated {today} &nbsp;|&nbsp;
  ⚠️ Academic use only — not for production deployment.
</footer>
</body>
</html>"""
    return html
