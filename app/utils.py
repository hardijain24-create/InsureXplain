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
    """Build a self-contained HTML compliance report string.

    All user-derived / dataset-derived values are HTML-escaped before insertion.
    """
    import html as _html

    today = datetime.date.today().strftime("%d %B %Y")

    def esc(v):
        """Safely escape a value for insertion into HTML."""
        return _html.escape(str(v))

    metrics_rows = ""
    if model_metrics:
        for k, v in model_metrics.items():
            metrics_rows += f"<tr><td>{esc(k)}</td><td><strong>{esc(v)}</strong></td></tr>"
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
             style="max-width:100%;border:1px solid #E6E8EB;border-radius:8px;"
             alt="SHAP Summary Plot"/>
        """

    n_fraud   = fraud_dist.get("Y", 0)
    n_total   = dataset_shape[0]
    n_genuine = fraud_dist.get("N", 0)
    fraud_pct   = (n_fraud   / n_total * 100) if isinstance(n_fraud,   int) and n_total > 0 else 0
    genuine_pct = (n_genuine / n_total * 100) if isinstance(n_genuine, int) and n_total > 0 else 0

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>InsureXplain — Compliance Report</title>
<style>
  /* ── Design system aligned with InsureXplain dashboard ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root {{
    --color-background: #F5F6F7;
    --color-surface: #FFFFFF;
    --color-text: #17191C;
    --color-accent: #9BCB45;
    --color-charcoal: #1E2126;
    --color-secondary-text: #737780;
    --color-border: #E6E8EB;
  }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    max-width: 920px;
    margin: 40px auto;
    color: var(--color-text);
    background: var(--color-background);
    padding: 0 24px 80px;
    font-size: 0.95rem;
    line-height: 1.65;
  }}
  /* Header */
  .report-header {{
    background: var(--color-charcoal);
    color: #FFFFFF;
    padding: 2rem 2.4rem;
    border-radius: 12px;
    margin-bottom: 2rem;
  }}
  .report-header h1 {{
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
  }}
  .report-header .accent {{ color: var(--color-accent); }}
  .report-header p {{
    color: #C8CDD4;
    margin: 0;
    font-size: 0.87rem;
  }}
  /* Section headings */
  h2 {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--color-text);
    border-left: 4px solid var(--color-accent);
    padding-left: 12px;
    margin-top: 2.4rem;
    margin-bottom: 0.8rem;
  }}
  h3 {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-text);
    margin-top: 1.4rem;
  }}
  /* Cards */
  .card {{
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
  }}
  /* Tables */
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 0.8rem 0 1.2rem;
    font-size: 0.87rem;
  }}
  th {{
    background: var(--color-charcoal);
    color: #FFFFFF;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  td {{
    border-bottom: 1px solid var(--color-border);
    padding: 8px 12px;
  }}
  tr:nth-child(even) td {{ background: var(--color-background); }}
  /* Alert boxes */
  .alert {{
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.87rem;
    border-left: 4px solid;
  }}
  .alert-warning  {{ background: #FEF3C7; border-color: #FDE68A; color: #92400E; }}
  .alert-info     {{ background: #ECFDF5; border-color: #A7F3D0; color: #065F46; }}
  .alert-danger   {{ background: #FEECEC; border-color: #FECACA; color: #B91C1C; }}
  .alert-note     {{ background: #F4F8EA; border-color: var(--color-accent); color: #46552E; }}
  .alert strong   {{ font-weight: 700; }}
  /* Code blocks */
  pre {{
    background: var(--color-charcoal);
    color: var(--color-accent);
    padding: 1rem 1.2rem;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.83rem;
    white-space: pre-wrap;
    margin: 0.8rem 0;
  }}
  /* Footer */
  footer {{
    margin-top: 3rem;
    color: var(--color-secondary-text);
    font-size: 0.8rem;
    border-top: 1px solid var(--color-border);
    padding-top: 1rem;
    text-align: center;
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>Insure<span class="accent">X</span>plain &mdash; Regulatory Compliance Report</h1>
  <p>Generated: {esc(today)} &nbsp;|&nbsp; Project: InsureXplain: Interpretable Underwriting Engine
     &nbsp;|&nbsp; Course: Trustworthy and Responsible AI (Micro-Project)</p>
</div>

<div class="alert alert-warning">
  <strong>Disclaimer:</strong> This is an academic demonstration report. It does not constitute
  a formal regulatory compliance audit. No insurance decisions should be based on this output.
  The model predicts auto-insurance claim fraud risk only &mdash; it is not an underwriting
  eligibility, coverage-denial, or premium-pricing engine.
</div>

<h2>1. Project Overview</h2>
<div class="card">
  <p><strong>Prediction target:</strong> Binary fraud detection &mdash; whether an auto-insurance
  claim is fraudulent (<code>fraud_reported = Y</code>) or genuine (<code>fraud_reported = N</code>).</p>
  <p><strong>Scope:</strong> This model is a <em>decision-support tool</em>. It produces a risk
  probability to assist human investigators; it does not make final fraud determinations and is not
  used for underwriting eligibility, coverage denial, or premium setting.</p>
</div>

<h2>2. Dataset Description</h2>
<div class="card">
<table>
  <tr><th>Attribute</th><th>Value</th></tr>
  <tr><td>Total records</td><td>{n_total:,}</td></tr>
  <tr><td>Raw features</td><td>{esc(dataset_shape[1])}</td></tr>
  <tr><td>Fraudulent claims (Y)</td><td>{esc(n_fraud)} ({fraud_pct:.1f}%)</td></tr>
  <tr><td>Genuine claims (N)</td><td>{esc(n_genuine)} ({genuine_pct:.1f}%)</td></tr>
  <tr><td>Source domain</td><td>Auto-insurance claims (synthetic/public dataset)</td></tr>
</table>
</div>
<div class="alert alert-note">
  <strong>Dataset limitations:</strong> The dataset is a publicly available synthetic/semi-synthetic
  dataset. It may not reflect real-world fraud patterns. Class imbalance (~25% fraud) exists and is
  handled via <code>class_weight='balanced'</code>.
</div>

<h2>3. Model Details</h2>
<div class="card">
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Algorithm</td><td>Random Forest Classifier</td></tr>
  <tr><td>n_estimators</td><td>300</td></tr>
  <tr><td>max_depth</td><td>8</td></tr>
  <tr><td>class_weight</td><td>balanced</td></tr>
  <tr><td>Evaluation</td><td>5-fold cross-validation + held-out 20% test set</td></tr>
  <tr><td>Explainability</td><td>SHAP TreeExplainer + LIME LimeTabularExplainer</td></tr>
</table>
</div>

<h2>4. Model Performance Metrics</h2>
<div class="card">
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {metrics_rows}
</table>
<p><em>Note: Accuracy is not the primary metric due to class imbalance. Precision, Recall, F1,
and ROC-AUC are reported instead. Threshold = 0.5.</em></p>
</div>

<h2>5. Global Explainability (SHAP)</h2>
<div class="card">
{img_section}
<pre>{esc(shap_global_text)}</pre>
</div>
<div class="alert alert-note">
  <strong>Important:</strong> Global SHAP importance shows which features most consistently
  influence the model's output across the dataset. High importance does not imply causation.
  These are statistical associations in historical data.
</div>

<h2>6. Feature Interactions</h2>
<div class="card">
<pre>{esc(interaction_text)}</pre>
</div>
<div class="alert alert-note">
  <strong>Important:</strong> Feature associations shown here are statistical correlations,
  not causal relationships. A feature correlated with fraud in historical data does not
  necessarily cause fraud. Causal inference would require domain expertise and controlled study designs.
</div>

<h2>7. Proxy &amp; Fairness Audit</h2>
<div class="card">
<pre>{esc(proxy_audit_text)}</pre>
</div>
<div class="alert alert-warning">
  <strong>Proxy Discrimination Risk:</strong> Even if protected attributes are excluded from the
  model, correlated features may act as proxies. The findings above identify potential proxies that
  should be reviewed by domain experts and legal counsel before any deployment.
</div>
<div class="alert alert-danger">
  <strong>Limitation:</strong> This dataset does not contain ground-truth demographic group labels
  sufficient for a full group fairness analysis (e.g. demographic parity, equalised odds).
  Additional data collection and domain expert review is required before any real-world deployment.
</div>

<h2>8. Limitations and Recommended Safeguards</h2>
<div class="card">
<ul>
  <li>Human review is required before any adverse action is taken based on model output.</li>
  <li>The model should be retrained periodically on fresh data to avoid concept drift.</li>
  <li>SHAP and LIME explanations are approximations; they should be validated on held-out data.</li>
  <li>Proxy variable analysis should be expanded with a diverse demographic dataset.</li>
  <li>Consider differential privacy or federated learning if real PII is ever used.</li>
  <li>Implement model monitoring in production to detect distribution shift.</li>
  <li>Conduct periodic third-party audits of model fairness and accuracy.</li>
  <li>The model uses <code>insured_sex</code> directly &mdash; legal review required before deployment.</li>
  <li>Race, ethnicity and national origin data are absent &mdash; full group fairness cannot be assessed.</li>
</ul>
</div>

<h2>9. Conclusion</h2>
<div class="card">
  <p>The InsureXplain prototype demonstrates interpretable, responsible AI applied to insurance
  fraud risk detection. SHAP and LIME provide complementary local and global explanations.
  The proxy audit identifies features warranting further legal review. The model's outputs are
  decision-support signals only and should always be combined with human judgement.</p>
  <p>This report covers only what can be assessed from the available data. Claims of full
  regulatory compliance or assurance of fairness are explicitly not made.</p>
</div>

<footer>
  InsureXplain &mdash; College Micro-Project &nbsp;|&nbsp; Trustworthy and Responsible AI
  &nbsp;|&nbsp; Generated {esc(today)}
  &nbsp;|&nbsp; Academic use only &mdash; not for production deployment.
</footer>
</body>
</html>"""
    return html_str

