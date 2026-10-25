"""
tests/test_deliverables.py
==========================
Unit tests for all three InsureXplain deliverables.
Run with: python -m pytest tests/ -v

These tests use the existing saved model artifacts and the original dataset.
They do NOT retrain the model.
"""

import sys
import os
import datetime
import pytest
import numpy as np
import pandas as pd
import joblib

# ── Ensure project root and app/ are on sys.path ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Import pure utilities (no Streamlit dependency)
from utils import (
    generate_adverse_action_notice,
    build_compliance_report_html,
    plain_english_factor,
    risk_category,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_artifacts():
    """Load model, feature columns, and categorical columns once per session."""
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    categorical_columns = joblib.load(os.path.join(MODELS_DIR, "categorical_columns.pkl"))
    return model, feature_columns, categorical_columns


@pytest.fixture(scope="session")
def raw_data():
    """Load the original dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "insurance_claims.csv"))


@pytest.fixture(scope="session")
def prepared_data(model_artifacts, raw_data):
    """Run the full preprocessing pipeline and return (df_feat, X_aligned)."""
    model, feature_columns, categorical_columns = model_artifacts

    def engineer_features(df):
        df = df.copy()
        if 'incident_date' in df.columns and 'policy_bind_date' in df.columns:
            df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
            df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
            df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days
        if 'months_as_customer' in df.columns:
            df['customer_tenure_years'] = df['months_as_customer'] / 12
            if 'total_claim_amount' in df.columns:
                df['claim_amount_per_tenure_year'] = (
                    df['total_claim_amount'] / df['customer_tenure_years'].replace(0, 0.1)
                )
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

    df = raw_data.replace('?', np.nan)
    df_feat = engineer_features(df)
    drop_cols = ['fraud_reported', 'fraud_flag', 'incident_date', 'policy_bind_date',
                 'incident_location', 'policy_number', 'insured_zip', '_c39']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors='ignore')
    cat_cols_present = [c for c in categorical_columns if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols_present, drop_first=True)
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_feat, X_aligned


# ---------------------------------------------------------------------------
# Deliverable 1: Model loading and prediction
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_model_files_exist(self):
        """All three model artifact files must exist."""
        assert os.path.exists(os.path.join(MODELS_DIR, "fraud_model.pkl")), \
            "fraud_model.pkl missing"
        assert os.path.exists(os.path.join(MODELS_DIR, "feature_columns.pkl")), \
            "feature_columns.pkl missing"
        assert os.path.exists(os.path.join(MODELS_DIR, "categorical_columns.pkl")), \
            "categorical_columns.pkl missing"

    def test_model_type(self, model_artifacts):
        """Model must be a RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier
        model, _, _ = model_artifacts
        assert isinstance(model, RandomForestClassifier), \
            f"Expected RandomForestClassifier, got {type(model)}"

    def test_feature_columns_count(self, model_artifacts):
        """Model should have >100 features after one-hot encoding."""
        _, feature_columns, _ = model_artifacts
        assert len(feature_columns) > 100, \
            f"Feature columns count {len(feature_columns)} seems too low"

    def test_prediction_shape(self, model_artifacts, prepared_data):
        """predict_proba output shape must match input rows and have 2 classes."""
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        probs = model.predict_proba(X_aligned)
        assert probs.shape == (len(X_aligned), 2), \
            f"Expected shape ({len(X_aligned)}, 2), got {probs.shape}"

    def test_probabilities_valid(self, model_artifacts, prepared_data):
        """All probabilities must be in [0, 1] and sum to 1 per row."""
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        probs = model.predict_proba(X_aligned)
        assert np.all(probs >= 0) and np.all(probs <= 1), \
            "Probabilities out of [0, 1] range"
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-6,
                                   err_msg="Row probabilities do not sum to 1")

    def test_fraud_column_at_index_1(self, model_artifacts):
        """Fraud class (1) must be at index 1 in model.classes_."""
        model, _, _ = model_artifacts
        classes = list(model.classes_)
        assert 1 in classes, "Class 1 (Fraud) not found in model classes"
        assert classes.index(1) == 1, \
            f"Expected fraud class at index 1, got {classes.index(1)}"

    def test_risk_category_boundaries(self):
        """risk_category must correctly classify probabilities."""
        assert risk_category(0.7) == "High"
        assert risk_category(0.6) == "High"
        assert risk_category(0.59) == "Medium"
        assert risk_category(0.3) == "Medium"
        assert risk_category(0.29) == "Low"
        assert risk_category(0.0) == "Low"


# ---------------------------------------------------------------------------
# Deliverable 1: SHAP explanation generation
# ---------------------------------------------------------------------------

class TestSHAPExplanation:
    def test_shap_values_shape(self, model_artifacts, prepared_data):
        """SHAP values array must match (n_samples, n_features)."""
        import shap
        model, feature_columns, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_aligned.head(50))
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vals = raw_shap[:, :, 1]
        else:
            shap_vals = raw_shap
        assert shap_vals.shape == (50, len(feature_columns)), \
            f"SHAP values shape mismatch: {shap_vals.shape}"

    def test_shap_values_are_finite(self, model_artifacts, prepared_data):
        """SHAP values must be finite real numbers (no NaN/Inf)."""
        import shap
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_aligned.head(20))
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1]
        else:
            shap_vals = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap
        assert np.all(np.isfinite(shap_vals)), "SHAP values contain NaN or Inf"

    def test_top3_extraction_returns_3_factors(self, model_artifacts, prepared_data):
        """Top-3 factor extraction must return exactly 3 factors."""
        import shap
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_aligned.head(5))
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1]
        else:
            shap_vals = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap

        claim_shap = shap_vals[0]
        top3 = (pd.DataFrame({'feature': X_aligned.columns, 'impact': claim_shap})
                  .sort_values('impact', key=abs, ascending=False)
                  .head(3)
                  .to_dict('records'))

        assert len(top3) == 3, f"Expected 3 factors, got {len(top3)}"
        for f in top3:
            assert 'feature' in f, "Factor missing 'feature' key"
            assert 'impact' in f, "Factor missing 'impact' key"

    def test_top3_ranked_by_magnitude(self, model_artifacts, prepared_data):
        """Top-3 must be sorted descending by |SHAP| value."""
        import shap
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_aligned.head(5))
        if isinstance(raw_shap, list):
            sv = raw_shap[1]
        else:
            sv = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap

        top3 = (pd.DataFrame({'feature': X_aligned.columns, 'impact': sv[0]})
                  .sort_values('impact', key=abs, ascending=False)
                  .head(3)
                  .to_dict('records'))
        abs_vals = [abs(f['impact']) for f in top3]
        assert abs_vals == sorted(abs_vals, reverse=True), \
            "Top-3 SHAP factors not sorted by absolute magnitude"

    def test_shap_base_value_exists(self, model_artifacts, prepared_data):
        """TreeExplainer must expose a usable expected_value (base value)."""
        import shap
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = shap.TreeExplainer(model)
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base = ev[1]
        else:
            base = ev
        assert isinstance(float(base), float), "Expected value must be numeric"
        assert 0.0 <= float(base) <= 1.0, \
            f"Expected value {base} outside [0, 1] — unexpected for probability output"


# ---------------------------------------------------------------------------
# Deliverable 1: LIME explanation generation
# ---------------------------------------------------------------------------

class TestLIMEExplanation:
    def test_lime_import(self):
        """lime must be importable."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            pytest.fail("lime is not installed. Run: pip install lime")

    def test_lime_explainer_builds(self, model_artifacts, prepared_data):
        """LimeTabularExplainer must build without errors."""
        from lime.lime_tabular import LimeTabularExplainer
        _, feature_columns, _ = model_artifacts
        _, X_aligned = prepared_data
        explainer = LimeTabularExplainer(
            training_data=X_aligned.values[:100],
            feature_names=list(X_aligned.columns),
            class_names=["Genuine", "Fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        assert explainer is not None

    def test_lime_explanation_single_record(self, model_artifacts, prepared_data):
        """LIME must produce at least 3 feature contributions for a single record."""
        from lime.lime_tabular import LimeTabularExplainer
        model, feature_columns, _ = model_artifacts
        _, X_aligned = prepared_data

        explainer = LimeTabularExplainer(
            training_data=X_aligned.values[:200],
            feature_names=list(X_aligned.columns),
            class_names=["Genuine", "Fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        exp = explainer.explain_instance(
            data_row=X_aligned.values[0],
            predict_fn=model.predict_proba,
            num_features=5,
            num_samples=200,
            labels=(1,)
        )
        results = exp.as_list(label=1)
        assert len(results) >= 3, f"Expected at least 3 LIME factors, got {len(results)}"
        for feat_desc, contrib in results:
            assert isinstance(feat_desc, str)
            assert isinstance(contrib, float)

    def test_lime_reproducible_with_seed(self, model_artifacts, prepared_data):
        """Two LIME runs with same random_state must produce the same top factor."""
        from lime.lime_tabular import LimeTabularExplainer
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data

        def run():
            e = LimeTabularExplainer(
                training_data=X_aligned.values[:200],
                feature_names=list(X_aligned.columns),
                class_names=["Genuine", "Fraud"],
                mode="classification",
                discretize_continuous=True,
                random_state=42
            )
            result = e.explain_instance(
                data_row=X_aligned.values[5],
                predict_fn=model.predict_proba,
                num_features=3,
                num_samples=300,
                labels=(1,)
            )
            return result.as_list(label=1)[0]

        r1, r2 = run(), run()
        assert r1[0] == r2[0], \
            f"LIME top feature not reproducible: '{r1[0]}' vs '{r2[0]}'"

    def test_lime_contributions_are_floats(self, model_artifacts, prepared_data):
        """All LIME contribution values must be Python floats."""
        from lime.lime_tabular import LimeTabularExplainer
        model, _, _ = model_artifacts
        _, X_aligned = prepared_data

        e = LimeTabularExplainer(
            training_data=X_aligned.values[:200],
            feature_names=list(X_aligned.columns),
            class_names=["Genuine", "Fraud"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        exp = e.explain_instance(
            data_row=X_aligned.values[0],
            predict_fn=model.predict_proba,
            num_features=5,
            num_samples=200,
            labels=(1,)
        )
        for _, contrib in exp.as_list(label=1):
            assert isinstance(contrib, float), f"Contribution {contrib} is not a float"


# ---------------------------------------------------------------------------
# Deliverable 2: Adverse Action Notice
# ---------------------------------------------------------------------------

SAMPLE_SHAP_TOP3 = [
    {'feature': 'total_claim_amount', 'impact': 0.123},
    {'feature': 'missing_doc_count', 'impact': 0.087},
    {'feature': 'is_odd_hour_incident', 'impact': -0.045},
]
SAMPLE_LIME_TOP3 = [
    ('total_claim_amount > 50000', 0.11),
    ('missing_doc_count > 1', 0.08),
    ('is_odd_hour_incident <= 0', -0.04),
]


class TestAdverseActionNotice:
    def test_notice_generates(self):
        """Notice generation must succeed without raising."""
        notice = generate_adverse_action_notice(
            claim_id="POL-12345", risk_prob=0.73, risk_cat="High",
            shap_top3=SAMPLE_SHAP_TOP3, lime_top3=SAMPLE_LIME_TOP3,
        )
        assert notice is not None and len(notice) > 200

    def test_notice_contains_claim_id(self):
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High", SAMPLE_SHAP_TOP3, SAMPLE_LIME_TOP3)
        assert "POL-12345" in notice

    def test_notice_contains_probability(self):
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High", SAMPLE_SHAP_TOP3, SAMPLE_LIME_TOP3)
        assert "73%" in notice or "73.0%" in notice

    def test_notice_contains_human_review(self):
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High", SAMPLE_SHAP_TOP3, SAMPLE_LIME_TOP3)
        assert "human" in notice.lower()

    def test_notice_contains_year(self):
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High", SAMPLE_SHAP_TOP3, SAMPLE_LIME_TOP3)
        assert str(datetime.date.today().year) in notice

    def test_notice_plain_english_direction(self):
        """Notice must describe factor direction in plain English."""
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High", SAMPLE_SHAP_TOP3, SAMPLE_LIME_TOP3)
        assert "increased" in notice.lower() or "decreased" in notice.lower()

    def test_notice_shap_vs_lime_agreement_note_same_direction(self):
        """When SHAP and LIME top factors agree, notice must say so."""
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High",
            shap_top3=[{'feature': 'total_claim_amount', 'impact': 0.1}],
            lime_top3=[('total_claim_amount > 50000', 0.09)],
        )
        assert "agree" in notice.lower()

    def test_notice_shap_vs_lime_disagreement_note(self):
        """When SHAP and LIME top factors disagree on direction, notice must flag it."""
        notice = generate_adverse_action_notice(
            "POL-12345", 0.73, "High",
            shap_top3=[{'feature': 'total_claim_amount', 'impact': 0.1}],
            lime_top3=[('total_claim_amount > 50000', -0.09)],  # opposite direction
        )
        assert "disagree" in notice.lower()

    def test_notice_no_lime_graceful(self):
        """Notice must work with empty lime_top3."""
        notice = generate_adverse_action_notice(
            "POL-99999", 0.55, "Medium", SAMPLE_SHAP_TOP3, lime_top3=[])
        assert "POL-99999" in notice
        assert len(notice) > 100

    def test_plain_english_factor_total_claim(self):
        result = plain_english_factor("total_claim_amount")
        assert "claim" in result.lower() or "Total" in result

    def test_plain_english_factor_encoded(self):
        """One-hot suffix must be included in the output."""
        result = plain_english_factor("incident_severity_Major")
        assert "Major" in result or "major" in result.lower()

    def test_plain_english_factor_unknown(self):
        """Unknown features must return a cleaned-up string."""
        result = plain_english_factor("some_unknown_feature_x")
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# Deliverable 3: Compliance Report
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def _make_report(self, **overrides):
        defaults = dict(
            dataset_shape=(1000, 39),
            fraud_dist={'Y': 247, 'N': 753},
            model_metrics={"Precision": "0.82", "Recall": "0.75", "F1": "0.78", "ROC-AUC": "0.89"},
            shap_global_text="Top features: total_claim_amount",
            proxy_audit_text="insured_sex is a protected attribute.",
            interaction_text="r=0.82 (total_claim_amount, vehicle_claim_ratio)",
        )
        defaults.update(overrides)
        return build_compliance_report_html(**defaults)

    def test_report_generates(self):
        report = self._make_report()
        assert report is not None and len(report) > 500

    def test_report_is_html(self):
        report = self._make_report()
        assert "<!DOCTYPE html>" in report
        assert "</html>" in report

    def test_report_has_disclaimer(self):
        report = self._make_report()
        assert "disclaimer" in report.lower() or "academic" in report.lower()

    def test_report_proxy_text_included(self):
        report = self._make_report(proxy_audit_text="UNIQUE_PROXY_TEXT_XYZ")
        assert "UNIQUE_PROXY_TEXT_XYZ" in report

    def test_report_custom_metrics_reflected(self):
        report = self._make_report(model_metrics={"Precision": "0.999", "Recall": "0.888"})
        assert "0.999" in report
        assert "0.888" in report

    def test_report_no_metrics_graceful(self):
        """Report with empty model_metrics must still generate."""
        report = self._make_report(model_metrics={})
        assert "Upload labeled data" in report  # fallback message

    def test_report_shap_image_embedded(self):
        """When a base64 image is supplied, it must appear in the HTML."""
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        report = self._make_report(global_shap_img_b64=fake_b64)
        assert fake_b64 in report

    def test_report_contains_dataset_stats(self):
        report = self._make_report()
        assert "1,000" in report or "1000" in report  # total records
        assert "247" in report  # fraud count


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, model_artifacts, raw_data, prepared_data):
        """Full pipeline: load → preprocess → predict → SHAP → top-3 → notice."""
        import shap as shap_lib

        model, feature_columns, _ = model_artifacts
        df_feat, X_aligned = prepared_data

        # Score all claims
        probs = model.predict_proba(X_aligned)[:, 1]
        assert len(probs) == len(raw_data)

        # SHAP on first 10 rows
        explainer = shap_lib.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_aligned.head(10))
        if isinstance(raw_shap, list):
            sv = raw_shap[1]
        else:
            sv = raw_shap[:, :, 1] if raw_shap.ndim == 3 else raw_shap

        # Top-3 for row 0
        top3 = (pd.DataFrame({'feature': X_aligned.columns, 'impact': sv[0]})
                  .sort_values('impact', key=abs, ascending=False)
                  .head(3)
                  .to_dict('records'))

        # Notice
        cat = risk_category(probs[0])
        notice = generate_adverse_action_notice(
            claim_id="E2E-001", risk_prob=probs[0], risk_cat=cat,
            shap_top3=top3, lime_top3=[],
        )
        assert "E2E-001" in notice
        assert len(notice) > 100

        # Compliance report
        report = build_compliance_report_html(
            dataset_shape=df_feat.shape,
            fraud_dist={'Y': 247, 'N': 753},
            model_metrics={},
            shap_global_text="Top feature: " + top3[0]['feature'],
            proxy_audit_text="Proxy check complete.",
            interaction_text="N/A",
        )
        assert "<!DOCTYPE html>" in report
