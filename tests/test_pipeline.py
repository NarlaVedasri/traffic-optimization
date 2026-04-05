"""
Unit Tests — Traffic Optimization System
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_generator import generate_traffic_dataset
from src.pipeline.preprocessor import TrafficPreprocessor
from src.pipeline.model import TrafficSignalModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return generate_traffic_dataset(n_records=2000, seed=0)


@pytest.fixture(scope="module")
def pipeline(raw_df):
    prep = TrafficPreprocessor(artifacts_dir="/tmp/traffic_test_artifacts")
    X_train, X_test, y_train, y_test = prep.fit_transform(raw_df.copy())
    model = TrafficSignalModel(model_dir="/tmp/traffic_test_model")
    model.train(X_train, y_train, feature_names=prep.feature_columns)
    return prep, model, X_train, X_test, y_train, y_test


# ── Data Generator Tests ──────────────────────────────────────────────────────

class TestDataGenerator:
    def test_shape(self, raw_df):
        assert len(raw_df) == 2000

    def test_required_columns(self, raw_df):
        required = ["vehicle_count", "pedestrian_count", "avg_speed_kmh",
                    "queue_length_m", "optimal_green_s", "intersection_type"]
        for col in required:
            assert col in raw_df.columns, f"Missing: {col}"

    def test_target_range(self, raw_df):
        assert raw_df["optimal_green_s"].between(15, 120).all()

    def test_no_nulls_in_target(self, raw_df):
        assert raw_df["optimal_green_s"].isna().sum() == 0

    def test_intersection_types(self, raw_df):
        valid = {"highway_ramp", "urban_arterial", "residential"}
        assert set(raw_df["intersection_type"].unique()).issubset(valid)


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_output_shapes(self, pipeline):
        _, _, X_train, X_test, y_train, y_test = pipeline
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0]  == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_no_nans_after_preprocessing(self, pipeline):
        _, _, X_train, X_test, _, _ = pipeline
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_feature_engineering_columns(self, pipeline):
        prep = pipeline[0]
        expected_features = ["hour_sin", "hour_cos", "congestion_index",
                             "vehicle_per_lane", "mixed_traffic"]
        for f in expected_features:
            assert f in prep.feature_columns, f"Missing engineered feature: {f}"

    def test_scaled_range(self, pipeline):
        """Scaled values should be roughly standardised."""
        _, _, X_train, _, _, _ = pipeline
        assert abs(X_train.mean()) < 1.0       # near zero mean
        assert 0.5 < X_train.std() < 2.0       # near unit variance


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModel:
    def test_r2_above_threshold(self, pipeline):
        prep, model, _, X_test, _, y_test = pipeline
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.99, f"R² too low: {r2:.4f}"

    def test_prediction_range(self, pipeline):
        prep, model, _, X_test, _, _ = pipeline
        y_pred = model.predict(X_test)
        assert y_pred.min() >= 15
        assert y_pred.max() <= 120

    def test_prediction_with_ci(self, pipeline):
        prep, model, _, X_test, _, _ = pipeline
        mean, lo, hi = model.predict_with_ci(X_test[:50])
        assert mean.shape == lo.shape == hi.shape == (50,)
        assert (hi >= mean).all()
        assert (mean >= lo).all()

    def test_feature_importance_sums_to_one(self, pipeline):
        _, model, _, _, _, _ = pipeline
        fi = model.feature_importance_df()
        total = fi["importance"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_metrics_keys(self, pipeline):
        prep, model, _, X_test, _, y_test = pipeline
        metrics = model.evaluate(X_test, y_test)
        for key in ["r2_score", "mae_seconds", "rmse_seconds", "oob_score"]:
            assert key in metrics


# ── Integration Test ──────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline(self):
        df   = generate_traffic_dataset(n_records=500, seed=7)
        prep = TrafficPreprocessor(artifacts_dir="/tmp/e2e_artifacts")
        model = TrafficSignalModel(model_dir="/tmp/e2e_model")

        X_train, X_test, y_train, y_test = prep.fit_transform(df)
        model.train(X_train, y_train, feature_names=prep.feature_columns)
        metrics = model.evaluate(X_test, y_test)

        assert metrics["r2_score"] > 0.99
        assert metrics["mae_seconds"] < 2.0
