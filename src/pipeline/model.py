"""
Random Forest Regressor — Traffic Signal Timing Optimizer
Trains, evaluates, and persists the ML model.
"""

import numpy as np
import pandas as pd
import json
import os
import time
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score


class TrafficSignalModel:
    """
    Random Forest Regressor for predicting optimal green-phase duration.

    Design:
        - Tuned hyperparameters for signal timing regression
        - Supports feature importance analysis
        - Cross-validation for robust evaluation
        - Full model persistence via joblib
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model = RandomForestRegressor(
            n_estimators      = 200,
            max_depth         = None,          # fully grown trees
            min_samples_split = 2,
            min_samples_leaf  = 1,
            max_features      = "sqrt",
            bootstrap         = True,
            n_jobs            = -1,            # all CPU cores
            random_state      = 42,
            oob_score         = True,
        )
        self.metrics: dict       = {}
        self.feature_names: list = []

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "TrafficSignalModel":
        """Fit the model and record OOB score."""
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        print(f"[Model] Training RandomForest on {X_train.shape[0]:,} samples "
              f"({X_train.shape[1]} features)...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"[Model] Trained in {elapsed:.1f}s | OOB R²: {self.model.oob_score_:.6f}")
        return self

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        cv_folds: int = 5,
    ) -> dict:
        """Compute full evaluation metrics."""
        y_pred = self.model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        evs  = explained_variance_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1, None))) * 100

        # Cross-validation on training set (if provided)
        cv_scores = []
        if X_train is not None and y_train is not None:
            print(f"[Model] Running {cv_folds}-fold CV...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=cv_folds,
                scoring="r2", n_jobs=-1
            )
            print(f"[Model] CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.metrics = {
            "r2_score":             round(r2,  6),
            "mae_seconds":          round(mae,  4),
            "rmse_seconds":         round(rmse, 4),
            "mape_pct":             round(mape, 4),
            "explained_variance":   round(evs,  6),
            "oob_score":            round(self.model.oob_score_, 6),
            "cv_r2_mean":           round(float(np.mean(cv_scores)), 4) if len(cv_scores) else None,
            "cv_r2_std":            round(float(np.std(cv_scores)),  4) if len(cv_scores) else None,
            "n_test_samples":       len(y_test),
        }

        self._print_metrics()
        self.save_metrics()
        return self.metrics

    def _print_metrics(self):
        print("\n" + "=" * 50)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 50)
        for k, v in self.metrics.items():
            print(f"  {k:<28} {v}")
        print("=" * 50 + "\n")

    # ── Feature Importance ───────────────────────────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        imp = self.model.feature_importances_
        df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_pct"] = (df["importance"] * 100).round(3)
        return df

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).clip(15, 120)

    def predict_with_ci(
        self, X: np.ndarray, percentile: float = 90
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (mean_prediction, lower_bound, upper_bound) using
        individual tree predictions as a proxy for confidence.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        lo = (100 - percentile) / 2
        hi = 100 - lo
        return (
            tree_preds.mean(axis=0).clip(15, 120),
            np.percentile(tree_preds, lo, axis=0).clip(15, 120),
            np.percentile(tree_preds, hi, axis=0).clip(15, 120),
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        joblib.dump(self.model, f"{self.model_dir}/random_forest_model.pkl")
        joblib.dump(self.feature_names, f"{self.model_dir}/feature_names.pkl")
        print(f"[Model] Saved → {self.model_dir}/random_forest_model.pkl")

    def load(self):
        self.model         = joblib.load(f"{self.model_dir}/random_forest_model.pkl")
        self.feature_names = joblib.load(f"{self.model_dir}/feature_names.pkl")
        print(f"[Model] Loaded ← {self.model_dir}/random_forest_model.pkl")
        return self

    def save_metrics(self):
        path = f"{self.model_dir}/metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[Model] Metrics saved → {path}")
