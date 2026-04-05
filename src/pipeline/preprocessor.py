"""
Traffic Data Preprocessing Pipeline
Handles encoding, scaling, and feature engineering for the ML model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class TrafficPreprocessor:
    """
    Modular preprocessing pipeline for traffic signal timing data.

    Steps:
        1. Clean & validate raw data
        2. Feature engineering (time-based, interaction, lag features)
        3. Categorical encoding
        4. Feature scaling
    """

    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []
        self.is_fitted = False

    # ── 1. Cleaning ──────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, handle nulls, clip outliers."""
        print("[Preprocessor] Cleaning data...")
        df = df.drop_duplicates()
        df = df.dropna(subset=["optimal_green_s"])

        # Clip numerical outliers at 1st/99th percentile
        num_cols = ["vehicle_count", "pedestrian_count",
                    "avg_speed_kmh", "queue_length_m", "occupancy_pct"]
        for col in num_cols:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

        print(f"[Preprocessor] Clean shape: {df.shape}")
        return df

    # ── 2. Feature Engineering ───────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based, interaction, and ratio features."""
        print("[Preprocessor] Engineering features...")
        df = df.copy()

        # Cyclical time encoding (preserves periodicity)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Rush-hour binary flag
        df["is_morning_rush"] = df["hour"].between(7,  9).astype(int)
        df["is_evening_rush"] = df["hour"].between(16, 18).astype(int)
        df["is_night"]        = (~df["hour"].between(6, 22)).astype(int)

        # Traffic density per lane
        df["vehicle_per_lane"] = df["vehicle_count"] / df["num_lanes"].clip(lower=1)

        # Congestion index (0–1)
        max_possible = df["vehicle_count"].max() + 1
        df["congestion_index"] = (
            df["vehicle_count"] / max_possible * 0.5
            + df["occupancy_pct"] / 100 * 0.3
            + df["queue_length_m"] / df["queue_length_m"].max() * 0.2
        ).clip(0, 1)

        # Mixed-traffic pressure
        df["mixed_traffic"] = df["vehicle_count"] + df["pedestrian_count"] * 2

        # Interaction: incident × volume
        df["incident_volume"] = df["incident_severity"] * df["vehicle_count"]

        # Speed deviation from limit (assumed 60 km/h urban)
        df["speed_deviation"] = (60 - df["avg_speed_kmh"]).clip(lower=0)

        # Timing efficiency of existing signal
        df["current_efficiency"] = (
            df["existing_green_s"] / df["existing_cycle_s"].clip(lower=1)
        )

        print(f"[Preprocessor] Features engineered. Shape: {df.shape}")
        return df

    # ── 3. Encoding ──────────────────────────────────────────────────────────

    def encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        print("[Preprocessor] Encoding categoricals...")
        cat_cols = ["intersection_type", "weather_condition"]
        df = df.copy()

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))

        return df

    # ── 4. Scaling ───────────────────────────────────────────────────────────

    def scale(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """StandardScale numerical features."""
        print("[Preprocessor] Scaling features...")
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    # ── Full pipeline ────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "optimal_green_s"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run complete preprocessing and return train/test splits.

        Returns:
            X_train, X_test, y_train, y_test (all numpy arrays)
        """
        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.encode(df, fit=True)

        # Drop non-feature columns
        drop_cols = [target_col, "timestamp", "intersection_id"]
        self.feature_columns = [c for c in df.columns if c not in drop_cols]

        X = df[self.feature_columns]
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scale(X_train, fit=True)
        X_test_s  = self.scale(X_test,  fit=False)

        self.is_fitted = True
        self.save()

        print(f"[Preprocessor] X_train: {X_train_s.shape}, X_test: {X_test_s.shape}")
        return X_train_s, X_test_s, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data (must be fitted first)."""
        assert self.is_fitted, "Call fit_transform first."
        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.encode(df, fit=False)
        X  = df[self.feature_columns]
        return self.scale(X, fit=False)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        joblib.dump(self.scaler,          f"{self.artifacts_dir}/scaler.pkl")
        joblib.dump(self.label_encoders,  f"{self.artifacts_dir}/label_encoders.pkl")
        joblib.dump(self.feature_columns, f"{self.artifacts_dir}/feature_columns.pkl")
        print(f"[Preprocessor] Artifacts saved → {self.artifacts_dir}/")

    def load(self):
        self.scaler          = joblib.load(f"{self.artifacts_dir}/scaler.pkl")
        self.label_encoders  = joblib.load(f"{self.artifacts_dir}/label_encoders.pkl")
        self.feature_columns = joblib.load(f"{self.artifacts_dir}/feature_columns.pkl")
        self.is_fitted = True
        print(f"[Preprocessor] Artifacts loaded ← {self.artifacts_dir}/")
