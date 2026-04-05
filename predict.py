"""
Inference Script — Predict optimal signal timing for new intersections.
Usage:
    python predict.py --input data/new_intersections.csv
    python predict.py --demo   (runs with synthetic demo data)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_generator import generate_traffic_dataset
from src.pipeline.preprocessor import TrafficPreprocessor
from src.pipeline.model import TrafficSignalModel


def load_artifacts():
    preprocessor = TrafficPreprocessor()
    preprocessor.load()
    model = TrafficSignalModel()
    model.load()
    return preprocessor, model


def predict(input_df: pd.DataFrame) -> pd.DataFrame:
    preprocessor, model = load_artifacts()
    X = preprocessor.transform(input_df)
    mean_pred, lo, hi = model.predict_with_ci(X, percentile=90)

    result = input_df[["intersection_id", "intersection_type",
                        "vehicle_count", "hour"]].copy()
    result["predicted_green_s"]  = mean_pred.round(1)
    result["ci_lower_90"]        = lo.round(1)
    result["ci_upper_90"]        = hi.round(1)
    result["time_savings_s"]     = (
        input_df["existing_green_s"].values - mean_pred
    ).round(1)

    return result


def demo():
    print("[Predict] Running demo with synthetic data...")
    df = generate_traffic_dataset(n_records=500, seed=99)
    result = predict(df)
    print("\nSample Predictions:")
    print(result.head(10).to_string(index=False))
    out = "outputs/predictions_demo.csv"
    result.to_csv(out, index=False)
    print(f"\n[Predict] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input CSV")
    parser.add_argument("--demo",  action="store_true", help="Run with demo data")
    args = parser.parse_args()

    if args.demo or not args.input:
        demo()
    else:
        df = pd.read_csv(args.input, parse_dates=["timestamp"])
        result = predict(df)
        out = "outputs/predictions.csv"
        result.to_csv(out, index=False)
        print(f"Predictions saved → {out}")
