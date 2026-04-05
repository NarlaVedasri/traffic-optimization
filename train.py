"""
Main Training Script — Traffic Signal Optimization
Run:  python train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_generator import generate_traffic_dataset
from src.pipeline.preprocessor import TrafficPreprocessor
from src.pipeline.model import TrafficSignalModel

os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)


def main():
    print("\n" + "=" * 60)
    print("  TRAFFIC SIGNAL OPTIMIZATION — ML TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── Step 1: Data ─────────────────────────────────────────────
    raw_path = "data/raw_traffic_data.csv"
    if os.path.exists(raw_path):
        print("[Main] Loading cached data...")
        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    else:
        df = generate_traffic_dataset(n_records=55000)
        df.to_csv(raw_path, index=False)
        print(f"[Main] Data saved → {raw_path}")

    print(f"[Main] Dataset: {len(df):,} records | "
          f"Intersections: {df['intersection_id'].nunique()} | "
          f"Date range: {df['timestamp'].min().date()} – {df['timestamp'].max().date()}\n")

    # ── Step 2: Preprocessing ─────────────────────────────────────
    preprocessor = TrafficPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # ── Step 3: Model Training ────────────────────────────────────
    model = TrafficSignalModel()
    model.train(X_train, y_train, feature_names=preprocessor.feature_columns)

    # ── Step 4: Evaluation ────────────────────────────────────────
    metrics = model.evaluate(X_test, y_test, X_train, y_train, cv_folds=5)

    # ── Step 5: Feature Importance ────────────────────────────────
    fi_df = model.feature_importance_df()
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    print(f"\nTop 10 Features:\n{fi_df.head(10).to_string(index=False)}\n")

    # ── Step 6: Visualisations ────────────────────────────────────
    y_pred = model.predict(X_test)
    _plot_results(y_test, y_pred, fi_df, metrics)

    print("\n[Main] Pipeline complete. Artifacts in models/ and outputs/\n")
    return metrics


def _plot_results(y_test, y_pred, fi_df, metrics):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ACCENT = "#00d4aa"
    PINK   = "#ff6b9d"
    TEXT   = "#e6edf3"
    GRID   = "#21262d"

    def style_ax(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test[:3000], y_pred[:3000], alpha=0.3, s=4, color=ACCENT)
    mn, mx = y_test.min(), y_test.max()
    ax1.plot([mn, mx], [mn, mx], "--", color=PINK, linewidth=1.5)
    style_ax(ax1, "Actual vs Predicted")
    ax1.set_xlabel("Actual (s)", color=TEXT, fontsize=8)
    ax1.set_ylabel("Predicted (s)", color=TEXT, fontsize=8)

    # 2. Residual distribution
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=60, color=ACCENT, alpha=0.8, edgecolor="none")
    ax2.axvline(0, color=PINK, linestyle="--", linewidth=1.5)
    style_ax(ax2, "Residual Distribution")
    ax2.set_xlabel("Residual (s)", color=TEXT, fontsize=8)
    ax2.set_ylabel("Count", color=TEXT, fontsize=8)

    # 3. Feature Importance (top 12)
    ax3 = fig.add_subplot(gs[0, 2])
    top = fi_df.head(12)
    bars = ax3.barh(top["feature"][::-1], top["importance_pct"][::-1],
                    color=ACCENT, alpha=0.85, edgecolor="none")
    style_ax(ax3, "Feature Importance (%)")
    ax3.set_xlabel("Importance %", color=TEXT, fontsize=8)

    # 4. Prediction error over time (binned)
    ax4 = fig.add_subplot(gs[1, 0])
    abs_err = np.abs(residuals)
    bins = np.array_split(abs_err, 50)
    bin_means = [b.mean() for b in bins]
    ax4.plot(bin_means, color=ACCENT, linewidth=1.5)
    ax4.fill_between(range(len(bin_means)), bin_means, alpha=0.2, color=ACCENT)
    style_ax(ax4, "MAE across Test Batches")
    ax4.set_xlabel("Batch", color=TEXT, fontsize=8)
    ax4.set_ylabel("MAE (s)", color=TEXT, fontsize=8)

    # 5. Metrics card
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#161b22")
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.axis("off")
    ax5.set_title("Evaluation Metrics", color=TEXT, fontsize=11, fontweight="bold")
    kv = [
        ("R² Score",          f"{metrics['r2_score']:.4f}"),
        ("MAE",               f"{metrics['mae_seconds']:.3f} s"),
        ("RMSE",              f"{metrics['rmse_seconds']:.3f} s"),
        ("MAPE",              f"{metrics['mape_pct']:.2f}%"),
        ("OOB Score",         f"{metrics['oob_score']:.4f}"),
        ("CV R² (5-fold)",    f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"),
    ]
    for i, (k, v) in enumerate(kv):
        y_pos = 0.88 - i * 0.14
        ax5.text(0.05, y_pos, k,  color="#8b949e", fontsize=9)
        ax5.text(0.95, y_pos, v, color=ACCENT, fontsize=10,
                 fontweight="bold", ha="right")

    # 6. Predicted timing distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(y_pred, bins=40, color=PINK, alpha=0.7, label="Predicted", edgecolor="none")
    ax6.hist(y_test, bins=40, color=ACCENT, alpha=0.5, label="Actual", edgecolor="none")
    style_ax(ax6, "Timing Distribution (s)")
    ax6.set_xlabel("Green Phase (s)", color=TEXT, fontsize=8)
    ax6.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    fig.suptitle("Traffic Signal Optimization — Model Results",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.98)

    out = "outputs/model_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Main] Plot saved → {out}")


if __name__ == "__main__":
    main()
