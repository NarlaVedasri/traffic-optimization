# 🚦 Data-Driven Traffic Signal Optimization System

> **Summer 2024** | ML · Python · SQL · Random Forest · Feature Engineering

An end-to-end machine learning system that processes **50,000+ traffic records** to optimize signal timing at intersections, achieving **R² ≈ 0.9999** using a tuned Random Forest Regressor.

---

## 📊 Results

| Metric | Value |
|---|---|
| R² Score | **0.9999** |
| MAE | ~0.3 seconds |
| RMSE | ~0.4 seconds |
| OOB Score | ~0.9998 |
| CV R² (5-fold) | 0.9998 ± 0.0001 |


traffic-optimization/outputs/model_results.png



---

## 🏗️ Project Structure

```
traffic-optimization/
│
├── src/
│   ├── pipeline/
│   │   ├── preprocessor.py      # Encoding, scaling, feature engineering
│   │   └── model.py             # Random Forest training + evaluation
│   └── utils/
│       └── data_generator.py    # Synthetic 55K-record dataset generator
│
├── notebooks/
│   └── eda_and_analysis.ipynb   # EDA, feature importance, visualisations
│
├── tests/
│   └── test_pipeline.py         # Unit + integration tests (pytest)
│
├── data/                        # Raw CSV data (generated on first run)
├── models/                      # Saved model artifacts (.pkl)
├── outputs/                     # Plots, metrics, predictions
│
├── train.py                     # ▶ Main training script
├── predict.py                   # ▶ Inference on new data
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/traffic-optimization.git
cd traffic-optimization
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Generate **55,000 traffic records** across 50 intersections
- Run the full preprocessing pipeline (encoding + scaling + feature engineering)
- Train the Random Forest Regressor
- Output metrics, plots, and saved artifacts

### 3. Run Predictions

```bash
# Demo mode (uses synthetic data)
python predict.py --demo

# Custom data
python predict.py --input data/my_intersections.csv
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Open the Notebook

```bash
cd notebooks
jupyter notebook eda_and_analysis.ipynb
```

---

## ⚙️ ML Pipeline

### Data
- **55,000 records** across 50 intersections (3 types: highway ramp, urban arterial, residential)
- Time range: full year with 5-minute granularity
- Features: vehicle count, pedestrian count, speed, queue length, weather, incidents

### Feature Engineering (`preprocessor.py`)
| Feature | Description |
|---|---|
| `hour_sin / hour_cos` | Cyclical encoding of hour of day |
| `dow_sin / dow_cos` | Cyclical encoding of day of week |
| `is_morning_rush` | Binary flag: 7–9 AM |
| `is_evening_rush` | Binary flag: 4–6 PM |
| `congestion_index` | Composite score (volume + occupancy + queue) |
| `vehicle_per_lane` | Traffic density per lane |
| `mixed_traffic` | Vehicles + weighted pedestrian count |
| `incident_volume` | Interaction: severity × vehicle count |
| `speed_deviation` | Delta from 60 km/h speed limit |
| `current_efficiency` | Existing green/cycle ratio |

### Model (`model.py`)
- **Algorithm**: `sklearn.ensemble.RandomForestRegressor`
- `n_estimators=200`, `max_features='sqrt'`, `oob_score=True`
- Cross-validation: 5-fold CV on training set
- Confidence intervals via individual tree predictions

---

## 📈 Outputs

| File | Description |
|---|---|
| `outputs/model_results.png` | 6-panel evaluation dashboard |
| `outputs/feature_importance.csv` | Ranked feature importances |
| `outputs/predictions_demo.csv` | Sample predictions with 90% CI |
| `models/random_forest_model.pkl` | Saved model |
| `models/metrics.json` | Evaluation metrics |
| `models/artifacts/scaler.pkl` | Fitted scaler |


Here is the model prediction result:

![Prediction Output](c:\Users\narla\Downloads\traffic-optimization\traffic-optimization\outputs\model_results.png)

---

## 🧠 Key Design Decisions

- **Modular SDLC pipeline**: Clean separation of data, preprocessing, model, and inference
- **Cyclical time encoding**: Avoids discontinuity at midnight/week boundaries
- **OOB scoring**: Free validation without a separate hold-out during training
- **Confidence intervals**: Individual tree predictions used as a proxy ensemble distribution
- **Reproducibility**: All random seeds fixed; artifacts persisted via joblib

---

## 📋 Requirements

- Python ≥ 3.10
- scikit-learn ≥ 1.4
- pandas ≥ 2.1
- numpy ≥ 1.26
- matplotlib ≥ 3.8

---
👩‍💻 Created By

Vedasri Narla

Feel free to ⭐ star or fork the project if you found it interesting!

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
