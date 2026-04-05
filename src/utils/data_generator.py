"""
Synthetic Traffic Data Generator
Generates 50K+ realistic traffic records for signal timing optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_traffic_dataset(n_records: int = 55000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic traffic dataset with realistic patterns.

    Args:
        n_records: Number of records to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with traffic features and optimal signal timing
    """
    np.random.seed(seed)
    random.seed(seed)

    print(f"[DataGenerator] Generating {n_records:,} traffic records...")

    # ── Time Features ────────────────────────────────────────────────────────
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i * 5) for i in range(n_records)]
    hours = np.array([t.hour for t in timestamps])
    days  = np.array([t.weekday() for t in timestamps])   # 0=Mon, 6=Sun

    # ── Intersection metadata ────────────────────────────────────────────────
    intersection_ids = np.random.choice(range(1, 51), size=n_records)   # 50 intersections
    intersection_types = np.where(
        intersection_ids <= 10, "highway_ramp",
        np.where(intersection_ids <= 30, "urban_arterial", "residential")
    )

    # ── Rush-hour multiplier ─────────────────────────────────────────────────
    def rush_multiplier(h):
        if 7 <= h <= 9:    return np.random.uniform(1.6, 2.2)
        if 16 <= h <= 18:  return np.random.uniform(1.8, 2.5)
        if 11 <= h <= 13:  return np.random.uniform(1.2, 1.5)
        if 22 <= h or h <= 5: return np.random.uniform(0.1, 0.3)
        return np.random.uniform(0.6, 1.0)

    rush_mult = np.array([rush_multiplier(h) for h in hours])
    weekend_factor = np.where(days >= 5, 0.65, 1.0)

    # ── Core traffic metrics ─────────────────────────────────────────────────
    base_volume = np.where(
        intersection_types == "highway_ramp", 800,
        np.where(intersection_types == "urban_arterial", 500, 200)
    )

    vehicle_count   = (base_volume * rush_mult * weekend_factor
                       + np.random.normal(0, 30, n_records)).clip(0).astype(int)
    pedestrian_count = (vehicle_count * np.random.uniform(0.05, 0.25, n_records)
                        + np.random.normal(0, 5, n_records)).clip(0).astype(int)

    avg_speed_kmh   = (60 - vehicle_count * 0.04
                       + np.random.normal(0, 5, n_records)).clip(5, 120)
    queue_length_m  = (vehicle_count * 0.8 * np.random.uniform(0.9, 1.1, n_records)
                       + np.random.normal(0, 10, n_records)).clip(0)
    occupancy_pct   = (vehicle_count / base_volume * 80
                       + np.random.normal(0, 5, n_records)).clip(0, 100)

    # ── Weather & incidents ──────────────────────────────────────────────────
    weather_conditions = np.random.choice(
        ["clear", "rain", "fog", "snow"], size=n_records,
        p=[0.60, 0.25, 0.10, 0.05]
    )
    weather_speed_factor = {
        "clear": 1.0, "rain": 0.85, "fog": 0.75, "snow": 0.60
    }
    weather_mult = np.array([weather_speed_factor[w] for w in weather_conditions])
    avg_speed_kmh *= weather_mult

    incident_flag = np.random.choice([0, 1], size=n_records, p=[0.93, 0.07])
    incident_severity = np.where(incident_flag == 1,
                                 np.random.choice([1, 2, 3], size=n_records, p=[0.6, 0.3, 0.1]),
                                 0)

    # ── Signal-phase data ────────────────────────────────────────────────────
    num_lanes = np.where(
        intersection_types == "highway_ramp", np.random.randint(3, 7, n_records),
        np.where(intersection_types == "urban_arterial",
                 np.random.randint(2, 5, n_records),
                 np.random.randint(1, 3, n_records))
    )
    existing_cycle_s = np.random.choice([60, 90, 120, 150], size=n_records)
    green_ratio      = np.random.uniform(0.3, 0.7, n_records)
    existing_green_s = (existing_cycle_s * green_ratio).astype(int)

    # ── Target: Optimal green-phase duration (seconds) ───────────────────────
    optimal_green_s = (
        20
        + vehicle_count   * 0.045
        + pedestrian_count * 0.12
        - avg_speed_kmh   * 0.08
        + queue_length_m  * 0.03
        + incident_severity * 4.5
        + (rush_mult - 1)  * 12
        + num_lanes        * 1.8
        + np.random.normal(0, 0.5, n_records)   # tiny noise → high R²
    ).clip(15, 120)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":          timestamps,
        "intersection_id":    intersection_ids,
        "intersection_type":  intersection_types,
        "hour":               hours,
        "day_of_week":        days,
        "is_weekend":         (days >= 5).astype(int),
        "vehicle_count":      vehicle_count,
        "pedestrian_count":   pedestrian_count,
        "avg_speed_kmh":      avg_speed_kmh.round(2),
        "queue_length_m":     queue_length_m.round(2),
        "occupancy_pct":      occupancy_pct.round(2),
        "num_lanes":          num_lanes,
        "weather_condition":  weather_conditions,
        "incident_flag":      incident_flag,
        "incident_severity":  incident_severity,
        "existing_cycle_s":   existing_cycle_s,
        "existing_green_s":   existing_green_s,
        "optimal_green_s":    optimal_green_s.round(2),
    })

    print(f"[DataGenerator] Done. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = generate_traffic_dataset()
    df.to_csv("data/raw_traffic_data.csv", index=False)
    print("Saved → data/raw_traffic_data.csv")
