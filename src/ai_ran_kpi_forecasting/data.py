"""Data loading and synthetic telemetry generation."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_ran_kpi_data(
    path: str | Path,
    timestamp_col: str = "timestamp",
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Load generic RAN KPI telemetry from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in data.")
    if cell_id_col not in df.columns:
        raise ValueError(f"Cell ID column '{cell_id_col}' not found in data.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    return df.sort_values([cell_id_col, timestamp_col]).reset_index(drop=True)


def filter_cell(
    df: pd.DataFrame,
    cell_id: str | None,
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Filter to one cell, defaulting to the densest cell."""
    if cell_id_col not in df.columns:
        raise ValueError(f"cell_id_col '{cell_id_col}' not found in data.")

    if cell_id is not None:
        df_cell = df[df[cell_id_col] == cell_id].copy()
        if df_cell.empty:
            raise ValueError(f"No rows found for cell_id='{cell_id}'.")
        return df_cell.reset_index(drop=True)

    counts = df[cell_id_col].value_counts()
    best_cell_id = counts.idxmax()
    return df[df[cell_id_col] == best_cell_id].copy().reset_index(drop=True)


def _normalize_telecom_italia_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "Square id": "cell_id",
        "square_id": "cell_id",
        "Square_id": "cell_id",
        "Time interval": "time_interval",
        "time_interval": "time_interval",
        "Time_interval": "time_interval",
        "Country code": "country_code",
        "country_code": "country_code",
        "SMS-in activity": "sms_in",
        "SMS-out activity": "sms_out",
        "Call-in activity": "call_in",
        "Call-out activity": "call_out",
        "Internet traffic activity": "internet_traffic",
        "sms_in": "sms_in",
        "sms_out": "sms_out",
        "call_in": "call_in",
        "call_out": "call_out",
        "internet_traffic": "internet_traffic",
    }
    df = df.rename(columns=col_map)
    missing = [c for c in ["cell_id", "time_interval"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Telecom Italia columns: {missing}")
    return df


def load_telecom_italia_mi(data_path: str | Path, aggregate: str = "hourly") -> pd.DataFrame:
    """Load Telecom Italia MI files into the generic telemetry schema."""
    data_path = Path(data_path)
    if data_path.is_dir():
        files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_path}")
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        files = [str(data_path)]

    dfs = [_normalize_telecom_italia_columns(pd.read_csv(path)) for path in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "cell_id"])

    agg_cols = [c for c in df.columns if c in {"sms_in", "sms_out", "call_in", "call_out", "internet_traffic"}]
    if not agg_cols:
        raise ValueError("No activity columns found in Telecom Italia data.")

    df = df.groupby(["cell_id", "timestamp"], as_index=False)[agg_cols].sum()
    if aggregate == "10min":
        return df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    if aggregate != "hourly":
        raise ValueError("aggregate must be '10min' or 'hourly'.")

    return (
        df.set_index("timestamp")
        .groupby("cell_id")
        .resample("1h")
        .sum(numeric_only=True)
        .reset_index()
        .sort_values(["cell_id", "timestamp"])
        .reset_index(drop=True)
    )


def generate_synthetic_telemetry(
    cells: int = 3,
    periods: int = 168,
    freq: str = "1h",
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate compact synthetic telecom telemetry for demos and tests."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    rows: list[dict[str, object]] = []

    for cell_idx in range(1, cells + 1):
        cell_id = f"CELL_{cell_idx:03d}"
        phase = rng.uniform(0, np.pi)
        baseline = rng.uniform(35, 55)
        for idx, ts in enumerate(timestamps):
            hour_cycle = np.sin(2 * np.pi * (idx % 24) / 24 + phase)
            weekly_cycle = np.sin(2 * np.pi * idx / (24 * 7) + phase / 2)
            load = baseline + 18 * hour_cycle + 8 * weekly_cycle + rng.normal(0, 3)
            prb_dl_util = float(np.clip(load, 5, 98))
            rrc_users = int(np.clip(80 + prb_dl_util * 2.0 + rng.normal(0, 15), 10, 400))
            throughput = float(np.clip(150 - prb_dl_util * 1.15 + rng.normal(0, 8), 5, 180))
            latency = float(np.clip(12 + prb_dl_util * 0.16 + rng.normal(0, 1.2), 5, 60))
            rows.append(
                {
                    "timestamp": ts,
                    "cell_id": cell_id,
                    "prb_dl_util": round(prb_dl_util, 3),
                    "throughput_mbps": round(throughput, 3),
                    "rrc_users": rrc_users,
                    "latency_ms": round(latency, 3),
                }
            )

    return pd.DataFrame(rows)
