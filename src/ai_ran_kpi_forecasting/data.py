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
        site_id = f"SITE_{(cell_idx - 1) // 3 + 1:02d}"
        sector_id = f"{site_id}_SEC_{((cell_idx - 1) % 3) + 1}"
        phase = rng.uniform(0, np.pi)
        baseline = rng.uniform(35, 55)
        for idx, ts in enumerate(timestamps):
            hour_cycle = np.sin(2 * np.pi * (idx % 24) / 24 + phase)
            weekly_cycle = np.sin(2 * np.pi * idx / (24 * 7) + phase / 2)
            load = baseline + 18 * hour_cycle + 8 * weekly_cycle + rng.normal(0, 3)
            prb_dl_util = float(np.clip(load, 5, 98))
            prb_ul_util = float(np.clip(prb_dl_util * 0.62 + rng.normal(0, 4), 4, 95))
            rrc_users = int(np.clip(80 + prb_dl_util * 2.0 + rng.normal(0, 15), 10, 400))
            throughput_dl = float(np.clip(180 - prb_dl_util * 1.35 + rng.normal(0, 8), 5, 220))
            throughput_ul = float(np.clip(90 - prb_ul_util * 0.95 + rng.normal(0, 5), 2, 120))
            latency = float(np.clip(12 + prb_dl_util * 0.16 + rng.normal(0, 1.2), 5, 60))
            packet_loss = float(np.clip((prb_dl_util - 55) * 0.05 + rng.normal(0, 0.2), 0, 5))
            sinr = float(np.clip(24 - prb_dl_util * 0.08 + rng.normal(0, 0.9), 1, 30))
            rows.append(
                {
                    "timestamp": ts,
                    "cell_id": cell_id,
                    "site_id": site_id,
                    "sector_id": sector_id,
                    "technology": "5G NSA",
                    "band": "n78",
                    "prb_dl_util": round(prb_dl_util, 3),
                    "prb_ul_util": round(prb_ul_util, 3),
                    "throughput_dl_mbps": round(throughput_dl, 3),
                    "throughput_ul_mbps": round(throughput_ul, 3),
                    "rrc_users": rrc_users,
                    "latency_ms": round(latency, 3),
                    "packet_loss_pct": round(packet_loss, 3),
                    "sinr_db": round(sinr, 3),
                }
            )

    return pd.DataFrame(rows)


def generate_congestion_telemetry(
    cells: int = 3,
    periods: int = 168,
    freq: str = "1h",
    seed: int = 42,
    start: str = "2024-01-01",
    congested_cell: str = "CELL_001",
    shock_start: float = 0.62,
    shock_duration: int = 18,
) -> pd.DataFrame:
    """Generate a synthetic congestion event for before/after reporting."""
    df = generate_synthetic_telemetry(cells=cells, periods=periods, freq=freq, seed=seed, start=start)
    if congested_cell not in set(df["cell_id"]):
        congested_cell = df["cell_id"].iloc[0]

    cell_mask = df["cell_id"] == congested_cell
    cell_df = df.loc[cell_mask].copy()
    shock_idx = int(len(cell_df) * shock_start)
    shock_idx = max(0, min(shock_idx, max(len(cell_df) - 1, 0)))
    end_idx = min(len(cell_df), shock_idx + shock_duration)

    if end_idx > shock_idx:
        shock_slice = cell_df.index[shock_idx:end_idx]
        ramp = np.linspace(0.15, 1.0, len(shock_slice))
        df.loc[shock_slice, "prb_dl_util"] = np.clip(df.loc[shock_slice, "prb_dl_util"] + 22 * ramp, 5, 99)
        df.loc[shock_slice, "prb_ul_util"] = np.clip(df.loc[shock_slice, "prb_ul_util"] + 12 * ramp, 4, 99)
        df.loc[shock_slice, "throughput_dl_mbps"] = np.clip(df.loc[shock_slice, "throughput_dl_mbps"] - 38 * ramp, 2, None)
        df.loc[shock_slice, "throughput_ul_mbps"] = np.clip(df.loc[shock_slice, "throughput_ul_mbps"] - 16 * ramp, 1, None)
        df.loc[shock_slice, "latency_ms"] = np.clip(df.loc[shock_slice, "latency_ms"] + 9 * ramp, 5, None)
        df.loc[shock_slice, "packet_loss_pct"] = np.clip(df.loc[shock_slice, "packet_loss_pct"] + 1.8 * ramp, 0, 12)
        df.loc[shock_slice, "sinr_db"] = np.clip(df.loc[shock_slice, "sinr_db"] - 4.5 * ramp, 1, 30)
        df.loc[shock_slice, "rrc_users"] = np.clip(df.loc[shock_slice, "rrc_users"] + (30 * ramp).astype(int), 10, 500)

    return df.reset_index(drop=True)


def generate_backhaul_telemetry(
    cells: int = 3,
    periods: int = 168,
    freq: str = "1h",
    seed: int = 42,
    start: str = "2024-01-01",
    affected_cell: str = "CELL_001",
    shock_start: float = 0.58,
    shock_duration: int = 20,
) -> pd.DataFrame:
    """Generate a synthetic backhaul saturation event for dashboard evidence."""
    df = generate_synthetic_telemetry(cells=cells, periods=periods, freq=freq, seed=seed, start=start)
    if affected_cell not in set(df["cell_id"]):
        affected_cell = df["cell_id"].iloc[0]

    cell_df = df.loc[df["cell_id"] == affected_cell].copy()
    shock_idx = int(len(cell_df) * shock_start)
    shock_idx = max(0, min(shock_idx, max(len(cell_df) - 1, 0)))
    end_idx = min(len(cell_df), shock_idx + shock_duration)

    if end_idx > shock_idx:
        shock_slice = cell_df.index[shock_idx:end_idx]
        ramp = np.linspace(0.25, 1.0, len(shock_slice))
        df.loc[shock_slice, "throughput_dl_mbps"] = np.clip(df.loc[shock_slice, "throughput_dl_mbps"] - 48 * ramp, 1, None)
        df.loc[shock_slice, "throughput_ul_mbps"] = np.clip(df.loc[shock_slice, "throughput_ul_mbps"] - 22 * ramp, 1, None)
        df.loc[shock_slice, "latency_ms"] = np.clip(df.loc[shock_slice, "latency_ms"] + 14 * ramp, 5, None)
        df.loc[shock_slice, "packet_loss_pct"] = np.clip(df.loc[shock_slice, "packet_loss_pct"] + 2.4 * ramp, 0, 15)
        df.loc[shock_slice, "sinr_db"] = np.clip(df.loc[shock_slice, "sinr_db"] - 2.5 * ramp, 1, 30)
        df.loc[shock_slice, "prb_dl_util"] = np.clip(df.loc[shock_slice, "prb_dl_util"] + 4 * ramp, 5, 99)
        df.loc[shock_slice, "prb_ul_util"] = np.clip(df.loc[shock_slice, "prb_ul_util"] + 3 * ramp, 4, 99)

    return df.reset_index(drop=True)


def generate_outage_telemetry(
    cells: int = 3,
    periods: int = 168,
    freq: str = "1h",
    seed: int = 42,
    start: str = "2024-01-01",
    affected_cell: str = "CELL_001",
    shock_start: float = 0.66,
    shock_duration: int = 14,
) -> pd.DataFrame:
    """Generate a synthetic cell outage and recovery event for dashboard evidence."""
    df = generate_synthetic_telemetry(cells=cells, periods=periods, freq=freq, seed=seed, start=start)
    if affected_cell not in set(df["cell_id"]):
        affected_cell = df["cell_id"].iloc[0]

    cell_df = df.loc[df["cell_id"] == affected_cell].copy()
    shock_idx = int(len(cell_df) * shock_start)
    shock_idx = max(0, min(shock_idx, max(len(cell_df) - 1, 0)))
    end_idx = min(len(cell_df), shock_idx + shock_duration)

    if end_idx > shock_idx:
        shock_slice = cell_df.index[shock_idx:end_idx]
        ramp = np.linspace(0.1, 1.0, len(shock_slice))
        recovery = np.linspace(1.0, 0.2, len(shock_slice))
        df.loc[shock_slice, "prb_dl_util"] = np.clip(df.loc[shock_slice, "prb_dl_util"] * (1 - 0.85 * ramp), 0, 99)
        df.loc[shock_slice, "prb_ul_util"] = np.clip(df.loc[shock_slice, "prb_ul_util"] * (1 - 0.75 * ramp), 0, 99)
        df.loc[shock_slice, "throughput_dl_mbps"] = np.clip(df.loc[shock_slice, "throughput_dl_mbps"] * (0.08 + 0.35 * recovery), 0, None)
        df.loc[shock_slice, "throughput_ul_mbps"] = np.clip(df.loc[shock_slice, "throughput_ul_mbps"] * (0.10 + 0.30 * recovery), 0, None)
        df.loc[shock_slice, "latency_ms"] = np.clip(df.loc[shock_slice, "latency_ms"] + 22 * ramp, 5, None)
        df.loc[shock_slice, "packet_loss_pct"] = np.clip(df.loc[shock_slice, "packet_loss_pct"] + 4.0 * ramp, 0, 25)
        df.loc[shock_slice, "sinr_db"] = np.clip(df.loc[shock_slice, "sinr_db"] - 8.0 * ramp, 0, 30)
        df.loc[shock_slice, "rrc_users"] = np.clip(df.loc[shock_slice, "rrc_users"] * (0.45 + 0.35 * recovery), 5, 500).astype(int)

    return df.reset_index(drop=True)
