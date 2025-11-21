#!/usr/bin/env python
"""
AI-RAN KPI Forecasting Script

Supports:
1) Generic RAN KPI CSV (timestamp + cell_id + KPI columns)
2) Telecom Italia "Telecommunications - SMS, Call, Internet - MI" dataset

Usage examples:

Generic CSV:
    python ai_ran_kpi_forecasting.py \
        --dataset-type generic \
        --data ./data/ran_kpi_sample.csv \
        --timestamp-col timestamp \
        --cell-id-col cell_id \
        --kpi-col prb_dl_util \
        --cell-id CELL_001 \
        --horizon 24

Telecom Italia MI (directory with CSVs):
    python ai_ran_kpi_forecasting.py \
        --dataset-type telecom-italia-mi \
        --data ./data/telecom_italia_mi \
        --aggregate hourly \
        --kpi-col internet_traffic \
        --horizon 24

This script:
- Loads RAN KPI time-series data for one "cell".
- Engineers lag + calendar features.
- Trains an XGBoost (or RandomForest) regressor to forecast a target KPI.
- Evaluates on a time-ordered hold-out set.
- Produces a short-horizon autoregressive forecast for the selected cell.

You can extend this into an MLOps / MLflow / RAPIDS pipeline.
"""

import argparse
import glob
import os
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    _HAS_XGB = False


# -------------------------------------------------------------------
# Data Loading: Generic RAN CSV
# -------------------------------------------------------------------

def load_ran_kpi_data(
    path: str,
    timestamp_col: str = "timestamp",
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Load generic RAN KPI data from CSV and perform basic cleaning.

    Expected columns:
    - timestamp_col: datetime string
    - cell_id_col:   e.g. 'CELL_001'
    - KPI columns:   numeric
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in data.")

    if cell_id_col not in df.columns:
        raise ValueError(f"Cell ID column '{cell_id_col}' not found in data.")

    # Parse timestamp and sort
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(by=[cell_id_col, timestamp_col])

    return df


def filter_cell(
    df: pd.DataFrame,
    cell_id: Optional[str],
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Filter data to one cell. If cell_id is None, chooses the
    cell with the most samples.
    """
    if cell_id_col not in df.columns:
        raise ValueError(f"cell_id_col '{cell_id_col}' not found in data.")

    if cell_id is not None:
        df_cell = df[df[cell_id_col] == cell_id].copy()
        if df_cell.empty:
            raise ValueError(f"No rows found for cell_id='{cell_id}'.")
        return df_cell

    # Choose the densest cell by count
    counts = df[cell_id_col].value_counts()
    best_cell_id = counts.idxmax()
    print(f"[INFO] No cell-id specified. Using densest cell: {best_cell_id} (n={counts.max()})")
    return df[df[cell_id_col] == best_cell_id].copy()


# -------------------------------------------------------------------
# Data Loading: Telecom Italia MI (Option 2)
# -------------------------------------------------------------------

def _normalize_telecom_italia_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Telecom Italia 'Telecommunications - SMS, Call, Internet - MI' columns
    into a consistent schema.

    Typical original columns:
    - 'Square id', 'Time interval', 'Country code'
    - 'SMS-in activity', 'SMS-out activity'
    - 'Call-in activity', 'Call-out activity'
    - 'Internet traffic activity'
    """
    col_map = {
        # IDs / time
        "Square id": "cell_id",
        "square_id": "cell_id",
        "Square_id": "cell_id",

        "Time interval": "time_interval",
        "time_interval": "time_interval",
        "Time_interval": "time_interval",

        "Country code": "country_code",
        "country_code": "country_code",

        # Activities (handle canonical and lower variants)
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

    required_cols = ["cell_id", "time_interval"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Telecom Italia columns: {missing}")

    return df


def load_telecom_italia_mi(
    data_path: str,
    aggregate: str = "hourly",
) -> pd.DataFrame:
    """
    Load Telecom Italia 'Telecommunications - SMS, Call, Internet - MI' data and
    transform it to a generic RAN KPI schema:

        timestamp (UTC), cell_id, sms_in, sms_out, call_in, call_out, internet_traffic

    Parameters
    ----------
    data_path : str
        Either a directory containing multiple daily CSVs, or a single CSV file.
    aggregate : {'10min', 'hourly'}
        '10min' : keep original 10-min aggregation (after summing over country_code).
        'hourly': resample per cell to 1H buckets.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with at least ['timestamp', 'cell_id', 'internet_traffic'] plus
        other activity columns if present.
    """
    # 1) Discover files
    if os.path.isdir(data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {data_path}")
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        files = [data_path]

    dfs = []
    for f in files:
        tmp = pd.read_csv(f)
        tmp = _normalize_telecom_italia_columns(tmp)
        dfs.append(tmp)

    if not dfs:
        raise ValueError("No data loaded from Telecom Italia MI files.")

    df = pd.concat(dfs, ignore_index=True)

    # 2) Convert time_interval (UNIX ms) -> timestamp (UTC)
    df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "cell_id"])

    # 3) Sum over country_code for each (cell_id, timestamp)
    agg_cols = [c for c in df.columns if c in [
        "sms_in", "sms_out", "call_in", "call_out", "internet_traffic"
    ]]
    if not agg_cols:
        raise ValueError("No activity columns (sms/call/internet) found in Telecom Italia data.")

    df = (
        df.groupby(["cell_id", "timestamp"], as_index=False)[agg_cols]
          .sum()
    )

    if aggregate == "10min":
        # Already in ~10-min slots after groupby
        return df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    # 4) Optional resample to hourly per cell_id
    df = df.set_index("timestamp")

    resampled = (
        df.groupby("cell_id")
          .resample("1H")
          .sum()
          .reset_index()
          .sort_values(["cell_id", "timestamp"])
          .reset_index(drop=True)
    )

    return resampled


# -------------------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------------------

def choose_kpi_column(df: pd.DataFrame, kpi_col: Optional[str]) -> str:
    """
    If KPI column is given, validate it. Otherwise select
    the first numeric column (excluding obvious non-KPI cols).
    """
    if kpi_col is not None:
        if kpi_col not in df.columns:
            raise ValueError(f"kpi_col '{kpi_col}' not in DataFrame.")
        if not np.issubdtype(df[kpi_col].dtype, np.number):
            raise ValueError(f"kpi_col '{kpi_col}' is not numeric.")
        return kpi_col

    # Heuristic: numeric columns except obvious IDs
    exclude = {"cell_id", "Square id", "square_id"}
    numeric_cols = [
        c for c in df.columns
        if np.issubdtype(df[c].dtype, np.number) and c not in exclude
    ]
    if not numeric_cols:
        raise ValueError("No numeric KPI column detected. Please specify --kpi-col.")

    chosen = numeric_cols[0]
    print(f"[INFO] No KPI column specified. Using numeric column: '{chosen}'")
    return chosen


def engineer_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add calendar/cycle features from timestamp.
    """
    df = df.copy()
    if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

    df["hour"] = df[timestamp_col].dt.hour
    df["dayofweek"] = df[timestamp_col].dt.dayofweek
    df["dayofmonth"] = df[timestamp_col].dt.day
    df["weekofyear"] = df[timestamp_col].dt.isocalendar().week.astype(int)

    # Simple cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
) -> pd.DataFrame:
    """
    Create lag features for the target KPI.

    NOTE: For a real RAN system you'd likely create:
    - Multiple KPIs as features (e.g. PRB, throughput, RRC, latency)
    - Rolling windows, min/max, std, etc.
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into X (features) and y (target).
    """
    if exclude_cols is None:
        exclude_cols = []

    exclude_cols = set(exclude_cols + [target_col])

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float).copy()
    return X, y


# -------------------------------------------------------------------
# Model Training & Evaluation
# -------------------------------------------------------------------

def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-ordered train/test split.
    """
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough samples ({n}) for train/test split.")

    split_idx = int(n * (1.0 - test_size))
    split_idx = max(1, min(split_idx, n - 1))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def build_regressor(random_state: int = 42):
    """
    Build regressor model:
    - Prefer XGBRegressor if installed.
    - Otherwise fall back to RandomForestRegressor.
    """
    if _HAS_XGB:
        print("[INFO] Using XGBRegressor.")
        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",  # can switch to 'gpu_hist' for GPU
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        print("[INFO] xgboost not installed; using RandomForestRegressor.")
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", model),
        ]
    )
    return pipeline


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Train regressor and compute metrics.
    """
    model = build_regressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(
        np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-6, None))
    ) * 100.0

    print("\n================= Evaluation (Hold-out) =================")
    print(f"Samples (train): {len(X_train):,}")
    print(f"Samples (test):  {len(X_test):,}")
    print(f"RMSE:            {rmse:0.4f}")
    print(f"MAE:             {mae:0.4f}")
    print(f"MAPE:            {mape:0.2f}%")
    print("=======================================================\n")

    return model, {"rmse": rmse, "mae": mae, "mape": mape}


# -------------------------------------------------------------------
# Short-Horizon Forecasting (Autoregressive)
# -------------------------------------------------------------------

def forecast_autoregressive(
    model,
    df_full: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    horizon: int,
    lags: List[int],
) -> pd.DataFrame:
    """
    Simple autoregressive multi-step forecast using lagged KPI values.

    - df_full must already contain lag features.
    - We will roll forward horizon steps, feeding predictions back
      into the lag columns.
    """
    df_full = df_full.copy()

    # Start from the last row
    last_row = df_full.iloc[-1:].copy()

    # Determine feature columns used during training (exclude target)
    feature_cols = [c for c in last_row.columns if c not in [target_col]]

    forecasts = []

    last_timestamp = last_row[timestamp_col].iloc[0]
    inferred_freq = pd.infer_freq(df_full[timestamp_col])
    if inferred_freq is None:
        # fallback: use median difference
        diffs = df_full[timestamp_col].diff().dropna()
        step = diffs.median()
    else:
        step = pd.tseries.frequencies.to_offset(inferred_freq)

    current_state = last_row.copy()

    for step_idx in range(1, horizon + 1):
        # Predict for current state
        X_curr = current_state[feature_cols]
        y_hat = model.predict(X_curr)[0]

        # Advance time
        last_timestamp = last_timestamp + step
        # Create next row
        next_row = current_state.copy()
        next_row[timestamp_col] = last_timestamp
        next_row[target_col] = y_hat

        # Update lag features
        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            if lag == 1:
                next_row[col_name] = current_state[target_col].iloc[0]
            else:
                prev_col = f"{target_col}_lag_{lag - 1}"
                if prev_col in current_state.columns:
                    next_row[col_name] = current_state[prev_col].iloc[0]

        forecasts.append(
            {
                "timestamp": last_timestamp,
                "forecast_step": step_idx,
                "y_hat": float(y_hat),
            }
        )

        current_state = next_row

    return pd.DataFrame(forecasts)


# -------------------------------------------------------------------
# CLI Interface
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-RAN KPI Forecasting (per-cell, univariate KPI)."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["generic", "telecom-italia-mi"],
        default="generic",
        help=(
            "Dataset format: "
            "'generic' = your own CSV with timestamp/cell_id; "
            "'telecom-italia-mi' = Telecom Italia Big Data Challenge MI files."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "Path to RAN KPI CSV (generic) OR directory/single CSV of "
            "Telecom Italia MI data (depending on --dataset-type)."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Timestamp column name (used when --dataset-type=generic).",
    )
    parser.add_argument(
        "--cell-id-col",
        type=str,
        default="cell_id",
        help="Cell ID column name (used when --dataset-type=generic).",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["10min", "hourly"],
        default="hourly",
        help=(
            "Aggregation level for Telecom Italia MI data: "
            "'10min' = keep original; 'hourly' = resample per cell to 1H."
        ),
    )
    parser.add_argument(
        "--cell-id",
        type=str,
        default=None,
        help="Specific cell_id to model. If omitted, chooses the densest cell.",
    )
    parser.add_argument(
        "--kpi-col",
        type=str,
        default=None,
        help="KPI column to forecast. If omitted, first numeric column is chosen.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples for time-ordered test split.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon (# steps) for autoregressive forecast.",
    )
    parser.add_argument(
        "--lags",
        type=str,
        default="1,2,3,6,12",
        help="Comma-separated lags to use for autoregressive features.",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = parse_args()
    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]

    print("=======================================================")
    print("         AI-RAN KPI Forecasting (Per-Cell KPI)         ")
    print("=======================================================")
    print(f"Dataset type   : {args.dataset_type}")
    print(f"Data path      : {args.data}")
    print(f"Timestamp col  : {args.timestamp_col}")
    print(f"Cell ID col    : {args.cell_id_col}")
    print(f"Cell ID        : {args.cell_id}")
    print(f"Test size      : {args.test_size}")
    print(f"Lags           : {lags}")
    print(f"Horizon (steps): {args.horizon}")
    if args.dataset_type == "telecom-italia-mi":
        print(f"TI aggregation : {args.aggregate}")
    print("=======================================================\n")

    # 1) Load data
    if args.dataset_type == "telecom-italia-mi":
        df = load_telecom_italia_mi(
            data_path=args.data,
            aggregate=args.aggregate,
        )
        # Override to aligned names used downstream
        args.timestamp_col = "timestamp"
        args.cell_id_col = "cell_id"
    else:
        df = load_ran_kpi_data(
            path=args.data,
            timestamp_col=args.timestamp_col,
            cell_id_col=args.cell_id_col,
        )

    # 2) Filter to one cell
    df_cell = filter_cell(
        df,
        cell_id=args.cell_id,
        cell_id_col=args.cell_id_col,
    )

    # 3) Choose KPI column
    kpi_col = choose_kpi_column(df_cell, args.kpi_col)
    print(f"[INFO] Target KPI column: '{kpi_col}'")

    # 4) Feature engineering
    df_fe = engineer_time_features(df_cell, timestamp_col=args.timestamp_col)
    df_fe = add_lag_features(df_fe, target_col=kpi_col, lags=lags)

    exclude_cols = [args.timestamp_col, args.cell_id_col]
    X, y = build_feature_matrix(
        df_fe,
        target_col=kpi_col,
        timestamp_col=args.timestamp_col,
        exclude_cols=exclude_cols,
    )

    # 5) Train/test split (temporal)
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X,
        y,
        test_size=args.test_size,
    )

    # 6) Train & evaluate
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 7) Short-horizon forecast
    print("[INFO] Generating autoregressive forecast...")
    df_forecast = forecast_autoregressive(
        model=model,
        df_full=df_fe,
        target_col=kpi_col,
        timestamp_col=args.timestamp_col,
        horizon=args.horizon,
        lags=lags,
    )

    print("=============== Short-Horizon Forecast ===============")
    print(df_forecast.head(20).to_string(index=False))
    print("======================================================")
    print("\n[INFO] Done. Consider adding MLflow logging, GPU acceleration, or "
          "multi-KPI models for production AI-RAN forecasting.")


if __name__ == "__main__":
    main()
