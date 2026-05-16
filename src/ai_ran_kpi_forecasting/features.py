"""Feature engineering for KPI forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


_DEFAULT_ID_COLS: frozenset[str] = frozenset({"cell_id", "site_id", "Square id", "square_id", "Square_id"})


def choose_kpi_column(df: pd.DataFrame, kpi_col: str | None, exclude_cols: Sequence[str] | None = None) -> str:
    """Validate or infer the target KPI column."""
    if kpi_col is not None:
        if kpi_col not in df.columns:
            raise ValueError(f"kpi_col '{kpi_col}' not in DataFrame.")
        if not np.issubdtype(df[kpi_col].dtype, np.number):
            raise ValueError(f"kpi_col '{kpi_col}' is not numeric.")
        return kpi_col

    exclude = _DEFAULT_ID_COLS | set(exclude_cols or [])
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in exclude]
    if not numeric_cols:
        raise ValueError("No numeric KPI column detected. Please specify --kpi-col.")
    return numeric_cols[0]


def engineer_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar and cyclical time features."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

    df["hour"] = df[timestamp_col].dt.hour
    df["dayofweek"] = df[timestamp_col].dt.dayofweek
    df["dayofmonth"] = df[timestamp_col].dt.day
    df["weekofyear"] = df[timestamp_col].dt.isocalendar().week.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: Sequence[int]) -> pd.DataFrame:
    """Create lag features for the target KPI."""
    df = df.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError("lags must be positive integers.")
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df.dropna().reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature-engineered frame into numeric X and y."""
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(float).copy()
    if X.empty:
        raise ValueError("No numeric feature columns available after exclusions.")
    return X, y
