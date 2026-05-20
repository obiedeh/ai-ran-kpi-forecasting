"""Forecast pipeline orchestration and autoregressive prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ai_ran_kpi_forecasting.data import filter_cell, load_ran_kpi_data, load_telecom_italia_mi
from ai_ran_kpi_forecasting.features import (
    add_lag_features,
    build_feature_matrix,
    choose_kpi_column,
    engineer_time_features,
)
from ai_ran_kpi_forecasting.models import (
    DEFAULT_MODEL,
    feature_importance_frame,
    train_and_evaluate,
)


@dataclass
class ForecastRunResult:
    model: Any  # one of: RidgeForecastRegressor, _GradientBoostingWrapper, _MLPWrapper
    metrics: dict[str, float]
    forecast: pd.DataFrame
    holdout: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_sample: pd.DataFrame
    target_col: str
    cell_id: str
    model_name: str


def parse_lags(lags: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse comma-separated or sequence lag values."""
    if isinstance(lags, str):
        parsed = [int(item.strip()) for item in lags.split(",") if item.strip()]
    else:
        parsed = [int(item) for item in lags]
    if not parsed:
        raise ValueError("At least one lag is required.")
    if any(lag <= 0 for lag in parsed):
        raise ValueError("lags must be positive integers.")
    return parsed


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-ordered train/test split."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough samples ({n}) for train/test split.")
    split_idx = max(1, min(int(n * (1.0 - test_size)), n - 1))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def forecast_autoregressive(
    model: Any,
    df_full: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    horizon: int,
    lags: list[int],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate a short-horizon autoregressive forecast."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    last_row = df_full.iloc[-1:].copy()
    last_timestamp = last_row[timestamp_col].iloc[0]
    inferred_freq = pd.infer_freq(df_full[timestamp_col])
    if inferred_freq is None:
        step: pd.Timedelta = df_full[timestamp_col].diff().dropna().median()
    else:
        step = pd.tseries.frequencies.to_offset(inferred_freq).nanos * pd.Timedelta(1, "ns")

    current_state = last_row.copy()
    forecasts: list[dict[str, Any]] = []
    for step_idx in range(1, horizon + 1):
        y_hat = float(model.predict(current_state[feature_columns])[0])
        last_timestamp = last_timestamp + step
        next_row = current_state.copy()
        next_row[timestamp_col] = last_timestamp
        next_row[target_col] = y_hat

        for lag in lags:
            col_name = f"{target_col}_lag_{lag}"
            if lag == 1:
                next_row[col_name] = current_state[target_col].iloc[0]
            else:
                prev_col = f"{target_col}_lag_{lag - 1}"
                if prev_col in current_state.columns:
                    next_row[col_name] = current_state[prev_col].iloc[0]

        forecasts.append({"timestamp": last_timestamp, "forecast_step": step_idx, "y_hat": y_hat})
        current_state = next_row
    return pd.DataFrame(forecasts)


def run_forecast_pipeline(
    data: str,
    dataset_type: str = "generic",
    timestamp_col: str = "timestamp",
    cell_id_col: str = "cell_id",
    cell_id: str | None = None,
    kpi_col: str | None = None,
    aggregate: str = "hourly",
    test_size: float = 0.2,
    horizon: int = 24,
    lags: str | list[int] = "1,2,3,6,12",
    random_state: int = 42,
    model_name: str = DEFAULT_MODEL,
) -> ForecastRunResult:
    """Load telemetry, train a forecaster, and return all report-ready outputs."""
    lag_values = parse_lags(lags)
    if dataset_type == "telecom-italia-mi":
        df = load_telecom_italia_mi(data, aggregate=aggregate)
        timestamp_col = "timestamp"
        cell_id_col = "cell_id"
    elif dataset_type == "generic":
        df = load_ran_kpi_data(data, timestamp_col=timestamp_col, cell_id_col=cell_id_col)
    else:
        raise ValueError("dataset_type must be 'generic' or 'telecom-italia-mi'.")

    df_cell = filter_cell(df, cell_id=cell_id, cell_id_col=cell_id_col)
    selected_cell = str(df_cell[cell_id_col].iloc[0])
    target_col = choose_kpi_column(df_cell, kpi_col)

    df_fe = engineer_time_features(df_cell, timestamp_col=timestamp_col)
    df_fe = add_lag_features(df_fe, target_col=target_col, lags=lag_values)
    X, y = build_feature_matrix(df_fe, target_col=target_col, exclude_cols=[timestamp_col, cell_id_col])
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=test_size)
    model, metrics, holdout, model_name = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=random_state,
        model_name=model_name,
    )
    holdout.insert(0, "timestamp", df_fe.loc[X_test.index, timestamp_col].to_numpy())
    forecast = forecast_autoregressive(
        model=model,
        df_full=df_fe,
        target_col=target_col,
        timestamp_col=timestamp_col,
        horizon=horizon,
        lags=lag_values,
        feature_columns=list(X.columns),
    )
    importance = feature_importance_frame(model, list(X.columns))
    sample = X_test.head(min(50, len(X_test))).copy()
    return ForecastRunResult(
        model=model,
        metrics=metrics,
        forecast=forecast,
        holdout=holdout,
        feature_importance=importance,
        feature_sample=sample,
        target_col=target_col,
        cell_id=selected_cell,
        model_name=model_name,
    )
