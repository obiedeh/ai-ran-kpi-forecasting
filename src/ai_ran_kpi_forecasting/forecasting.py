"""Forecast model training and autoregressive prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai_ran_kpi_forecasting.data import filter_cell, load_ran_kpi_data, load_telecom_italia_mi
from ai_ran_kpi_forecasting.features import (
    add_lag_features,
    build_feature_matrix,
    choose_kpi_column,
    engineer_time_features,
)
from ai_ran_kpi_forecasting.metrics import regression_metrics

@dataclass
class ForecastRunResult:
    model: Any
    metrics: dict[str, float]
    forecast: pd.DataFrame
    holdout: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_sample: pd.DataFrame
    target_col: str
    cell_id: str
    model_name: str


class RidgeForecastRegressor:
    """Small deterministic regressor with standardization and ridge regularization."""

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecastRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.mean_ = X_arr.mean(axis=0)
        self.scale_ = X_arr.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        X_scaled = (X_arr - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        X_scaled = (X_arr - self.mean_) / self.scale_
        return self.intercept_ + X_scaled @ self.coef_


def parse_lags(lags: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse comma-separated or sequence lag values."""
    if isinstance(lags, str):
        parsed = [int(item.strip()) for item in lags.split(",") if item.strip()]
    else:
        parsed = [int(item) for item in lags]
    if not parsed:
        raise ValueError("At least one lag is required.")
    return parsed


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-ordered train/test split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough samples ({n}) for train/test split.")
    split_idx = max(1, min(int(n * (1.0 - test_size)), n - 1))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def build_regressor(random_state: int = 42) -> tuple[RidgeForecastRegressor, str]:
    """Build a deterministic baseline regressor."""
    return RidgeForecastRegressor(alpha=1e-4), "ridge_linear"


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[RidgeForecastRegressor, dict[str, float], pd.DataFrame, str]:
    """Train a model and return metrics plus hold-out predictions."""
    model, model_name = build_regressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    holdout = pd.DataFrame({"actual": y_test.to_numpy(), "prediction": y_pred}, index=y_test.index)
    return model, metrics, holdout, model_name


def forecast_autoregressive(
    model: RidgeForecastRegressor,
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
        step = df_full[timestamp_col].diff().dropna().median()
    else:
        step = pd.tseries.frequencies.to_offset(inferred_freq)

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


def feature_importance_frame(model: RidgeForecastRegressor, feature_columns: list[str]) -> pd.DataFrame:
    """Extract model-native feature importance when available."""
    values = getattr(model, "coef_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_columns, "importance": np.abs(values), "coefficient": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


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
