"""Model training utilities for KPI forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_ran_kpi_forecasting.metrics import regression_metrics


class RidgeForecastRegressor:
    """Small deterministic regressor with standardization and ridge regularization."""

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecastRegressor:
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


def build_regressor() -> tuple[RidgeForecastRegressor, str]:
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
    _ = random_state
    model, model_name = build_regressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    holdout = pd.DataFrame({"actual": y_test.to_numpy(), "prediction": y_pred}, index=y_test.index)
    return model, metrics, holdout, model_name


def feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract model-native feature importance when available."""
    values = getattr(model, "coef_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_columns, "importance": np.abs(values), "coefficient": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
