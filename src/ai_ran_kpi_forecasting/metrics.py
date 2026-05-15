"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute compact hold-out metrics for KPI forecasts."""
    y_true_arr = y_true.to_numpy(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / np.clip(np.abs(y_true_arr), 1e-6, None))) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}
