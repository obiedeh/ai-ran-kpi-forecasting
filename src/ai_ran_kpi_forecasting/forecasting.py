"""Backward-compatible imports for the renamed forecast module."""

from ai_ran_kpi_forecasting.forecast import (
    ForecastRunResult,
    forecast_autoregressive,
    parse_lags,
    run_forecast_pipeline,
    temporal_train_test_split,
)
from ai_ran_kpi_forecasting.models import (
    RidgeForecastRegressor,
    build_regressor,
    feature_importance_frame,
    train_and_evaluate,
)

__all__ = [
    "ForecastRunResult",
    "RidgeForecastRegressor",
    "build_regressor",
    "feature_importance_frame",
    "forecast_autoregressive",
    "parse_lags",
    "run_forecast_pipeline",
    "temporal_train_test_split",
    "train_and_evaluate",
]
