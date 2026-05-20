"""Tests for the three-model registry and the comparison runner contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline
from ai_ran_kpi_forecasting.models import (
    DEFAULT_MODEL,
    MODEL_NAMES,
    build_regressor,
    feature_importance_frame,
)

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "ran_kpi_sample.csv"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_build_regressor_each_supported_name(model_name: str):
    model, name = build_regressor(model_name)
    assert name == model_name
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_regressor_default_is_ridge():
    model, name = build_regressor()
    assert name == DEFAULT_MODEL == "ridge_linear"
    assert hasattr(model, "coef_")  # ridge has .coef_


def test_build_regressor_unknown_name_raises():
    with pytest.raises(ValueError):
        build_regressor("not-a-real-model")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_run_forecast_pipeline_with_each_model(model_name: str):
    """Each model trains end-to-end on the sample CSV and produces finite metrics."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=6,  # small horizon to keep the test fast
        model_name=model_name,
    )
    assert result.model_name == model_name
    assert result.target_col == "prb_dl_util"
    assert result.cell_id == "CELL_001"
    for metric in ("rmse", "mae", "mape"):
        value = result.metrics[metric]
        assert value >= 0, f"{model_name}: {metric} should be non-negative, got {value}"
        assert value < 1e6, f"{model_name}: {metric} is unreasonably large"
    # Forecast frame matches horizon length and has the right schema.
    assert len(result.forecast) == 6
    assert {"timestamp", "forecast_step", "y_hat"} <= set(result.forecast.columns)


def test_feature_importance_for_ridge_has_coefficient_column():
    """Linear models surface the coefficient sign in addition to the magnitude."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=6,
        model_name="ridge_linear",
    )
    importance = feature_importance_frame(result.model, list(result.feature_sample.columns))
    assert not importance.empty
    assert "coefficient" in importance.columns
    assert "importance" in importance.columns


def test_feature_importance_for_gradient_boosting_uses_native_importances():
    """Tree ensembles surface .feature_importances_ rather than .coef_."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=6,
        model_name="gradient_boosting",
    )
    importance = feature_importance_frame(result.model, list(result.feature_sample.columns))
    assert not importance.empty
    assert "importance" in importance.columns


def test_feature_importance_for_mlp_returns_empty_frame():
    """MLP has no native attribution → empty frame (would need permutation importance)."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=6,
        model_name="mlp",
    )
    importance = feature_importance_frame(result.model, list(result.feature_sample.columns))
    assert importance.empty


def test_comparison_runner_writes_per_model_artifacts(tmp_path: Path):
    """Smoke test for the comparison script: each model writes its own dir."""
    from scripts.run_model_comparison import _run_one_model

    rows = []
    for model_name in MODEL_NAMES:
        row = _run_one_model(
            model_name=model_name,
            data=_SAMPLE_CSV,
            output_dir=tmp_path,
            cell_id="CELL_001",
            kpi_col="prb_dl_util",
            horizon=6,
        )
        rows.append(row)
        # Each model gets its own subdirectory with metrics + holdout + forecast.
        model_dir = tmp_path / model_name
        assert (model_dir / "metrics.json").exists()
        assert (model_dir / "holdout.csv").exists()
        assert (model_dir / "forecast.csv").exists()
        # The JSON parses cleanly.
        data = json.loads((model_dir / "metrics.json").read_text())
        assert data["model_name"] == model_name
        assert data["cell_id"] == "CELL_001"
        for k in ("rmse", "mae", "mape"):
            assert k in data and data[k] >= 0
    assert len(rows) == len(MODEL_NAMES)
