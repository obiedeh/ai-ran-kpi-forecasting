from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_ran_kpi_forecasting.data import filter_cell, load_ran_kpi_data, load_telecom_italia_mi
from ai_ran_kpi_forecasting.features import add_lag_features
from ai_ran_kpi_forecasting.forecast import parse_lags, run_forecast_pipeline, temporal_train_test_split
from ai_ran_kpi_forecasting.metrics import regression_metrics

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "ran_kpi_sample.csv"


def test_data_loading_orders_by_cell_and_timestamp(tmp_path):
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T02:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z"],
            "cell_id": ["CELL_002", "CELL_001", "CELL_001"],
            "prb_dl_util": [30.0, 20.0, 10.0],
        }
    ).to_csv(path, index=False)

    df = load_ran_kpi_data(path)

    assert list(df["cell_id"]) == ["CELL_001", "CELL_001", "CELL_002"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert list(df["prb_dl_util"]) == [10.0, 20.0, 30.0]


def test_lag_feature_generation_drops_unavailable_rows():
    df = pd.DataFrame({"prb_dl_util": [10.0, 20.0, 30.0, 40.0]})

    lagged = add_lag_features(df, "prb_dl_util", [1, 2])

    assert list(lagged.columns) == ["prb_dl_util", "prb_dl_util_lag_1", "prb_dl_util_lag_2"]
    assert len(lagged) == 2
    assert lagged.iloc[0].to_dict() == {
        "prb_dl_util": 30.0,
        "prb_dl_util_lag_1": 20.0,
        "prb_dl_util_lag_2": 10.0,
    }


def test_temporal_train_test_split_preserves_order():
    X = pd.DataFrame({"idx": np.arange(10)})
    y = pd.Series(np.arange(10))

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.3)

    assert list(X_train["idx"]) == list(range(7))
    assert list(X_test["idx"]) == [7, 8, 9]
    assert list(y_train) == list(range(7))
    assert list(y_test) == [7, 8, 9]


def test_forecast_output_shape():
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=8,
        lags="1,2,3",
    )

    assert list(result.forecast.columns) == ["timestamp", "forecast_step", "y_hat"]
    assert result.forecast.shape == (8, 3)
    assert list(result.forecast["forecast_step"]) == list(range(1, 9))


def test_metrics_calculation():
    metrics = regression_metrics(pd.Series([10.0, 20.0, 40.0]), np.array([12.0, 18.0, 44.0]))

    assert round(metrics["mae"], 4) == 2.6667
    assert round(metrics["rmse"], 4) == 2.8284
    assert round(metrics["mape"], 4) == 13.3333


# --- Error-path tests ---

def test_load_ran_kpi_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_ran_kpi_data("/nonexistent/path/telemetry.csv")


def test_load_ran_kpi_data_missing_timestamp_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"cell_id": ["CELL_001"], "prb_dl_util": [50.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Timestamp column"):
        load_ran_kpi_data(path, timestamp_col="timestamp")


def test_load_ran_kpi_data_missing_cell_id_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "prb_dl_util": [50.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Cell ID column"):
        load_ran_kpi_data(path, cell_id_col="cell_id")


def test_parse_lags_rejects_zero():
    with pytest.raises(ValueError, match="positive"):
        parse_lags("0,1,2")


def test_parse_lags_rejects_negative():
    with pytest.raises(ValueError, match="positive"):
        parse_lags([-1, 2])


def test_parse_lags_rejects_empty():
    with pytest.raises(ValueError, match="At least one lag"):
        parse_lags("")


def test_temporal_train_test_split_rejects_too_few_samples():
    X = pd.DataFrame({"x": range(5)})
    y = pd.Series(range(5))
    with pytest.raises(ValueError, match="Not enough samples"):
        temporal_train_test_split(X, y, test_size=0.2)


def test_temporal_train_test_split_rejects_invalid_test_size():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series(range(20))
    with pytest.raises(ValueError, match="test_size"):
        temporal_train_test_split(X, y, test_size=1.5)


def test_add_lag_features_rejects_zero_lag():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="positive"):
        add_lag_features(df, "v", [0])


# --- Previously missing coverage ---


def test_load_telecom_italia_mi_parses_ms_timestamps_and_renames_columns(tmp_path):
    """Telecom Italia timestamps are milliseconds since epoch; columns must be renamed."""
    ts_base = 1_388_534_400_000  # 2014-01-01 00:00:00 UTC in ms
    hour_ms = 3_600_000
    df_raw = pd.DataFrame({
        "Square id": [1, 1, 2],
        "time_interval": [ts_base, ts_base + hour_ms, ts_base],
        "Internet traffic activity": [100.0, 200.0, 50.0],
    })
    path = tmp_path / "mi.csv"
    df_raw.to_csv(path, index=False)

    result = load_telecom_italia_mi(path, aggregate="10min")

    assert "cell_id" in result.columns
    assert "timestamp" in result.columns
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result["cell_id"].nunique() == 2
    assert result["internet_traffic"].sum() == pytest.approx(350.0)


def test_filter_cell_returns_densest_cell_when_no_cell_id_given():
    """When cell_id is None, filter_cell must return the cell with the most rows."""
    df = pd.DataFrame({
        "cell_id": ["CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B"],
        "prb_dl_util": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    result = filter_cell(df, cell_id=None)

    assert list(result["cell_id"].unique()) == ["CELL_A"]
    assert len(result) == 3


def test_forecast_autoregressive_produces_finite_sequential_steps():
    """Each forecast step must be finite, sequential, and have strictly increasing timestamps."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=5,
        lags="1,2",
    )
    fc = result.forecast

    assert list(fc["forecast_step"]) == [1, 2, 3, 4, 5]
    assert np.all(np.isfinite(fc["y_hat"].to_numpy()))
    assert (fc["timestamp"].diff().dropna() > pd.Timedelta(0)).all()
