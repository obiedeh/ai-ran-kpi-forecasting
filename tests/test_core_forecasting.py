import numpy as np
import pandas as pd

from ai_ran_kpi_forecasting.data import load_ran_kpi_data
from ai_ran_kpi_forecasting.features import add_lag_features
from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline, temporal_train_test_split
from ai_ran_kpi_forecasting.metrics import regression_metrics


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
        data="data/ran_kpi_sample.csv",
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
