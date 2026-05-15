from pathlib import Path

from ai_ran_kpi_forecasting.data import generate_synthetic_telemetry
from ai_ran_kpi_forecasting.forecasting import run_forecast_pipeline
from ai_ran_kpi_forecasting.reports import write_report_bundle


def test_synthetic_telemetry_schema():
    df = generate_synthetic_telemetry(cells=2, periods=24, seed=7)

    assert len(df) == 48
    assert {"timestamp", "cell_id", "prb_dl_util", "throughput_mbps", "rrc_users", "latency_ms"} <= set(df.columns)
    assert df["cell_id"].nunique() == 2


def test_forecast_pipeline_writes_report_bundle(tmp_path):
    result = run_forecast_pipeline(
        data="data/ran_kpi_sample.csv",
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=6,
        lags="1,2,3,6",
    )
    artifacts = write_report_bundle(result, tmp_path)

    assert result.target_col == "prb_dl_util"
    assert len(result.forecast) == 6
    assert result.metrics["rmse"] >= 0
    assert Path(artifacts["forecast_csv"]).exists()
    assert Path(artifacts["metrics_markdown"]).exists()
