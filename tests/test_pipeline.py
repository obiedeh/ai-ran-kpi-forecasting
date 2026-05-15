from pathlib import Path

from ai_ran_kpi_forecasting.data import (
    generate_backhaul_telemetry,
    generate_congestion_telemetry,
    generate_outage_telemetry,
    generate_synthetic_telemetry,
)
from ai_ran_kpi_forecasting.forecasting import run_forecast_pipeline
from ai_ran_kpi_forecasting.reports import (
    write_portal_page,
    write_publish_page,
    write_report_bundle,
    write_scenario_dashboard,
)


def test_synthetic_telemetry_schema():
    df = generate_synthetic_telemetry(cells=2, periods=24, seed=7)

    assert len(df) == 48
    assert {
        "timestamp",
        "cell_id",
        "site_id",
        "sector_id",
        "technology",
        "prb_dl_util",
        "prb_ul_util",
        "throughput_dl_mbps",
        "throughput_ul_mbps",
        "rrc_users",
        "latency_ms",
        "packet_loss_pct",
        "sinr_db",
    } <= set(df.columns)
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


def test_congestion_scenario_and_dashboard(tmp_path):
    baseline = generate_synthetic_telemetry(cells=1, periods=48, seed=3)
    congestion = generate_congestion_telemetry(cells=1, periods=48, seed=3, congested_cell="CELL_001")

    assert congestion["prb_dl_util"].max() >= baseline["prb_dl_util"].max()
    assert congestion["throughput_dl_mbps"].min() <= baseline["throughput_dl_mbps"].min()

    baseline_path = tmp_path / "baseline.csv"
    congestion_path = tmp_path / "congestion.csv"
    baseline.to_csv(baseline_path, index=False)
    congestion.to_csv(congestion_path, index=False)

    baseline_result = run_forecast_pipeline(data=str(baseline_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)
    congestion_result = run_forecast_pipeline(data=str(congestion_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)

    baseline_dir = tmp_path / "baseline_report"
    congestion_dir = tmp_path / "congestion_report"
    write_report_bundle(baseline_result, baseline_dir)
    write_report_bundle(congestion_result, congestion_dir)
    artifacts = write_scenario_dashboard(
        baseline_result=baseline_result,
        congestion_result=congestion_result,
        baseline_dir=baseline_dir,
        congestion_dir=congestion_dir,
        baseline_telemetry_path=baseline_path,
        congestion_telemetry_path=congestion_path,
        output_dir=tmp_path / "dashboard",
    )

    assert Path(artifacts["dashboard_html"]).exists()
    assert Path(artifacts["dashboard_summary"]).exists()


def test_backhaul_scenario_and_portal(tmp_path):
    baseline = generate_synthetic_telemetry(cells=1, periods=48, seed=11)
    backhaul = generate_backhaul_telemetry(cells=1, periods=48, seed=11, affected_cell="CELL_001")

    assert backhaul["throughput_dl_mbps"].min() <= baseline["throughput_dl_mbps"].min()
    assert backhaul["latency_ms"].max() >= baseline["latency_ms"].max()

    baseline_path = tmp_path / "baseline.csv"
    backhaul_path = tmp_path / "backhaul.csv"
    baseline.to_csv(baseline_path, index=False)
    backhaul.to_csv(backhaul_path, index=False)

    baseline_result = run_forecast_pipeline(data=str(baseline_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)
    backhaul_result = run_forecast_pipeline(data=str(backhaul_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)

    baseline_dir = tmp_path / "baseline_report"
    backhaul_dir = tmp_path / "backhaul_report"
    write_report_bundle(baseline_result, baseline_dir)
    write_report_bundle(backhaul_result, backhaul_dir)
    artifacts = write_scenario_dashboard(
        baseline_result=baseline_result,
        congestion_result=backhaul_result,
        baseline_dir=baseline_dir,
        congestion_dir=backhaul_dir,
        baseline_telemetry_path=baseline_path,
        congestion_telemetry_path=backhaul_path,
        output_dir=tmp_path / "dashboard",
        scenario_name="AI-RAN backhaul saturation scenario",
    )

    portal_path = write_portal_page(tmp_path / "reports" / "index.html")

    assert Path(artifacts["dashboard_html"]).exists()
    assert Path(portal_path).exists()
    assert "Backhaul Scenario" in portal_path.read_text(encoding="utf-8")


def test_outage_scenario_and_publish_page(tmp_path):
    baseline = generate_synthetic_telemetry(cells=1, periods=48, seed=17)
    outage = generate_outage_telemetry(cells=1, periods=48, seed=17, affected_cell="CELL_001")

    assert outage["throughput_dl_mbps"].min() <= baseline["throughput_dl_mbps"].min()
    assert outage["prb_dl_util"].min() <= baseline["prb_dl_util"].min()

    baseline_path = tmp_path / "baseline.csv"
    outage_path = tmp_path / "outage.csv"
    baseline.to_csv(baseline_path, index=False)
    outage.to_csv(outage_path, index=False)

    baseline_result = run_forecast_pipeline(data=str(baseline_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)
    outage_result = run_forecast_pipeline(data=str(outage_path), cell_id="CELL_001", kpi_col="prb_dl_util", horizon=6)

    baseline_dir = tmp_path / "baseline_report"
    outage_dir = tmp_path / "outage_report"
    write_report_bundle(baseline_result, baseline_dir)
    write_report_bundle(outage_result, outage_dir)
    artifacts = write_scenario_dashboard(
        baseline_result=baseline_result,
        congestion_result=outage_result,
        baseline_dir=baseline_dir,
        congestion_dir=outage_dir,
        baseline_telemetry_path=baseline_path,
        congestion_telemetry_path=outage_path,
        output_dir=tmp_path / "dashboard",
        scenario_name="AI-RAN cell outage recovery scenario",
    )

    publish_path = write_publish_page(tmp_path / "publish" / "latest")

    assert Path(artifacts["dashboard_summary"]).exists()
    assert Path(publish_path).exists()
    assert Path(publish_path).with_name("manifest.json").exists()
