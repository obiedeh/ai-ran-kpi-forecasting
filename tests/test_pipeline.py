import json
from pathlib import Path

import pandas as pd

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "ran_kpi_sample.csv"
_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

from ai_ran_kpi_forecasting.data import (
    generate_backhaul_telemetry,
    generate_congestion_telemetry,
    generate_outage_telemetry,
    generate_synthetic_telemetry,
)
from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline
from ai_ran_kpi_forecasting.reports import (
    write_portal_page,
    write_publish_page,
    write_report_bundle,
    write_scenario_dashboard,
)
from ai_ran_kpi_forecasting.visualization import plot_feature_importance, plot_forecast


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
        "edge_gpu_util_pct",
        "edge_memory_util_pct",
    } <= set(df.columns)
    assert df["cell_id"].nunique() == 2
    assert df["edge_gpu_util_pct"].between(5, 98).all()
    assert df["edge_memory_util_pct"].between(12, 95).all()


def test_forecast_pipeline_writes_report_bundle(tmp_path):
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
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
    assert Path(portal_path).with_name("dashboard.html").exists()
    portal_html = portal_path.read_text(encoding="utf-8")
    dashboard_html = Path(portal_path).with_name("dashboard.html").read_text(encoding="utf-8")
    assert "Backhaul Scenario" in portal_html
    assert "Executive KPI cards" in portal_html
    assert "Forecast horizon" in portal_html
    assert "Target KPI" in portal_html
    assert "Best measured model" in portal_html
    assert "Peak forecast" in portal_html
    assert "Benchmark status" in portal_html
    assert "Pending local dataset" in portal_html
    assert "Operational interpretation" in portal_html
    assert "Congestion risk tiers" in portal_html
    assert "Model reliability" in portal_html
    assert "Benchmark readiness: Telecom Italia MI" in portal_html
    assert "Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet." in portal_html
    assert "reports/forecast_examples/telecom_italia_mi/" in portal_html
    assert "Engineering boundaries" in portal_html
    assert "AI-RAN KPI Forecasting Dashboard" in dashboard_html
    assert "does not connect to a live RAN" in dashboard_html
    assert "does not perform autonomous control" in dashboard_html
    assert "scenarios/latest/congestion/dashboard/dashboard.html" in portal_html
    assert "schemas/kpm_input_v1.json" in portal_html
    assert "\\" not in portal_html


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


# --- Previously missing coverage ---


def test_plot_forecast_writes_valid_svg(tmp_path):
    """plot_forecast must produce a file whose content is a valid SVG root element."""
    holdout = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
        "actual": [40.0, 42.0, 41.0, 43.0],
        "prediction": [39.5, 42.5, 41.5, 43.5],
    })
    forecast_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 04:00", periods=2, freq="1h", tz="UTC"),
        "forecast_step": [1, 2],
        "y_hat": [44.0, 45.0],
    })
    out = tmp_path / "fc.svg"
    plot_forecast(holdout, forecast_df, out, "prb_dl_util")

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<svg")
    assert "prb_dl_util" in content


def test_plot_feature_importance_writes_valid_svg(tmp_path):
    """plot_feature_importance must write an SVG and not contain the old 'linear baseline' subtitle."""
    fi = pd.DataFrame({"feature": ["lag_1", "hour_sin", "dow_cos"], "importance": [0.5, 0.3, 0.1]})
    out = tmp_path / "fi.svg"
    plot_feature_importance(fi, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<svg")
    assert "linear baseline" not in content


def test_synthetic_telemetry_technology_matches_kpm_schema():
    """generate_synthetic_telemetry must emit 'technology' values in the KPM schema enum."""
    schema = json.loads((_SCHEMA_DIR / "kpm_input_v1.json").read_text(encoding="utf-8"))
    allowed = set(schema["definitions"]["KpmRecord"]["properties"]["technology"]["enum"])

    df = generate_synthetic_telemetry(cells=2, periods=5, seed=0)

    for tech in df["technology"].unique():
        assert tech in allowed, f"technology {tech!r} not in schema enum {allowed}"


def test_write_report_bundle_returns_feature_importance_svg_path(tmp_path):
    """write_report_bundle must include 'feature_importance_svg' in the returned dict."""
    result = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=4,
    )
    artifacts = write_report_bundle(result, tmp_path)

    assert "feature_importance_svg" in artifacts
    assert Path(artifacts["feature_importance_svg"]).exists()
