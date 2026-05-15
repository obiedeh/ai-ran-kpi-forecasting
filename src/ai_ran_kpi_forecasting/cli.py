"""Command line interface for AI-RAN KPI forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_ran_kpi_forecasting.data import (
    generate_backhaul_telemetry,
    generate_congestion_telemetry,
    generate_synthetic_telemetry,
)
from ai_ran_kpi_forecasting.forecasting import run_forecast_pipeline
from ai_ran_kpi_forecasting.reports import write_portal_page, write_report_bundle, write_scenario_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-RAN KPI forecasting and report generation.")
    subparsers = parser.add_subparsers(dest="command")

    forecast = subparsers.add_parser("forecast", help="Train a KPI forecaster and write report artifacts.")
    forecast.add_argument("--dataset-type", choices=["generic", "telecom-italia-mi"], default="generic")
    forecast.add_argument("--data", required=True)
    forecast.add_argument("--timestamp-col", default="timestamp")
    forecast.add_argument("--cell-id-col", default="cell_id")
    forecast.add_argument("--aggregate", choices=["10min", "hourly"], default="hourly")
    forecast.add_argument("--cell-id", default=None)
    forecast.add_argument("--kpi-col", default=None)
    forecast.add_argument("--test-size", type=float, default=0.2)
    forecast.add_argument("--horizon", type=int, default=24)
    forecast.add_argument("--lags", default="1,2,3,6,12")
    forecast.add_argument("--output-dir", default="reports/forecast_examples/latest")
    forecast.add_argument("--random-state", type=int, default=42)

    synthetic = subparsers.add_parser("generate-synthetic", help="Generate synthetic telecom telemetry CSV.")
    synthetic.add_argument("--output", default="data/synthetic_ran_kpi.csv")
    synthetic.add_argument("--cells", type=int, default=3)
    synthetic.add_argument("--periods", type=int, default=168)
    synthetic.add_argument("--freq", default="1h")
    synthetic.add_argument("--seed", type=int, default=42)

    scenario = subparsers.add_parser("scenario-demo", help="Generate a baseline vs congestion scenario dashboard.")
    scenario.add_argument("--output-dir", default="reports/scenarios/latest")
    scenario.add_argument("--cells", type=int, default=3)
    scenario.add_argument("--periods", type=int, default=168)
    scenario.add_argument("--freq", default="1h")
    scenario.add_argument("--seed", type=int, default=42)
    scenario.add_argument("--cell-id", default="CELL_001")
    scenario.add_argument("--kpi-col", default="prb_dl_util")
    scenario.add_argument("--horizon", type=int, default=24)
    scenario.add_argument("--scenario-type", choices=["congestion", "backhaul"], default="congestion")

    portal = subparsers.add_parser("portal", help="Generate the top-level evidence portal page.")
    portal.add_argument("--output", default="reports/index.html")

    return parser


def run_forecast(args: argparse.Namespace) -> int:
    result = run_forecast_pipeline(
        data=args.data,
        dataset_type=args.dataset_type,
        timestamp_col=args.timestamp_col,
        cell_id_col=args.cell_id_col,
        cell_id=args.cell_id,
        kpi_col=args.kpi_col,
        aggregate=args.aggregate,
        test_size=args.test_size,
        horizon=args.horizon,
        lags=args.lags,
        random_state=args.random_state,
    )
    artifacts = write_report_bundle(result, args.output_dir)
    print(f"Cell: {result.cell_id}")
    print(f"Target KPI: {result.target_col}")
    print(f"Model: {result.model_name}")
    print(f"RMSE: {result.metrics['rmse']:.4f}")
    print(f"MAE: {result.metrics['mae']:.4f}")
    print(f"MAPE: {result.metrics['mape']:.2f}%")
    print(f"Report directory: {Path(args.output_dir).resolve()}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


def run_generate_synthetic(args: argparse.Namespace) -> int:
    df = generate_synthetic_telemetry(cells=args.cells, periods=args.periods, freq=args.freq, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")
    return 0


def run_scenario_demo(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = generate_synthetic_telemetry(cells=args.cells, periods=args.periods, freq=args.freq, seed=args.seed)
    if args.scenario_type == "backhaul":
        scenario_data = generate_backhaul_telemetry(
            cells=args.cells,
            periods=args.periods,
            freq=args.freq,
            seed=args.seed,
            affected_cell=args.cell_id,
        )
        scenario_name = "AI-RAN backhaul saturation scenario"
        shock_start = 0.58
        shock_duration = 20
    else:
        scenario_data = generate_congestion_telemetry(
            cells=args.cells,
            periods=args.periods,
            freq=args.freq,
            seed=args.seed,
            congested_cell=args.cell_id,
        )
        scenario_name = "AI-RAN congestion scenario"
        shock_start = 0.62
        shock_duration = 18

    baseline_data_path = output_dir / "baseline_telemetry.csv"
    scenario_data_path = output_dir / f"{args.scenario_type}_telemetry.csv"
    baseline_data.to_csv(baseline_data_path, index=False)
    scenario_data.to_csv(scenario_data_path, index=False)

    baseline_result = run_forecast_pipeline(
        data=str(baseline_data_path),
        dataset_type="generic",
        cell_id=args.cell_id,
        kpi_col=args.kpi_col,
        horizon=args.horizon,
    )
    scenario_result = run_forecast_pipeline(
        data=str(scenario_data_path),
        dataset_type="generic",
        cell_id=args.cell_id,
        kpi_col=args.kpi_col,
        horizon=args.horizon,
    )

    baseline_report_dir = output_dir / "baseline"
    scenario_report_dir = output_dir / args.scenario_type
    dashboard_dir = output_dir / "dashboard"
    write_report_bundle(baseline_result, baseline_report_dir)
    write_report_bundle(scenario_result, scenario_report_dir)
    dashboard_artifacts = write_scenario_dashboard(
        baseline_result=baseline_result,
        congestion_result=scenario_result,
        baseline_dir=baseline_report_dir,
        congestion_dir=scenario_report_dir,
        baseline_telemetry_path=baseline_data_path,
        congestion_telemetry_path=scenario_data_path,
        output_dir=dashboard_dir,
        scenario_name=scenario_name,
        shock_start=shock_start,
        shock_duration=shock_duration,
    )

    metadata = {
        "baseline_data": str(baseline_data_path),
        "scenario_data": str(scenario_data_path),
        "scenario_type": args.scenario_type,
        "shock_start": shock_start,
        "shock_duration": shock_duration,
        "baseline_report": str(baseline_report_dir),
        "scenario_report": str(scenario_report_dir),
        **dashboard_artifacts,
    }
    (output_dir / "scenario_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Scenario dashboard: {dashboard_artifacts['dashboard_html']}")
    print(f"Baseline report: {baseline_report_dir}")
    print(f"Scenario report: {scenario_report_dir}")
    return 0


def run_portal(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    portal_path = write_portal_page(output)
    print(f"Wrote portal page to {portal_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "forecast":
        return run_forecast(args)
    if args.command == "generate-synthetic":
        return run_generate_synthetic(args)
    if args.command == "scenario-demo":
        return run_scenario_demo(args)
    if args.command == "portal":
        return run_portal(args)

    # Backward-compatible default: run the sample forecast when no subcommand is provided.
    return run_forecast(
        argparse.Namespace(
            data="data/ran_kpi_sample.csv",
            dataset_type="generic",
            timestamp_col="timestamp",
            cell_id_col="cell_id",
            aggregate="hourly",
            cell_id="CELL_001",
            kpi_col="prb_dl_util",
            test_size=0.2,
            horizon=24,
            lags="1,2,3,6,12",
            output_dir="reports/forecast_examples/latest",
            random_state=42,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
