"""Command line interface for AI-RAN KPI forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_ran_kpi_forecasting.data import generate_synthetic_telemetry
from ai_ran_kpi_forecasting.forecasting import run_forecast_pipeline
from ai_ran_kpi_forecasting.reports import write_report_bundle


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "forecast":
        return run_forecast(args)
    if args.command == "generate-synthetic":
        return run_generate_synthetic(args)

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
