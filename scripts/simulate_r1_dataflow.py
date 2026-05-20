"""End-to-end R1 → forecast → A1 dataflow simulation.

Demonstrates the rApp's input/output planes on synthetic data without
exercising any wire protocol:

    1. Read a KPM-shaped CSV (matches schemas/kpm_input_v1.json).
    2. Run the forecasting pipeline (any of ridge / gradient_boosting / mlp).
    3. Build an A1 traffic-steering policy candidate (matches schemas/a1_policy_v1.json).
    4. Write both the forecast bundle and the policy candidate to disk.

The point is to show that the rApp's input contract, ML pipeline, and
output contract all connect end-to-end as a single dataflow. Wire-protocol
integration (R1 service registration, A1-P transport) is intentionally
out of scope — see ``docs/AI_RAN_INTEGRATION.md`` for that boundary.

Run::

    python scripts/simulate_r1_dataflow.py \\
        --kpm-input data/ran_kpi_sample.csv \\
        --cell-id CELL_001 \\
        --kpi-col prb_dl_util \\
        --threshold-pct 80 \\
        --output-dir reports/r1_dataflow_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_ran_kpi_forecasting.a1_policy import build_a1_policy_candidate
from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the rApp's R1 → forecast → A1 dataflow.",
    )
    parser.add_argument(
        "--kpm-input",
        type=Path,
        default=Path("data/ran_kpi_sample.csv"),
        help="KPM-shaped CSV input (matches schemas/kpm_input_v1.json).",
    )
    parser.add_argument("--cell-id", type=str, default="CELL_001")
    parser.add_argument("--kpi-col", type=str, default="prb_dl_util")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--model-name",
        type=str,
        default="ridge_linear",
        choices=["ridge_linear", "gradient_boosting", "mlp"],
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=80.0,
        help="Forecast peak threshold (%) above which the policy recommends offload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/r1_dataflow_demo"),
        help="Directory for the demo output bundle.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== R1 input ===")
    print(f"  KPM-shaped CSV: {args.kpm_input}")
    print(f"  Cell: {args.cell_id} · KPI: {args.kpi_col} · Model: {args.model_name}")

    print("=== Forecast (rApp ML pipeline) ===")
    result = run_forecast_pipeline(
        data=str(args.kpm_input),
        dataset_type="generic",
        cell_id=args.cell_id,
        kpi_col=args.kpi_col,
        horizon=args.horizon,
        model_name=args.model_name,
    )
    print(
        f"  Holdout metrics — RMSE {result.metrics['rmse']:.4f} · "
        f"MAE {result.metrics['mae']:.4f} · "
        f"MAPE {result.metrics['mape']:.4f}%"
    )
    peak_y = float(result.forecast["y_hat"].max())
    peak_ts = result.forecast.loc[result.forecast["y_hat"].idxmax(), "timestamp"]
    print(f"  Forecast peak: {peak_y:.2f} at {peak_ts}")

    print("=== A1 policy output (candidate, not enforced) ===")
    policy = build_a1_policy_candidate(result, threshold_pct=args.threshold_pct)
    print(f"  policy_id:    {policy['policy_id']}")
    print(f"  policy_type:  {policy['policy_type']}")
    print(f"  action:       {policy['recommendation']['action']}")
    print(f"  rationale:    {policy['recommendation']['rationale']}")

    # Persist the dataflow bundle.
    policy_path = args.output_dir / "a1_policy_candidate.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    forecast_path = args.output_dir / "forecast.csv"
    result.forecast.to_csv(forecast_path, index=False)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": result.model_name,
                "cell_id": result.cell_id,
                "target_col": result.target_col,
                **{k: float(v) for k, v in result.metrics.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = args.output_dir / "dataflow_summary.md"
    summary_path.write_text(
        f"""# R1 → forecast → A1 dataflow demo

This bundle is the end-to-end output of the rApp's data plane on
**synthetic / sample** telemetry. The wire protocol (R1 service
registration, A1-P transport) is **not** exercised.

| Stage | Artifact |
|---|---|
| R1 input | `{args.kpm_input}` (matches `schemas/kpm_input_v1.json`) |
| Forecast | [`forecast.csv`](forecast.csv) — {len(result.forecast)} steps |
| Metrics | [`metrics.json`](metrics.json) — RMSE {result.metrics['rmse']:.4f} |
| A1 output | [`a1_policy_candidate.json`](a1_policy_candidate.json) (matches `schemas/a1_policy_v1.json`) |

Action recommended: **{policy['recommendation']['action']}**

Reproduce: `python scripts/simulate_r1_dataflow.py --output-dir {args.output_dir}`
""",
        encoding="utf-8",
    )

    print("=== Outputs ===")
    print(f"  {policy_path}")
    print(f"  {forecast_path}")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
