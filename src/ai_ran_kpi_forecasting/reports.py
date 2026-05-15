"""Report generation for reproducible forecast runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_ran_kpi_forecasting.forecasting import ForecastRunResult
from ai_ran_kpi_forecasting.explainability import write_shap_summary
from ai_ran_kpi_forecasting.visualization import plot_feature_importance, plot_forecast, plot_pre_post_impact


def write_report_bundle(result: ForecastRunResult, output_dir: str | Path) -> dict[str, str]:
    """Write CSV, JSON, Markdown, and plot artifacts for one forecast run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_csv = output_dir / f"{result.target_col}_forecast.csv"
    holdout_csv = output_dir / f"{result.target_col}_holdout.csv"
    metrics_json = output_dir / "metrics.json"
    metrics_md = output_dir / "metrics_summary.md"
    forecast_svg = output_dir / f"{result.target_col}_forecast.svg"
    impact_svg = output_dir / f"{result.target_col}_impact.svg"
    importance_csv = output_dir / "feature_importance.csv"
    importance_svg = output_dir / "feature_importance.svg"
    shap_svg = output_dir / "shap_summary.svg"

    result.forecast.to_csv(forecast_csv, index=False)
    result.holdout.to_csv(holdout_csv, index=False)
    result.feature_importance.to_csv(importance_csv, index=False)
    metrics_json.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")
    plot_forecast(result.holdout, result.forecast, forecast_svg, result.target_col)
    plot_pre_post_impact(result.holdout, result.forecast, impact_svg, result.target_col)
    plot_feature_importance(result.feature_importance, importance_svg)
    shap_path = write_shap_summary(result.model, result.feature_sample, shap_svg)

    metrics_lines = [
        "# Forecast Metrics Summary",
        "",
        f"- Cell: `{result.cell_id}`",
        f"- Target KPI: `{result.target_col}`",
        f"- Model: `{result.model_name}`",
        f"- RMSE: `{result.metrics['rmse']:.4f}`",
        f"- MAE: `{result.metrics['mae']:.4f}`",
        f"- MAPE: `{result.metrics['mape']:.2f}%`",
        "",
        "These artifacts provide a reproducible benchmark for AI-RAN KPI observability workflows.",
    ]
    metrics_md.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    return {
        "forecast_csv": str(forecast_csv),
        "holdout_csv": str(holdout_csv),
        "metrics_json": str(metrics_json),
        "metrics_markdown": str(metrics_md),
        "forecast_plot": str(forecast_svg),
        "impact_plot": str(impact_svg),
        "feature_importance_csv": str(importance_csv),
        "shap_summary": str(shap_path) if shap_path is not None else "",
    }


def _svg_content(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_scenario_dashboard(
    baseline_result: ForecastRunResult,
    congestion_result: ForecastRunResult,
    baseline_dir: str | Path,
    congestion_dir: str | Path,
    baseline_telemetry_path: str | Path,
    congestion_telemetry_path: str | Path,
    output_dir: str | Path,
    scenario_name: str = "Congestion scenario",
    shock_start: float = 0.62,
    shock_duration: int = 18,
) -> dict[str, str]:
    """Write a compact operational dashboard comparing baseline and congestion runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(baseline_dir)
    congestion_dir = Path(congestion_dir)
    baseline_telemetry_path = Path(baseline_telemetry_path)
    congestion_telemetry_path = Path(congestion_telemetry_path)

    baseline_impact = baseline_dir / f"{baseline_result.target_col}_impact.svg"
    congestion_impact = congestion_dir / f"{congestion_result.target_col}_impact.svg"
    baseline_forecast = baseline_dir / f"{baseline_result.target_col}_forecast.svg"
    congestion_forecast = congestion_dir / f"{congestion_result.target_col}_forecast.svg"
    dashboard_html = output_dir / "dashboard.html"
    summary_md = output_dir / "dashboard_summary.md"

    baseline_telemetry = pd.read_csv(baseline_telemetry_path)
    congestion_telemetry = pd.read_csv(congestion_telemetry_path)
    baseline_cell = baseline_result.cell_id
    congestion_cell = congestion_result.cell_id
    baseline_cell_df = baseline_telemetry[baseline_telemetry["cell_id"] == baseline_cell].copy()
    congestion_cell_df = congestion_telemetry[congestion_telemetry["cell_id"] == congestion_cell].copy()
    shock_idx = max(0, min(int(len(congestion_cell_df) * shock_start), max(len(congestion_cell_df) - 1, 0)))
    shock_end = min(len(congestion_cell_df), shock_idx + shock_duration)
    pre_window = congestion_cell_df.iloc[max(0, shock_idx - shock_duration) : shock_idx]
    shock_window = congestion_cell_df.iloc[shock_idx:shock_end]

    def _mean(frame: pd.DataFrame, col: str) -> float:
        return float(frame[col].mean()) if not frame.empty else 0.0

    baseline_before = _mean(baseline_cell_df.iloc[:shock_idx], baseline_result.target_col)
    congestion_before = _mean(pre_window, congestion_result.target_col)
    congestion_after = _mean(shock_window, congestion_result.target_col)
    baseline_after = _mean(baseline_cell_df.iloc[shock_idx:shock_end], baseline_result.target_col)
    congestion_tp_before = _mean(pre_window, "throughput_dl_mbps")
    congestion_tp_after = _mean(shock_window, "throughput_dl_mbps")
    congestion_latency_before = _mean(pre_window, "latency_ms")
    congestion_latency_after = _mean(shock_window, "latency_ms")
    baseline_peak = float(baseline_cell_df[baseline_result.target_col].max()) if not baseline_cell_df.empty else 0.0
    congestion_peak = float(congestion_cell_df[congestion_result.target_col].max()) if not congestion_cell_df.empty else 0.0

    delta_before = congestion_after - congestion_before
    delta_after = congestion_result.forecast["y_hat"].mean() - congestion_after
    congestion_score = congestion_result.metrics["rmse"] / max(baseline_result.metrics["rmse"], 1e-9)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{scenario_name}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #dbe4ee;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #2563eb;
      --red: #dc2626;
      --green: #0f766e;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    .small {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
    }}
    .two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .viz {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 8px;
      overflow: hidden;
    }}
    .viz svg {{ width: 100%; height: auto; display: block; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #e0f2fe;
      color: #0c4a6e;
    }}
    .warn {{ color: var(--red); }}
    @media (max-width: 1100px) {{
      .grid, .two {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 760px) {{
      .grid, .two {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{scenario_name}</h1>
    <div class="sub">Wireless KPI observability pack for a baseline cell and a congestion shock on {congestion_result.cell_id}.</div>
    <div class="grid">
      <div class="card"><div class="label">Baseline RMSE</div><div class="value">{baseline_result.metrics['rmse']:.2f}</div><div class="small">{baseline_result.target_col}</div></div>
      <div class="card"><div class="label">Congestion RMSE</div><div class="value">{congestion_result.metrics['rmse']:.2f}</div><div class="small">{congestion_result.target_col}</div></div>
      <div class="card"><div class="label">PRB uplift</div><div class="value {'warn' if delta_before > 0 else ''}">{delta_before:+.2f}</div><div class="small">shock window minus pre-shock</div></div>
      <div class="card"><div class="label">Forecast RMSE ratio</div><div class="value">{congestion_score:.2f}x</div><div class="small">higher than baseline indicates stress</div></div>
    </div>

    <div class="section">
      <div class="pill">Network health summary</div>
      <table>
        <thead>
          <tr><th>Run</th><th>Cell</th><th>Target</th><th>Pre-shock Mean</th><th>Shock Mean</th><th>Peak</th><th>MAE</th></tr>
        </thead>
        <tbody>
          <tr><td>Baseline</td><td>{baseline_result.cell_id}</td><td>{baseline_result.target_col}</td><td>{baseline_before:.2f}</td><td>{baseline_after:.2f}</td><td>{baseline_peak:.2f}</td><td>{baseline_result.metrics['mae']:.2f}</td></tr>
          <tr><td>Congestion</td><td>{congestion_result.cell_id}</td><td>{congestion_result.target_col}</td><td>{congestion_before:.2f}</td><td>{congestion_after:.2f}</td><td>{congestion_peak:.2f}</td><td>{congestion_result.metrics['mae']:.2f}</td></tr>
        </tbody>
      </table>
      <div class="small">Throughput before/after: {congestion_tp_before:.2f} → {congestion_tp_after:.2f} Mbps. Latency before/after: {congestion_latency_before:.2f} → {congestion_latency_after:.2f} ms.</div>
    </div>

    <div class="two">
      <div class="section">
        <div class="pill">Baseline forecast</div>
        <div class="viz">{_svg_content(baseline_forecast)}</div>
      </div>
      <div class="section">
        <div class="pill">Congestion forecast</div>
        <div class="viz">{_svg_content(congestion_forecast)}</div>
      </div>
    </div>

    <div class="two">
      <div class="section">
        <div class="pill">Baseline impact</div>
        <div class="viz">{_svg_content(baseline_impact)}</div>
      </div>
      <div class="section">
        <div class="pill">Congestion impact</div>
        <div class="viz">{_svg_content(congestion_impact)}</div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    dashboard_html.write_text(html, encoding="utf-8")

    summary_md.write_text(
        "\n".join(
            [
                f"# {scenario_name}",
                "",
        f"- Baseline cell: `{baseline_result.cell_id}`",
        f"- Congestion cell: `{congestion_result.cell_id}`",
        f"- Baseline RMSE: `{baseline_result.metrics['rmse']:.4f}`",
        f"- Congestion RMSE: `{congestion_result.metrics['rmse']:.4f}`",
        f"- Baseline target mean: `{baseline_before:.2f}`",
        f"- Congestion pre-shock mean: `{congestion_before:.2f}`",
        f"- Congestion shock mean: `{congestion_after:.2f}`",
        f"- Congestion delta vs pre-shock: `{delta_before:+.2f}`",
        f"- Forecast delta vs baseline: `{delta_after:+.2f}`",
    ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "dashboard_html": str(dashboard_html),
        "dashboard_summary": str(summary_md),
    }
