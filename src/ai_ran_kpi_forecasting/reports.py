"""Report generation for reproducible forecast runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from ai_ran_kpi_forecasting.explainability import write_shap_summary
from ai_ran_kpi_forecasting.forecast import ForecastRunResult
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
        "These artifacts provide a reproducible benchmark for AI-RAN and edge infrastructure forecasting workflows.",
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
        "feature_importance_svg": str(importance_svg),
        "shap_summary": str(shap_path) if shap_path is not None else "",
    }


def _svg_content(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_portal_page(output_path: str | Path) -> Path:
    """Write a lightweight portal that links the publishable report bundles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = output_path.parent
    forecast_root = root / "forecast_examples" / "latest"
    scenarios_root = root / "scenarios" / "latest"

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    cards = [
        {
            "title": "Tech brief (1 page)",
            "desc": "One-page hiring-manager / tech-lead read: what this rApp pattern is, evidence summary, three-model comparison, credibility boundary, try-it-in-5-minutes.",
            "links": [
                ("TECH_BRIEF.md", rel(root.parent / "TECH_BRIEF.md")),
                ("README.md", rel(root.parent / "README.md")),
            ],
        },
        {
            "title": "rApp pattern artifacts",
            "desc": "Code-backed AI-for-RAN claims: rApp manifest, KPM input + A1 policy JSON Schemas, end-to-end R1 → forecast → A1 dataflow demo, integration recipe.",
            "links": [
                ("rapp_manifest.yaml", rel(root.parent / "rapp_manifest.yaml")),
                ("KPM input schema", rel(root.parent / "schemas" / "kpm_input_v1.json")),
                ("A1 policy schema", rel(root.parent / "schemas" / "a1_policy_v1.json")),
                ("R1 dataflow demo", rel(root / "r1_dataflow_demo" / "dataflow_summary.md")),
                ("AI-RAN integration", rel(root.parent / "docs" / "AI_RAN_INTEGRATION.md")),
            ],
        },
        {
            "title": "Three-model comparison",
            "desc": "Ridge / GradientBoosting / MLP head-to-head on the same KPI, same temporal split. Honest small-data finding surfaced; Telecom Italia MI still TO MEASURE.",
            "links": [
                ("Comparison table (MD)", rel(root / "model_comparison" / "comparison_metrics.md")),
                ("Comparison table (CSV)", rel(root / "model_comparison" / "comparison_metrics.csv")),
                ("Comparison overlay SVG", rel(root / "model_comparison" / "comparison_overlay.svg")),
            ],
        },
        {
            "title": "Forecast Evidence Pack",
            "desc": "Baseline KPI forecast, metrics, impact plot, and feature importance for the sample RAN telemetry.",
            "links": [
                ("Metrics summary", rel(forecast_root / "metrics_summary.md")),
                ("Forecast SVG", rel(forecast_root / "prb_dl_util_forecast.svg")),
                ("Impact SVG", rel(forecast_root / "prb_dl_util_impact.svg")),
            ],
        },
        {
            "title": "Congestion Scenario",
            "desc": "Pre-shock versus shock-window evidence showing PRB, throughput, and latency stress on one cell.",
            "links": [
                ("Dashboard", rel(scenarios_root / "dashboard" / "dashboard.html")),
                ("Scenario summary", rel(scenarios_root / "dashboard" / "dashboard_summary.md")),
            ],
        },
        {
            "title": "Backhaul Scenario",
            "desc": "Backhaul saturation example with throughput collapse and latency growth under constrained transport.",
            "links": [
                ("Dashboard", rel(scenarios_root / "backhaul" / "dashboard" / "dashboard.html")),
                ("Scenario summary", rel(scenarios_root / "backhaul" / "dashboard" / "dashboard_summary.md")),
            ],
        },
        {
            "title": "Cell Outage Scenario",
            "desc": "Cell-level outage and recovery example with throughput collapse, low PRB, and packet loss spikes.",
            "links": [
                ("Dashboard", rel(scenarios_root / "outage" / "dashboard" / "dashboard.html")),
                ("Scenario summary", rel(scenarios_root / "outage" / "dashboard" / "dashboard_summary.md")),
            ],
        },
        {
            "title": "Data Contracts",
            "desc": "Telemetry assumptions and synthetic data contract for reproducible runs.",
            "links": [
                ("Data contract", rel(root.parent / "DATA_CONTRACT.md")),
                ("Scenario README", rel(root / "scenarios" / "README.md")),
                ("Forecast README", rel(root / "forecast_examples" / "README.md")),
            ],
        },
        {
            "title": "Release Bundle",
            "desc": "Static landing page for GitHub Pages, release uploads, or other publishable report hosting.",
            "links": [
                ("Publish page", rel(root / "publish" / "latest" / "index.html")),
                ("Manifest", rel(root / "publish" / "latest" / "manifest.json")),
            ],
        },
    ]

    def render_links(links: list[tuple[str, str]]) -> str:
        items = "".join(f'<li><a href="{href}">{label}</a></li>' for label, href in links)
        return f"<ul>{items}</ul>"

    html_cards = "\n".join(
        f"""<section class="card">
          <div class="eyebrow">{card['title']}</div>
          <p>{card['desc']}</p>
          {render_links(cast(list[tuple[str, str]], card['links']))}
        </section>"""
        for card in cards
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI-RAN KPI Forecasting Portal</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #dbe4ee;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #2563eb;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background:
        linear-gradient(180deg, rgba(37, 99, 235, 0.10), rgba(248, 250, 252, 0.0) 240px),
        var(--bg);
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    .sub {{ color: var(--muted); max-width: 900px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 22px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .eyebrow {{ color: var(--blue); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card p {{ color: var(--muted); line-height: 1.5; }}
    ul {{ padding-left: 18px; margin: 0; }}
    li {{ margin: 8px 0; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 880px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI-RAN KPI Forecasting Portal</h1>
    <div class="sub">
      Non-RT RIC rApp pattern for AI-for-RAN KPI forecasting. Schema-typed KPM input → three-model forecasting (Ridge / GBR / MLP) → A1 policy candidate output, plus pre/post scenario evidence for congestion / backhaul saturation / cell outage. Pattern, not deployment — wire-protocol integration with a live Non-RT RIC is documented but not exercised.
    </div>
    <div class="grid">
      {html_cards}
    </div>
    <div class="footer">
      Generated from the repository's report artifacts and synthetic telecom telemetry utilities.
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_publish_page(output_dir: str | Path) -> Path:
    """Write a release-friendly landing page for the report artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "portal": "../../index.html",
        "forecast_pack": "../../forecast_examples/latest/metrics_summary.md",
        "congestion": "../../scenarios/latest/congestion/dashboard/dashboard.html",
        "backhaul": "../../scenarios/latest/backhaul/dashboard/dashboard.html",
        "outage": "../../scenarios/latest/outage/dashboard/dashboard.html",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI-RAN KPI Forecasting Release Bundle</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ color: #475569; line-height: 1.55; }}
    .panel {{ background: #fff; border: 1px solid #dbe4ee; border-radius: 12px; padding: 18px 20px; margin-top: 18px; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 8px 0; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI-RAN KPI Forecasting Release Bundle</h1>
    <p>This page is the release-friendly entry point for the report artifacts. It is intended for GitHub Pages, release uploads, or any static hosting path that wants a single landing page.</p>
    <div class="panel">
      <strong>Included evidence</strong>
      <ul>
        <li><a href="../../index.html">Top-level portal</a></li>
        <li><a href="{manifest['forecast_pack']}">Forecast evidence pack</a></li>
        <li><a href="{manifest['congestion']}">Congestion scenario dashboard</a></li>
        <li><a href="{manifest['backhaul']}">Backhaul scenario dashboard</a></li>
        <li><a href="{manifest['outage']}">Cell outage scenario dashboard</a></li>
      </ul>
    </div>
    <div class="panel">
      <strong>Manifest</strong>
      <p>The bundle writes a machine-readable <code>manifest.json</code> alongside this page.</p>
    </div>
  </div>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


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
    <div class="sub">Wireless KPI observability pack comparing a baseline cell with a pre/post scenario event on {congestion_result.cell_id}.</div>
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
      <div class="small">Throughput before/after: {congestion_tp_before:.2f} to {congestion_tp_after:.2f} Mbps. Latency before/after: {congestion_latency_before:.2f} to {congestion_latency_after:.2f} ms.</div>
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
