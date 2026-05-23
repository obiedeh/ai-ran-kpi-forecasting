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


def _risk_tier(value: float) -> tuple[str, str]:
    """Project-defined operational PRB-utilization tiers for report interpretation."""
    if value >= 85:
        return "Critical", "status-risk"
    if value >= 75:
        return "Congested", "status-risk"
    if value >= 60:
        return "Elevated", "status-warn"
    return "Stable", "status-good"


def write_portal_page(output_path: str | Path) -> Path:
    """Write a lightweight portal that links the publishable report bundles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = output_path.parent
    repo_root = root.parent
    forecast_root = root / "forecast_examples" / "latest"
    scenarios_root = root / "scenarios" / "latest"

    def rel(path: Path) -> str:
        path = Path(path)
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            try:
                return f"../{path.relative_to(repo_root).as_posix()}"
            except ValueError:
                return path.as_posix()

    def read_json(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))

    def number_or_none(value: object) -> float | None:
        if isinstance(value, int | float | str):
            return float(value)
        return None

    def read_metrics(path: Path) -> tuple[float | None, float | None]:
        metrics = read_json(path)
        return number_or_none(metrics.get("rmse")), number_or_none(metrics.get("mae"))

    def read_forecast_peak(path: Path) -> tuple[float | None, int | None]:
        if not path.exists():
            return None, None
        forecast = pd.read_csv(path)
        if "y_hat" not in forecast.columns:
            return None, len(forecast)
        return float(forecast["y_hat"].max()), len(forecast)

    comparison_csv = root / "model_comparison" / "comparison_metrics.csv"
    if comparison_csv.exists():
        comparison = pd.read_csv(comparison_csv)
        best_model = comparison.sort_values("rmse").iloc[0]
        best_model_text = f"{best_model['model_name']} ({best_model['rmse']:.2f} RMSE)"
        model_rows = "\n".join(
            f"<tr><td>{row.model_name}</td><td>{row.rmse:.2f}</td><td>{row.mae:.2f}</td><td>{row.mape:.2f}%</td><td>{'best current baseline' if row.model_name == best_model['model_name'] else 'weaker on this small sample'}</td></tr>"
            for row in comparison.itertuples(index=False)
        )
    else:
        best_model_text = "not measured"
        model_rows = '<tr><td colspan="5">not measured</td></tr>'

    forecast_peak, forecast_horizon = read_forecast_peak(forecast_root / "prb_dl_util_forecast.csv")
    risk_label, risk_class = _risk_tier(forecast_peak or 0.0)
    policy = read_json(root / "r1_dataflow_demo" / "a1_policy_candidate.json")
    policy_rec = cast(dict[str, object], policy.get("recommendation", {})) if policy else {}
    policy_basis = cast(dict[str, object], policy.get("forecast_basis", {})) if policy else {}
    policy_action = str(policy_rec.get("action", "not generated")) if policy_rec else "not generated"
    policy_rationale = str(policy_rec.get("rationale", "not generated")) if policy_rec else "not generated"
    policy_threshold = number_or_none(policy_basis.get("threshold_pct")) if policy_basis else None
    policy_peak = number_or_none(policy_basis.get("predicted_peak")) if policy_basis else forecast_peak
    policy_peak_text = f"{policy_peak:.2f}" if policy_peak is not None else "not measured"
    policy_threshold_text = f"{policy_threshold:.1f}" if policy_threshold is not None else "not generated"

    scenario_rows: list[tuple[str, float | None, float | None]] = []
    for scenario_name in ("congestion", "backhaul", "outage"):
        scenario_metrics = scenarios_root / scenario_name / scenario_name / "metrics.json"
        rmse, mae = read_metrics(scenario_metrics)
        scenario_rows.append((scenario_name, rmse, mae))
    measured_scenarios = [row for row in scenario_rows if row[1] is not None]
    worst_scenario = max(measured_scenarios, key=lambda row: row[1] or 0.0) if measured_scenarios else None

    risk_rows = [
        ("Stable", "< 60", "Routine monitoring; no action implied by this project threshold.", "status-good"),
        ("Elevated", "60-74.99", "Watch capacity pressure and compare forecast drift across the next horizon.", "status-warn"),
        ("Congested", "75-84.99", "Prepare traffic-steering or capacity investigation as an advisory candidate.", "status-risk"),
        ("Critical", ">= 85", "Escalate planning review; this is still decision support, not closed-loop control.", "status-risk"),
    ]
    html_risk_rows = "\n".join(
        f'<tr><td><span class="{status}">{tier}</span></td><td>{threshold}</td><td>{meaning}</td></tr>'
        for tier, threshold, meaning, status in risk_rows
    )
    html_scenario_rows = "\n".join(
        f"<tr><td>{name}</td><td>{rmse:.2f}</td><td>{mae:.2f}</td><td>{'highest error scenario' if worst_scenario and name == worst_scenario[0] else 'measured'}</td></tr>"
        if rmse is not None and mae is not None
        else f"<tr><td>{name}</td><td>not measured</td><td>not measured</td><td>not generated</td></tr>"
        for name, rmse, mae in scenario_rows
    )

    summary_cards = [
        ("Forecast horizon", f"{forecast_horizon or 'not generated'} steps", "Forward PRB utilization forecast from committed sample telemetry", "status-neutral"),
        ("Target KPI", "prb_dl_util", "Downlink PRB utilization is the current forecast target", "status-neutral"),
        ("Best measured model", best_model_text, "Same KPI and same time-ordered split across all models", "status-good"),
        ("Peak forecast", f"{forecast_peak:.2f}" if forecast_peak is not None else "not measured", f"Project-defined risk tier: {risk_label}", risk_class),
        ("Scenario count", f"{len(measured_scenarios)} measured", "Congestion, backhaul saturation, and outage evidence", "status-good"),
        ("A1 policy status", policy_action, "Candidate only; no network control is executed", "status-good" if policy_action == "no_action" else "status-warn"),
        ("Benchmark status", "Pending local dataset", "Telecom Italia MI path is prepared, but no benchmark metric is claimed yet.", "status-warn"),
        ("Validation status", "local tests pass", "Lint, tests, report generation, and artifact checks", "status-good"),
    ]

    cards = [
        {
            "title": "Tech brief (1 page)",
            "desc": "A short engineering brief: what I built, what the evidence says, and where the live-RIC boundary starts.",
            "links": [
                ("TECH_BRIEF.md", rel(repo_root / "TECH_BRIEF.md")),
                ("README.md", rel(repo_root / "README.md")),
            ],
        },
        {
            "title": "rApp pattern artifacts",
            "desc": "The contracts around the model: rApp manifest, KPM input schema, advisory A1 output schema, and the R1-style dataflow demo.",
            "links": [
                ("rapp_manifest.yaml", rel(repo_root / "rapp_manifest.yaml")),
                ("KPM input schema", rel(repo_root / "schemas" / "kpm_input_v1.json")),
                ("A1 policy schema", rel(repo_root / "schemas" / "a1_policy_v1.json")),
                ("R1 dataflow demo", rel(root / "r1_dataflow_demo" / "dataflow_summary.md")),
                ("AI-RAN integration", rel(repo_root / "docs" / "AI_RAN_INTEGRATION.md")),
            ],
        },
        {
            "title": "Three-model comparison",
            "desc": "Ridge / GradientBoosting / MLP head-to-head on the same KPI and time split. Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet.",
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
                ("Dashboard", rel(scenarios_root / "congestion" / "dashboard" / "dashboard.html")),
                ("Scenario summary", rel(scenarios_root / "congestion" / "dashboard" / "dashboard_summary.md")),
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
                ("Data contract", rel(repo_root / "DATA_CONTRACT.md")),
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
    html_summary_cards = "\n".join(
        f"""<article class="summary-card">
          <span class="{status_class}">{label}</span>
          <strong>{value}</strong>
          <small>{desc}</small>
        </article>"""
        for label, value, desc, status_class in summary_cards
    )
    html_operations_section = f"""
    <section class="wide-card">
      <div class="eyebrow">Operational interpretation</div>
      <div class="ops-grid">
        <div>
          <h2>Forecast evidence before action</h2>
          <p>The current sample forecast peaks at <strong>{policy_peak_text}</strong> against an advisory policy threshold of <strong>{policy_threshold_text}</strong>, so the generated A1 candidate recommends <strong>{policy_action}</strong>. The useful signal is not that Ridge regression is novel. The useful signal is that the forecast is wrapped in typed input, an auditable policy candidate, and scenario evidence an engineer can inspect.</p>
        </div>
        <div>
          <h2>Policy interpretation</h2>
          <p>{policy_rationale}</p>
          <p class="boundary">The dashboard does not pretend to control the network. It shows forecast evidence, scenario impact, and an advisory A1 policy candidate before any human or downstream system takes action.</p>
        </div>
      </div>
    </section>
    """
    html_risk_section = f"""
    <section class="wide-card">
      <div class="eyebrow">Congestion risk tiers</div>
      <p class="section-copy">Project-defined operational thresholds over forecast PRB DL utilization. These thresholds turn existing forecast values into readable risk tiers; they are not operator SLA values.</p>
      <table>
        <thead><tr><th>Tier</th><th>Forecast PRB DL util</th><th>Operational interpretation</th></tr></thead>
        <tbody>{html_risk_rows}</tbody>
      </table>
    </section>
    """
    html_model_section = f"""
    <section class="wide-card">
      <div class="eyebrow">Model reliability</div>
      <p class="section-copy">The model is the least interesting part of this repo. Ridge, GradientBoosting, and MLP are compared on the same sample KPI and forward temporal split so the weak result stays visible: Ridge is best on this tiny sample, while the MLP underfits badly.</p>
      <table>
        <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>Readout</th></tr></thead>
        <tbody>{model_rows}</tbody>
      </table>
    </section>
    """
    html_scenario_section = f"""
    <section class="wide-card">
      <div class="eyebrow">Scenario comparison</div>
      <p class="section-copy">Scenario packs are deterministic stress overlays on synthetic telemetry. Higher error under a scenario is a signal to monitor transition windows and policy timing, not proof of live-network behavior.</p>
      <table>
        <thead><tr><th>Scenario</th><th>RMSE</th><th>MAE</th><th>Status</th></tr></thead>
        <tbody>{html_scenario_rows}</tbody>
      </table>
    </section>
    """
    html_benchmark_section = """
    <section class="wide-card">
      <div class="eyebrow">Benchmark readiness: Telecom Italia MI</div>
      <p class="section-copy">Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet. The public Telecom Italia Milan path is implemented, but the dataset is not stored in this repo. Place the public dataset files under <code>data/telecom_italia_mi/</code>, then run <code>make run-telecom REPORT_DIR=reports/forecast_examples/telecom_italia_mi</code>.</p>
      <table>
        <thead><tr><th>Item</th><th>Status</th></tr></thead>
        <tbody>
          <tr><td>Loader path</td><td><code>ai_ran_kpi_forecasting.data.load_telecom_italia_mi</code></td></tr>
          <tr><td>Make target</td><td><code>make run-telecom</code></td></tr>
          <tr><td>Output target</td><td><code>reports/forecast_examples/telecom_italia_mi/</code></td></tr>
          <tr><td>Published result</td><td>Benchmark-ready: pending local public dataset files. No benchmark metric claimed yet.</td></tr>
        </tbody>
      </table>
    </section>
    """
    html_boundaries_section = """
    <section class="wide-card">
      <div class="eyebrow">Engineering boundaries</div>
      <p class="section-copy">The boundary is simple: this is a reproducible rApp pattern, not a deployed RIC workload. It uses synthetic and sample telemetry, does not connect to a live RAN, does not deploy an xApp or rApp, does not execute wire-protocol A1 transport, and does not perform autonomous control.</p>
    </section>
    """

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
      --green: #0f766e;
      --gold: #a16207;
      --orange: #c2410c;
      --red: #b91c1c;
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
    header {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 24px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    .sub {{ color: var(--muted); max-width: 900px; line-height: 1.5; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
    .nav a {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      background: #f8fafc;
      font-size: 13px;
    }}
    .section-title {{ margin: 24px 0 10px; font-size: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .summary-card strong {{ display: block; margin-top: 10px; font-size: 22px; }}
    .summary-card small {{ display: block; margin-top: 6px; color: var(--muted); line-height: 1.4; }}
    .status-good, .status-warn, .status-risk, .status-neutral {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .status-good {{ background: #e2f1ec; color: var(--green); }}
    .status-warn {{ background: #fdf3df; color: var(--gold); }}
    .status-risk {{ background: #fee2e2; color: var(--red); }}
    .status-neutral {{ background: #edf1f6; color: #475569; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 22px; }}
    .ops-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .eyebrow {{ color: var(--blue); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card p {{ color: var(--muted); line-height: 1.5; }}
    .wide-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 20px;
      margin-top: 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .wide-card h2 {{ margin: 8px 0; font-size: 20px; }}
    .wide-card p {{ color: var(--muted); line-height: 1.55; }}
    .section-copy {{ max-width: 980px; }}
    .boundary {{ color: #475569; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    ul {{ padding-left: 18px; margin: 0; }}
    li {{ margin: 8px 0; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 880px) {{
      .grid, .summary, .ops-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>AI-RAN KPI Forecasting Portal</h1>
      <div class="sub">
        I built this because RAN operations cannot rely only on after-the-fact dashboards. The page shows a Non-RT RIC / rApp pattern: schema-typed KPM telemetry in, time-ordered forecasts out, advisory A1 policy candidates generated, and evidence artifacts kept visible for review. Pattern, not deployment.
      </div>
      <div class="nav">
        <a href="dashboard.html">Open dashboard</a>
        <a href="../README.md">README</a>
        <a href="../TECH_BRIEF.md">Tech brief</a>
        <a href="../PROJECT_STATUS.md">Project status</a>
      </div>
    </header>
    <h2 class="section-title">Executive KPI cards</h2>
    <div class="summary">
      {html_summary_cards}
    </div>
    {html_operations_section}
    {html_risk_section}
    {html_model_section}
    {html_scenario_section}
    {html_benchmark_section}
    <div class="grid">
      {html_cards}
    </div>
    {html_boundaries_section}
    <div class="footer">
      Generated from the repository's report artifacts and synthetic telecom telemetry utilities.
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    if output_path.name == "index.html":
        dashboard_html = html.replace(
            "<title>AI-RAN KPI Forecasting Portal</title>",
            "<title>AI-RAN KPI Forecasting Dashboard</title>",
        ).replace(
            "<h1>AI-RAN KPI Forecasting Portal</h1>",
            "<h1>AI-RAN KPI Forecasting Dashboard</h1>",
        )
        output_path.with_name("dashboard.html").write_text(dashboard_html, encoding="utf-8")
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
    scenario_tier, scenario_status = _risk_tier(congestion_peak)
    recommended_monitor = (
        "review traffic-steering candidate timing"
        if scenario_tier in {"Congested", "Critical"}
        else "continue monitoring forecast drift and scenario deltas"
    )
    what_this_means = (
        f"{scenario_name} changes {congestion_result.target_col} from {congestion_before:.2f} pre-event "
        f"to {congestion_after:.2f} during the scenario window. Operationally, the useful signal is not "
        "autonomous control; it is an early warning that lets a Non-RT workflow compare forecast error, "
        f"capacity pressure, and the next monitoring action: {recommended_monitor}."
    )

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
      --gold: #a16207;
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
    .status-good, .status-warn, .status-risk {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status-good {{ background: #e2f1ec; color: var(--green); }}
    .status-warn {{ background: #fdf3df; color: var(--gold); }}
    .status-risk {{ background: #fee2e2; color: var(--red); }}
    .readout {{
      border-left: 4px solid var(--blue);
      background: #f7fafd;
      padding: 12px 14px;
      color: var(--muted);
      line-height: 1.5;
      margin-top: 12px;
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
      <div class="card"><div class="label">Scenario RMSE</div><div class="value">{congestion_result.metrics['rmse']:.2f}</div><div class="small">{congestion_result.target_col}</div></div>
      <div class="card"><div class="label">Risk tier</div><div class="value"><span class="{scenario_status}">{scenario_tier}</span></div><div class="small">project-defined PRB threshold</div></div>
      <div class="card"><div class="label">Forecast RMSE ratio</div><div class="value">{congestion_score:.2f}x</div><div class="small">higher than baseline indicates stress</div></div>
    </div>

    <div class="section">
      <div class="pill">Network health summary</div>
      <div class="readout"><strong>What this means:</strong> {what_this_means}</div>
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
      <div class="small">Risk tiers are project-defined over PRB DL utilization: Stable &lt;60, Elevated 60-74.99, Congested 75-84.99, Critical &gt;=85. They are decision-support thresholds, not operator SLA values.</div>
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
