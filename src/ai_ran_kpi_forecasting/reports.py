"""Report generation for reproducible forecast runs."""

from __future__ import annotations

import json
from pathlib import Path

from ai_ran_kpi_forecasting.forecasting import ForecastRunResult
from ai_ran_kpi_forecasting.explainability import write_shap_summary
from ai_ran_kpi_forecasting.visualization import plot_feature_importance, plot_forecast


def write_report_bundle(result: ForecastRunResult, output_dir: str | Path) -> dict[str, str]:
    """Write CSV, JSON, Markdown, and plot artifacts for one forecast run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_csv = output_dir / f"{result.target_col}_forecast.csv"
    holdout_csv = output_dir / f"{result.target_col}_holdout.csv"
    metrics_json = output_dir / "metrics.json"
    metrics_md = output_dir / "metrics_summary.md"
    forecast_svg = output_dir / f"{result.target_col}_forecast.svg"
    importance_csv = output_dir / "feature_importance.csv"
    importance_svg = output_dir / "feature_importance.svg"
    shap_svg = output_dir / "shap_summary.svg"

    result.forecast.to_csv(forecast_csv, index=False)
    result.holdout.to_csv(holdout_csv, index=False)
    result.feature_importance.to_csv(importance_csv, index=False)
    metrics_json.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")
    plot_forecast(result.holdout, result.forecast, forecast_svg, result.target_col)
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
        "feature_importance_csv": str(importance_csv),
        "shap_summary": str(shap_path) if shap_path is not None else "",
    }
