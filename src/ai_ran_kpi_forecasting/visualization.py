"""Plotting helpers for KPI forecast reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_forecast(
    holdout: pd.DataFrame,
    forecast: pd.DataFrame,
    output_path: str | Path,
    target_col: str,
) -> Path:
    """Write a compact actual-vs-predicted and forward forecast plot as SVG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual = holdout["actual"].to_list()
    predicted = holdout["prediction"].to_list()
    future = forecast["y_hat"].to_list()
    values = [value for seq in (actual, predicted, future) for value in seq]
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_max = y_min + 1.0

    width = 980
    height = 520
    pad = 60
    plot_w = width - pad * 2
    plot_h = height - pad * 2

    def x_pos(idx: int, count: int) -> float:
        if count <= 1:
            return pad
        return pad + (idx / (count - 1)) * plot_w

    def y_pos(value: float) -> float:
        return pad + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    def points(values: list[float]) -> str:
        count = len(values)
        return " ".join(f"{x_pos(i, count):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))

    start_ts = str(holdout["timestamp"].iloc[0]) if not holdout.empty else ""
    end_ts = str(forecast["timestamp"].iloc[-1]) if not forecast.empty else ""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial, sans-serif" font-size="22" fill="#111827">AI-RAN KPI forecast: {target_col}</text>
  <text x="{pad}" y="58" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">actual, hold-out prediction, and forward forecast</text>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <polyline points="{points(actual)}" fill="none" stroke="#111827" stroke-width="2"/>
  <polyline points="{points(predicted)}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <polyline points="{points(future)}" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>
  <rect x="{pad + 620}" y="{pad}" width="300" height="72" rx="8" fill="#f9fafb" stroke="#e5e7eb"/>
  <line x1="{pad + 636}" y1="{pad + 24}" x2="{pad + 670}" y2="{pad + 24}" stroke="#111827" stroke-width="2"/>
  <text x="{pad + 680}" y="{pad + 28}" font-family="Arial, sans-serif" font-size="12" fill="#111827">actual</text>
  <line x1="{pad + 636}" y1="{pad + 44}" x2="{pad + 670}" y2="{pad + 44}" stroke="#2563eb" stroke-width="2"/>
  <text x="{pad + 680}" y="{pad + 48}" font-family="Arial, sans-serif" font-size="12" fill="#111827">hold-out prediction</text>
  <line x1="{pad + 636}" y1="{pad + 64}" x2="{pad + 670}" y2="{pad + 64}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="{pad + 680}" y="{pad + 68}" font-family="Arial, sans-serif" font-size="12" fill="#111827">forecast</text>
  <text x="{pad}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{start_ts} → {end_ts}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: str | Path, top_n: int = 12) -> Path | None:
    """Write a feature-importance bar chart as SVG when importance exists."""
    if feature_importance.empty:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = feature_importance.head(top_n).sort_values("importance")

    width = 880
    height = max(180, 80 + len(top) * 32)
    pad = 60
    bar_h = 22
    gap = 10
    max_val = max(float(top["importance"].max()), 1e-9)
    rows = []
    for idx, row in enumerate(top.itertuples(index=False)):
        y = pad + idx * (bar_h + gap)
        bar_w = (row.importance / max_val) * (width - pad * 2 - 180)
        rows.append(
            f'<text x="{pad}" y="{y + 16}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{row.feature}</text>'
        )
        rows.append(
            f'<rect x="{pad + 170}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="#2563eb" />'
        )
        rows.append(
            f'<text x="{pad + 178 + bar_w:.1f}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{row.importance:.4f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial, sans-serif" font-size="22" fill="#111827">Model feature importance</text>
  <text x="{pad}" y="58" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">absolute coefficients from the linear baseline</text>
  {"".join(rows)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path
