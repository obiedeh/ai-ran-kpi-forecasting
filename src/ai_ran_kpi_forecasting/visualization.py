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
  <text x="{pad}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{start_ts} to {end_ts}</text>
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


def plot_pre_post_impact(
    holdout: pd.DataFrame,
    forecast: pd.DataFrame,
    output_path: str | Path,
    target_col: str,
) -> Path:
    """Write a pre/post impact summary as SVG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = holdout["actual"].to_list()
    after = forecast["y_hat"].to_list()
    before_mean = sum(before) / max(len(before), 1)
    after_mean = sum(after) / max(len(after), 1)
    delta = after_mean - before_mean
    delta_pct = (delta / before_mean * 100.0) if abs(before_mean) > 1e-9 else 0.0

    width = 1100
    height = 520
    pad = 56
    panel_w = 420
    panel_h = 300
    chart_bottom = pad + panel_h

    def line_points(values: list[float], x0: float, y0: float, w: float, h: float) -> str:
        if not values:
            return ""
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmax = vmin + 1.0
        count = len(values)
        pts = []
        for idx, val in enumerate(values):
            x = x0 + (idx / max(count - 1, 1)) * w
            y = y0 + h - ((val - vmin) / (vmax - vmin)) * h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    before_points = line_points(before[-min(len(before), 24):], pad, pad + 10, panel_w, panel_h - 40)
    after_points = line_points(after[: min(len(after), 24)], pad + panel_w + 120, pad + 10, panel_w, panel_h - 40)
    before_max = max(before) if before else 0.0
    after_max = max(after) if after else 0.0

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="34" font-family="Arial, sans-serif" font-size="24" fill="#111827">Before / After KPI impact</text>
  <text x="{pad}" y="58" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">historical telemetry versus forecast horizon</text>

  <rect x="{pad}" y="{pad + 10}" width="{panel_w}" height="{panel_h}" rx="12" fill="#f9fafb" stroke="#e5e7eb"/>
  <rect x="{pad + panel_w + 120}" y="{pad + 10}" width="{panel_w}" height="{panel_h}" rx="12" fill="#f9fafb" stroke="#e5e7eb"/>
  <text x="{pad + 16}" y="{pad + 36}" font-family="Arial, sans-serif" font-size="14" fill="#111827">Before: actual history</text>
  <text x="{pad + panel_w + 136}" y="{pad + 36}" font-family="Arial, sans-serif" font-size="14" fill="#111827">After: forecast horizon</text>

  <polyline points="{before_points}" fill="none" stroke="#111827" stroke-width="2"/>
  <polyline points="{after_points}" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>

  <line x1="{pad}" y1="{chart_bottom}" x2="{pad + panel_w}" y2="{chart_bottom}" stroke="#d1d5db" stroke-width="1"/>
  <line x1="{pad + panel_w + 120}" y1="{chart_bottom}" x2="{pad + panel_w * 2 + 120}" y2="{chart_bottom}" stroke="#d1d5db" stroke-width="1"/>

  <rect x="{pad}" y="{height - 126}" width="300" height="82" rx="12" fill="#111827"/>
  <text x="{pad + 18}" y="{height - 94}" font-family="Arial, sans-serif" font-size="13" fill="#cbd5e1">Before average</text>
  <text x="{pad + 18}" y="{height - 68}" font-family="Arial, sans-serif" font-size="26" fill="#ffffff">{before_mean:.2f}</text>

  <rect x="{pad + 320}" y="{height - 126}" width="300" height="82" rx="12" fill="#1d4ed8"/>
  <text x="{pad + 338}" y="{height - 94}" font-family="Arial, sans-serif" font-size="13" fill="#dbeafe">After average</text>
  <text x="{pad + 338}" y="{height - 68}" font-family="Arial, sans-serif" font-size="26" fill="#ffffff">{after_mean:.2f}</text>

  <rect x="{pad + 640}" y="{height - 126}" width="380" height="82" rx="12" fill="#f3f4f6" stroke="#e5e7eb"/>
  <text x="{pad + 658}" y="{height - 94}" font-family="Arial, sans-serif" font-size="13" fill="#111827">Impact</text>
  <text x="{pad + 658}" y="{height - 68}" font-family="Arial, sans-serif" font-size="26" fill="#111827">{delta:+.2f} ({delta_pct:+.1f}%)</text>
  <text x="{pad + 658}" y="{height - 44}" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">peak before {before_max:.2f} | peak after {after_max:.2f}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path
