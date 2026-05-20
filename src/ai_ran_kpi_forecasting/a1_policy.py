"""Forecast → A1 traffic-steering policy candidate.

Converts a :class:`ForecastRunResult` into a JSON candidate that conforms to
``schemas/a1_policy_v1.json``. The Non-RT RIC's policy producer would emit
exactly this shape over A1-P to the Near-RT RIC — this module is the
producer-side shape, on synthetic / sample forecasts.

If the forecast peak exceeds the configured threshold within the validity
window, ``action`` is ``"load_balance_offload"``; otherwise ``"no_action"``
(the candidate is still emitted so the policy plane has a paper trail of
"we looked, nothing to do").

This module exercises **policy shape**, not the **A1-P wire protocol**.
See ``docs/AI_RAN_INTEGRATION.md`` for the explicit boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_ran_kpi_forecasting.forecast import ForecastRunResult


def _now_iso() -> str:
    """ISO-8601 UTC timestamp without microseconds (the A1 producer convention)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _to_iso(dt: object) -> str:
    """Coerce pandas timestamp / datetime / string into the A1 policy ISO format."""
    # pandas.Timestamp duck-types: has .isoformat()
    iso = getattr(dt, "isoformat", lambda: str(dt))()
    if iso.endswith("+00:00"):
        return iso.replace("+00:00", "Z")
    if "+" not in iso and "Z" not in iso:
        # Assume UTC if naive — the rApp's data contract requires UTC timestamps.
        return iso + "Z"
    return iso


def build_a1_policy_candidate(
    result: ForecastRunResult,
    threshold_pct: float = 80.0,
    validity_duration_minutes: int | None = None,
) -> dict:
    """Build an A1 traffic-steering policy candidate from a forecast result.

    Parameters
    ----------
    result:
        A completed forecast (output of ``run_forecast_pipeline``).
    threshold_pct:
        If the predicted peak value of ``result.target_col`` exceeds this
        threshold within the validity window, the policy recommends
        ``load_balance_offload``. Otherwise ``no_action``.
        Default of 80 % is the conventional PRB-utilization congestion
        threshold used in textbook Non-RT RIC examples.
    validity_duration_minutes:
        Override for the validity duration. If ``None``, the duration is
        inferred from the number of forecast steps multiplied by the
        approximate step size (assumed 60 minutes per step for hourly
        telemetry).

    Returns
    -------
    A dict matching ``schemas/a1_policy_v1.json``.
    """
    forecast = result.forecast
    if forecast.empty:
        raise ValueError("Forecast is empty; cannot build A1 policy candidate.")

    peak_idx = forecast["y_hat"].idxmax()
    peak_value = float(forecast.loc[peak_idx, "y_hat"])
    peak_timestamp = _to_iso(forecast.loc[peak_idx, "timestamp"])
    validity_start = _to_iso(forecast["timestamp"].iloc[0])
    n_steps = len(forecast)

    if validity_duration_minutes is None:
        # Infer step size from the timestamps; default to 60 minutes per step.
        if n_steps >= 2:
            step = (
                forecast["timestamp"].iloc[1] - forecast["timestamp"].iloc[0]
            ).total_seconds() / 60.0
            step = max(1.0, step)
        else:
            step = 60.0
        validity_duration_minutes = int(n_steps * step)

    crossed = peak_value > threshold_pct
    if crossed:
        action = "load_balance_offload"
        rationale = (
            f"Forecast {result.target_col} on {result.cell_id} reaches "
            f"{peak_value:.2f} at {peak_timestamp} (model={result.model_name}, "
            f"threshold={threshold_pct:.1f}). Recommend offloading workload to "
            f"neighbor cells with headroom."
        )
    else:
        action = "no_action"
        rationale = (
            f"Forecast {result.target_col} on {result.cell_id} peaks at "
            f"{peak_value:.2f} (model={result.model_name}, "
            f"threshold={threshold_pct:.1f}). No policy action required."
        )

    # policy_id format: <type-shortcode>-<cell>-<validity-start>
    type_shortcode = "ts"  # traffic_steering
    policy_id = f"{type_shortcode}-{result.cell_id}-{validity_start.replace(':', '').replace('-', '')}"

    return {
        "policy_id": policy_id,
        "policy_type": "traffic_steering",
        "policy_type_version": "v1",
        "scope": {"cell_id": result.cell_id},
        "validity_window": {
            "start": validity_start,
            "duration_minutes": validity_duration_minutes,
        },
        "forecast_basis": {
            "model_name": result.model_name,
            "target_kpi": result.target_col,
            "predicted_peak": peak_value,
            "predicted_peak_timestamp": peak_timestamp,
            "threshold_pct": threshold_pct,
            "model_metrics_ref": "reports/forecast_examples/latest/metrics.json",
        },
        "recommendation": {
            "action": action,
            "target_cells": [],  # Near-RT RIC selects from neighbor topology
            "rationale": rationale,
        },
    }
