"""Tests for forecast → A1 policy candidate conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_ran_kpi_forecasting.a1_policy import build_a1_policy_candidate
from ai_ran_kpi_forecasting.forecast import run_forecast_pipeline

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "ran_kpi_sample.csv"
_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "a1_policy_v1.json"


def _run_sample_forecast(horizon: int = 6, model_name: str = "ridge_linear"):
    return run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id="CELL_001",
        kpi_col="prb_dl_util",
        horizon=horizon,
        model_name=model_name,
    )


def test_policy_candidate_has_required_top_level_fields():
    """Every A1 policy candidate must have the v1 schema's required keys."""
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result)
    required = {
        "policy_id",
        "policy_type",
        "scope",
        "validity_window",
        "forecast_basis",
        "recommendation",
    }
    assert required <= set(policy.keys())


def test_policy_type_is_traffic_steering():
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result)
    assert policy["policy_type"] == "traffic_steering"
    assert policy["policy_type_version"] == "v1"


def test_policy_scope_references_correct_cell():
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result)
    assert policy["scope"] == {"cell_id": "CELL_001"}


def test_validity_window_has_iso_start_and_positive_duration():
    result = _run_sample_forecast(horizon=12)
    policy = build_a1_policy_candidate(result)
    vw = policy["validity_window"]
    assert "start" in vw and isinstance(vw["start"], str)
    # ISO-8601 with Z suffix per the rApp convention
    assert vw["start"].endswith("Z")
    assert vw["duration_minutes"] >= 1


def test_forecast_basis_references_the_actual_model():
    """The forecast_basis block must carry the model name so an operator
    can trace the policy back to the artifact it was derived from."""
    for name in ("ridge_linear", "gradient_boosting", "mlp"):
        result = _run_sample_forecast(model_name=name)
        policy = build_a1_policy_candidate(result)
        assert policy["forecast_basis"]["model_name"] == name
        assert policy["forecast_basis"]["target_kpi"] == "prb_dl_util"
        assert "predicted_peak" in policy["forecast_basis"]


def test_action_is_offload_when_peak_above_threshold():
    """Set a very low threshold so the forecast peak is guaranteed to exceed it."""
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result, threshold_pct=0.0)
    assert policy["recommendation"]["action"] == "load_balance_offload"
    assert "rationale" in policy["recommendation"]


def test_action_is_no_action_when_peak_below_threshold():
    """Set an unreachably high threshold so the forecast peak cannot cross."""
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result, threshold_pct=1e9)
    assert policy["recommendation"]["action"] == "no_action"


def test_policy_id_is_unique_per_cell_and_validity():
    """Two candidates for different cells must have different policy_ids."""
    result_a = _run_sample_forecast()
    policy_a = build_a1_policy_candidate(result_a)
    # The sample CSV has multiple cells; rerun on a different cell.
    import pandas as pd
    df = pd.read_csv(_SAMPLE_CSV)
    other_cells = [c for c in df["cell_id"].unique() if c != "CELL_001"]
    if not other_cells:
        pytest.skip("Sample CSV has only one cell; can't test uniqueness here.")
    result_b = run_forecast_pipeline(
        data=str(_SAMPLE_CSV),
        cell_id=str(other_cells[0]),
        kpi_col="prb_dl_util",
        horizon=6,
    )
    policy_b = build_a1_policy_candidate(result_b)
    assert policy_a["policy_id"] != policy_b["policy_id"]


def test_policy_matches_committed_schema_required_fields():
    """The schema enumerates required top-level fields; the policy must satisfy them."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    result = _run_sample_forecast()
    policy = build_a1_policy_candidate(result)
    missing = required - set(policy.keys())
    assert not missing, f"Policy is missing schema-required fields: {missing}"
