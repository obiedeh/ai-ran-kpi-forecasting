"""ONNX export parity and manifest tests. Skip when the onnx extra is not installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")

from scripts.export_onnx import INPUT_NAME, export_all  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "ran_kpi_sample.csv"


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("onnx")
    manifest = export_all(DATA, out)
    return out, manifest


def test_manifest_lists_all_three_models_with_hashes(exported) -> None:
    out, manifest = exported
    names = [m["model_name"] for m in manifest["models"]]
    assert names == ["ridge_linear", "gradient_boosting", "mlp"]
    for m in manifest["models"]:
        path = out / m["file"]
        assert path.exists() and path.stat().st_size == m["bytes"]
        assert len(m["sha256"]) == 64
        assert m["input_shape"] == [None, 16]
        assert len(m["feature_names"]) == 16
    assert manifest["dataset"]["rows"] == 48
    assert manifest["opset"] == 17
    assert "not forecast accuracy" in manifest["claim_boundary"]


def test_onnx_predictions_match_sklearn_on_holdout_sample(exported) -> None:
    _, manifest = exported
    for m in manifest["models"]:
        parity = m["onnx_vs_sklearn_parity"]
        assert parity["n_rows"] > 0
        # float32 graph vs float64 sklearn on values in the tens: tree sums are the loosest.
        assert parity["max_abs_diff"] < 1e-2, (m["model_name"], parity)


def test_exported_models_run_in_onnxruntime_with_batch_one(exported) -> None:
    out, manifest = exported
    for m in manifest["models"]:
        sess = ort.InferenceSession(str(out / m["file"]), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 16), dtype=np.float32)
        y = sess.run(None, {INPUT_NAME: x})[0]
        assert np.asarray(y).shape[0] == 1


def test_sample_metrics_match_committed_comparison(exported) -> None:
    """The export retrains with the comparison settings, so metrics must agree."""
    _, manifest = exported
    committed = (Path(__file__).resolve().parents[1] / "reports" / "model_comparison" / "comparison_metrics.csv")
    if not committed.exists():
        pytest.skip("comparison_metrics.csv not present")
    rows = {}
    for line in committed.read_text().splitlines()[1:]:
        name, _cell, _target, rmse, mae, mape = line.split(",")
        rows[name] = float(rmse)
    for m in manifest["models"]:
        assert m["sample_metrics"]["rmse"] == pytest.approx(rows[m["model_name"]], rel=1e-6)


def test_manifest_json_roundtrip(exported) -> None:
    out, _ = exported
    data = json.loads((out / "manifest.json").read_text())
    assert data["training"]["cell_id"] == "CELL_001"
