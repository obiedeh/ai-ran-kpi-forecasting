"""Export the three sample-trained forecasters to ONNX for edge inference benchmarks.

Retrains Ridge, GradientBoosting and MLP on the committed sample dataset with
the exact settings used by ``scripts/run_model_comparison.py`` (CELL_001,
``prb_dl_util``, horizon 24, lags 1,2,3,6,12, temporal split), converts each
fitted model to ONNX (opset 17), verifies the ONNX output against the
scikit-learn prediction on the hold-out feature sample, and writes a manifest
with dataset and file hashes.

The exports exist so the models can be benchmarked on edge hardware with the
same inference harness used in ``jetson-edge-ai-security``. They carry no
accuracy claim beyond the sample-data metrics already in
``reports/model_comparison/``; forecast accuracy on public telecom data is
unmeasured until the Telecom Italia MI run exists.

Run::

    python scripts/export_onnx.py --output-dir models/exports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ai_ran_kpi_forecasting.forecast import ForecastRunResult, run_forecast_pipeline
from ai_ran_kpi_forecasting.models import MODEL_NAMES

INPUT_NAME = "X"
OPSET = 17
DEFAULTS = {"cell_id": "CELL_001", "kpi_col": "prb_dl_util", "horizon": 24, "lags": "1,2,3,6,12"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _to_sklearn_estimator(result: ForecastRunResult) -> Any:
    """Return a plain scikit-learn estimator equivalent to the fitted wrapper."""
    model = result.model
    name = result.model_name
    if name == "ridge_linear":
        # RidgeForecastRegressor standardizes then applies a linear map. Fold the
        # standardization into the coefficients so the ONNX graph is one MatMul+Add:
        #   y = b + ((x - mean) / scale) @ w  ==  (b - (mean/scale) @ w) + x @ (w/scale)
        from sklearn.linear_model import LinearRegression

        coef = np.asarray(model.coef_, dtype=float) / np.asarray(model.scale_, dtype=float)
        intercept = float(model.intercept_) - float(np.asarray(model.mean_, dtype=float) @ coef)
        lr = LinearRegression()
        lr.coef_ = coef
        lr.intercept_ = intercept
        lr.n_features_in_ = coef.shape[0]
        return lr
    if name == "gradient_boosting":
        return model._model
    if name == "mlp":
        from sklearn.pipeline import Pipeline

        return Pipeline([("scaler", model._scaler), ("mlp", model._model)])
    raise ValueError(f"Unknown model_name {name!r}")


def _convert(estimator: Any, n_features: int) -> bytes:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(
        estimator,
        initial_types=[(INPUT_NAME, FloatTensorType([None, n_features]))],
        target_opset=OPSET,
    )
    return bytes(onnx_model.SerializeToString())


def _parity(onnx_bytes: bytes, result: ForecastRunResult) -> dict[str, float]:
    import onnxruntime as ort

    x = result.feature_sample.to_numpy(dtype=np.float32)
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    onnx_pred = np.asarray(sess.run(None, {INPUT_NAME: x})[0], dtype=np.float64).reshape(-1)
    sk_pred = np.asarray(result.model.predict(result.feature_sample), dtype=np.float64).reshape(-1)
    diff = np.abs(onnx_pred - sk_pred)
    return {
        "n_rows": int(len(x)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / np.maximum(np.abs(sk_pred), 1e-9)).max()),
    }


def export_all(data: Path, output_dir: Path, **pipeline_kwargs: Any) -> dict[str, Any]:
    """Train, convert, verify and write every model. Returns the manifest."""
    import onnxruntime as ort
    import skl2onnx
    import sklearn

    params = {**DEFAULTS, **pipeline_kwargs}
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        result = run_forecast_pipeline(
            data=str(data), dataset_type="generic", cell_id=params["cell_id"], kpi_col=params["kpi_col"],
            horizon=int(params["horizon"]), lags=params["lags"], model_name=model_name,
        )
        n_features = len(result.feature_sample.columns)
        onnx_bytes = _convert(_to_sklearn_estimator(result), n_features)
        parity = _parity(onnx_bytes, result)
        out = output_dir / f"{model_name}_{result.target_col}.onnx"
        out.write_bytes(onnx_bytes)
        entries.append({
            "model_name": model_name,
            "file": out.name,
            "sha256": _sha256(out),
            "bytes": out.stat().st_size,
            "input_name": INPUT_NAME,
            "input_shape": [None, n_features],
            "feature_names": list(result.feature_sample.columns),
            "target_col": result.target_col,
            "cell_id": result.cell_id,
            "sample_metrics": {k: round(float(v), 6) for k, v in result.metrics.items()},
            "onnx_vs_sklearn_parity": parity,
        })
        print(f"{model_name:<18} -> {out.name} ({out.stat().st_size} B) max|diff| {parity['max_abs_diff']:.3g}")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "dataset": {"path": str(data), "sha256": _sha256(data), "rows": sum(1 for _ in data.open()) - 1},
        "training": {**params, "split": "temporal, last 20% held out", "seed": 42},
        "opset": OPSET,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "skl2onnx": skl2onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "claim_boundary": (
            "Sample-data exports for edge inference benchmarking. Metrics are on the 48-row synthetic "
            "sample; they are not forecast accuracy on public telecom data."
        ),
        "models": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/ran_kpi_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/exports"))
    args = parser.parse_args()
    manifest = export_all(args.data, args.output_dir)
    print(f"Wrote {args.output_dir / 'manifest.json'} ({len(manifest['models'])} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
