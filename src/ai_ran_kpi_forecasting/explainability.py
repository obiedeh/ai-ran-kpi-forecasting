"""Optional SHAP explainability outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_shap_summary(model, X_sample: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Write a SHAP beeswarm plot using a linear explainer for RidgeForecastRegressor."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return None

    if not hasattr(model, "coef_") or model.coef_ is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        X_arr = np.asarray(X_sample, dtype=float)
        X_scaled = (X_arr - model.mean_) / model.scale_
        explainer = shap.LinearExplainer(
            (model.coef_, model.intercept_),
            X_scaled,
            feature_names=list(X_sample.columns),
        )
        shap_values = explainer(X_scaled)
        shap.plots.beeswarm(shap_values, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()
    except Exception:
        return None
    return output_path
