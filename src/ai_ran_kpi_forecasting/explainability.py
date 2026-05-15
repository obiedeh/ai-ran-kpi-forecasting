"""Optional SHAP explainability outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_shap_summary(model, X_sample: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Write a SHAP summary plot if SHAP supports the installed model stack."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return None

    regressor = model.named_steps.get("regressor")
    if regressor is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        explainer = shap.Explainer(regressor)
        shap_values = explainer(X_sample)
        shap.plots.beeswarm(shap_values, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()
    except Exception:
        return None
    return output_path
