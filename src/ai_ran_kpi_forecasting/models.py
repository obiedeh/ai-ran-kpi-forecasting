"""Model training utilities for KPI forecasting.

Three models are supported, all from scikit-learn / NumPy — zero new
dependencies on top of the existing stack. The point of the three-model
comparison is to surface "which family wins on which KPI" honestly rather
than cherry-picking a single tuned model.

  - ``ridge_linear``       — hand-coded ridge regressor (deterministic, fast,
                              the credibility baseline)
  - ``gradient_boosting``  — sklearn GradientBoostingRegressor (tree ensemble)
  - ``mlp``                — sklearn MLPRegressor (small neural baseline)

All three are wrapped so they share the same fit / predict / feature-importance
contract — the rest of the pipeline doesn't have to special-case any of them.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from ai_ran_kpi_forecasting.metrics import regression_metrics

# Public registry of model names → builder factories. The default model name
# (the baseline) is the first entry. Add new entries here, not in scattered
# build_regressor variants.
MODEL_NAMES = ("ridge_linear", "gradient_boosting", "mlp")
DEFAULT_MODEL = MODEL_NAMES[0]


class RidgeForecastRegressor:
    """Small deterministic regressor with standardization and ridge regularization."""

    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeForecastRegressor:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.mean_ = X_arr.mean(axis=0)
        self.scale_ = X_arr.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        X_scaled = (X_arr - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        X_scaled = (X_arr - self.mean_) / self.scale_
        return cast(np.ndarray, self.intercept_ + X_scaled @ self.coef_)


class _GradientBoostingWrapper:
    """Thin wrapper around sklearn GradientBoostingRegressor.

    Adds deterministic seeding and the .coef_-less feature_importance contract
    so feature_importance_frame() can handle this model uniformly.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 200, max_depth: int = 3):
        from sklearn.ensemble import GradientBoostingRegressor

        self._model = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _GradientBoostingWrapper:
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self._model.predict(np.asarray(X, dtype=float)))

    @property
    def feature_importances_(self) -> np.ndarray:
        return cast(np.ndarray, self._model.feature_importances_)


class _MLPWrapper:
    """Thin wrapper around sklearn MLPRegressor.

    Small two-hidden-layer MLP (32 → 16) — kept intentionally small so the
    comparison is "small neural baseline" not "we threw a large model at it".
    Standardises features before passing to the network so training is stable
    on the raw KPI scales.
    """

    def __init__(self, random_state: int = 42, hidden_layer_sizes: tuple[int, ...] = (32, 16)):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
            tol=1e-5,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MLPWrapper:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_scaled = self._scaler.fit_transform(X_arr)
        self._model.fit(X_scaled, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(np.asarray(X, dtype=float))
        return cast(np.ndarray, self._model.predict(X_scaled))


def build_regressor(model_name: str = DEFAULT_MODEL) -> tuple[Any, str]:
    """Build a regressor by name. Returns ``(model, name)``."""
    if model_name == "ridge_linear":
        return RidgeForecastRegressor(alpha=1e-4), "ridge_linear"
    if model_name == "gradient_boosting":
        return _GradientBoostingWrapper(random_state=42), "gradient_boosting"
    if model_name == "mlp":
        return _MLPWrapper(random_state=42), "mlp"
    raise ValueError(
        f"Unknown model_name {model_name!r}. Expected one of: {MODEL_NAMES}."
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    model_name: str = DEFAULT_MODEL,
) -> tuple[Any, dict[str, float], pd.DataFrame, str]:
    """Train a model and return metrics + hold-out predictions.

    The ``random_state`` parameter is accepted for backwards compatibility;
    deterministic seeding is built into the model wrappers themselves.
    """
    _ = random_state
    model, name = build_regressor(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    holdout = pd.DataFrame(
        {"actual": y_test.to_numpy(), "prediction": y_pred}, index=y_test.index
    )
    return model, metrics, holdout, name


def feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract model-native feature importance when available.

    Supports three shapes:
      - ``.coef_`` (linear models — RidgeForecastRegressor)
      - ``.feature_importances_`` (tree ensembles — GradientBoosting wrapper)
      - neither → return empty frame (used by MLP, whose feature attribution
        would need permutation importance to be meaningful)
    """
    coefs = getattr(model, "coef_", None)
    if coefs is not None:
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": np.abs(coefs),
                    "coefficient": coefs,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": np.asarray(importances, dtype=float),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["feature", "importance"])
