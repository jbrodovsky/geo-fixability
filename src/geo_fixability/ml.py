"""
Machine-learning utilities for geo-fixability prediction.

Provides:
- build_xgb_pipeline: sklearn Pipeline wrapping XGBoost regressor
- compute_shap_values: SHAP TreeExplainer on a fitted pipeline
- plot_shap_summary: save a SHAP summary bar plot to disk
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_xgb_pipeline(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Return a sklearn Pipeline with StandardScaler + XGBRegressor.

    Drop-in replacement for the GradientBoosting pipeline in analyze_results.py.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default 500.
    max_depth : int
        Maximum tree depth. Default 6.
    learning_rate : float
        Shrinkage factor per round. Default 0.05.
    subsample : float
        Fraction of training rows sampled per round. Default 0.8.
    random_state : int
        Random seed. Default 42.

    Returns
    -------
    Pipeline
        Unfitted pipeline with keys ``"scaler"`` and ``"model"``.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            tree_method="hist",
            random_state=random_state,
            verbosity=0,
        )),
    ])


def compute_shap_values(
    pipeline: Pipeline,
    X: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Compute SHAP values for a fitted XGBoost pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline with ``"scaler"`` and ``"model"`` steps.
    X : ndarray, shape (N, n_features)
        Raw (unscaled) feature matrix.
    feature_names : list[str]
        Names for each column of X.

    Returns
    -------
    dict with keys:

    ``shap_values`` : ndarray, shape (N, n_features)
        SHAP value matrix.
    ``expected_value`` : float
        SHAP base value (mean prediction in log10 space).
    ``feature_names`` : list[str]
        Copy of the input feature names.
    ``X_scaled`` : ndarray, shape (N, n_features)
        Scaled feature matrix (input to TreeExplainer).
    """
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_scaled)

    return {
        "shap_values": sv,
        "expected_value": float(explainer.expected_value),
        "feature_names": list(feature_names),
        "X_scaled": X_scaled,
    }


def build_xgb_quantile_pipeline(
    quantile: float,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Return a sklearn Pipeline with StandardScaler + XGBRegressor for quantile regression.

    Parameters
    ----------
    quantile : float
        Target quantile in (0, 1), e.g. 0.05, 0.50, 0.95.
    n_estimators, max_depth, learning_rate, subsample, random_state :
        Same as ``build_xgb_pipeline``.

    Returns
    -------
    Pipeline
        Unfitted pipeline with keys ``"scaler"`` and ``"model"``.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            tree_method="hist",
            random_state=random_state,
            verbosity=0,
        )),
    ])


def plot_shap_summary(
    shap_dict: dict,
    outpath: str | Path,
    max_display: int = 20,
) -> None:
    """
    Save a SHAP mean-|value| bar plot to disk.

    Parameters
    ----------
    shap_dict : dict
        Output of ``compute_shap_values``.
    outpath : str or Path
        Destination file path for the PNG.
    max_display : int
        Maximum number of features to display. Default 20.
    """
    shap.summary_plot(
        shap_dict["shap_values"],
        features=shap_dict["X_scaled"],
        feature_names=shap_dict["feature_names"],
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
