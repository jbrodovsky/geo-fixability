"""
Analyze star-trajectory experiment results and fit a predictive model.

Two subcommands:

  analyze   Load the experiment CSV, generate EDA plots, train a
            GradientBoosting or XGBoost model on map + trajectory features,
            evaluate it, save the trained model, and (for XGBoost) generate
            a SHAP summary plot.

  predict   Load a terrain DEM (numpy .npy file or SRTM .hgt), extract map
            metrics, then sweep over a speed × duration grid and plot the
            predicted CEP fix accuracy.

Usage
-----
    uv run analyze-results analyze
    uv run analyze-results analyze --model-type xgb
    uv run analyze-results analyze --csv data/experiments/star_trajectory_results.csv \\
                                    --outdir data/analysis

    uv run analyze-results predict terrain.npy --dx 30 --sigma-meas 0.1 \\
                                    --sigma-init 500 --model data/analysis/model.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geo_fixability.metrics import map_information_content
from geo_fixability.ml import build_xgb_pipeline, build_xgb_quantile_pipeline, compute_shap_values, plot_shap_summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features derived from map_information_content (computable for real terrain)
MAP_FEATURES = [
    "gradient_rms",
    "information_index",
    "spatial_diversity_index",
    "localization_index",
    "localization_bound_m",
    "correlation_length_m",
]

# Trajectory / experimental settings
TRAJ_FEATURES = [
    "speed_m_s",
    "total_distance_m",
    "sigma_init_m",
]

# Trajectory-map interaction features (from features.extract_all_features)
# Added to address the gap noted in the paper (§III-D)
INTERACTION_FEATURES = [
    "gradient_rms_along_path",
    "gradient_perpendicular_mean",
    "cumulative_fim_trace",
    "effective_dof",
    "dx_m",
]

ALL_FEATURES = MAP_FEATURES + TRAJ_FEATURES

# Extended feature set including trajectory-map interaction features.
# Used by the XGBoost model when the interaction columns are present in the CSV.
ALL_FEATURES_EXTENDED = MAP_FEATURES + TRAJ_FEATURES + INTERACTION_FEATURES

# Primary target: CEP after navigation fix
TARGET = "pcrb_cep_final_m"

# Default sweep grids for prediction
DEFAULT_SPEEDS = [2.0, 5.0, 10.0, 20.0]       # m/s
DEFAULT_DURATIONS = [300.0, 600.0, 1200.0, 2400.0]  # s
N_ARMS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()

    # Remap pipeline.py column names to the names expected by this script
    _remap = {
        "speed": "speed_m_s",
        "duration": "duration_s",
        "sigma_init": "sigma_init_m",
        "final_cep_m": "pcrb_cep_final_m",
        "total_distance": "total_distance_m",
        "dx": "dx_m",
    }
    df = df.rename(columns={k: v for k, v in _remap.items() if k in df.columns and v not in df.columns})

    if "total_distance_m" not in df.columns:
        df["total_distance_m"] = df["speed_m_s"] * df["duration_s"]

    # spatial_diversity_index not yet emitted by pipeline.py; fill with NaN so
    # the extended feature set gracefully falls back to the base set
    if "spatial_diversity_index" not in df.columns:
        df["spatial_diversity_index"] = float("nan")

    return df


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Analyze subcommand
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> None:
    import json

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = _load_data(args.csv)
    print(f"  {len(df)} rows after filtering to status=ok")

    # Choose feature set: extended if interaction columns present and non-empty in CSV
    use_extended = all(
        c in df.columns and df[c].notna().any() for c in INTERACTION_FEATURES
    )
    features = ALL_FEATURES_EXTENDED if use_extended else ALL_FEATURES
    # Drop any feature columns that are entirely NaN (e.g. spatial_diversity_index
    # not yet emitted by the current pipeline)
    features = [f for f in features if f in df.columns and df[f].notna().any()]
    if use_extended:
        print(f"  Extended feature set ({len(features)} features, including interaction features)")
    else:
        print(f"  Base feature set ({len(features)} features)")

    model_type = getattr(args, "model_type", "gbr")

    # -------------------------------------------------------------------
    # EDA: correlation heatmap
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = features + [TARGET]
    corr = df[cols].corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(outdir / "eda_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {outdir / 'eda_correlation.png'}")

    # -------------------------------------------------------------------
    # EDA: CEP vs each feature (one file per feature)
    # -------------------------------------------------------------------
    for feat in features:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[feat], df[TARGET], s=2, alpha=0.2, rasterized=True)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel(TARGET, fontsize=9)
        ax.set_yscale("log")
        ax.set_title(f"Fix CEP vs {feat} (log scale)", fontsize=9)
        fig.tight_layout()
        fname = f"eda_scatter_{feat.replace('/', '_')}.png"
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {outdir / fname}")

    # -------------------------------------------------------------------
    # Train / evaluate model (log-space target for better relative error)
    # -------------------------------------------------------------------
    X = df[features].values
    y = np.log10(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "xgb":
        print(f"\nTraining XGBoost on {len(X_train)} samples ...")
        pipe = build_xgb_pipeline()
    else:
        print(f"\nTraining GradientBoosting on {len(X_train)} samples ...")
        pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test R²  = {r2:.4f}")
    print(f"  Test MAE = {mae:.4f} log10-m  (≈ ×{10**mae:.2f} factor error)")

    # Predicted vs actual plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=6, alpha=0.3, rasterized=True)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual log₁₀(CEP) [m]")
    ax.set_ylabel("Predicted log₁₀(CEP) [m]")
    ax.set_title(f"Model fit  R²={r2:.3f}  MAE={mae:.3f} log-m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "model_fit.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {outdir / 'model_fit.png'}")

    # -------------------------------------------------------------------
    # Feature importance (tree-based) and SHAP (XGBoost only)
    # -------------------------------------------------------------------
    model_obj = pipe.named_steps["model"]
    importance = model_obj.feature_importances_
    order = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(features)), importance[order])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Feature importance")
    model_label = "XGBoost" if model_type == "xgb" else "Gradient Boosting"
    ax.set_title(f"{model_label} feature importance")
    fig.tight_layout()
    fig.savefig(outdir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {outdir / 'feature_importance.png'}")

    if model_type == "xgb":
        print("  Computing SHAP values ...")
        shap_dict = compute_shap_values(pipe, X_test, features)
        shap_path = outdir / "shap_summary.png"
        plot_shap_summary(shap_dict, shap_path)
        print(f"  Saved {shap_path}")

    # -------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------
    model_path = outdir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nModel saved to {model_path}")

    meta = {
        "features": features,
        "target": TARGET,
        "target_transform": "log10",
        "model_type": model_type,
        "r2_test": float(r2),
        "mae_test_log10": float(mae),
    }
    with open(outdir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to {outdir / 'model_meta.json'}")


# ---------------------------------------------------------------------------
# Ablation subcommand (Task 4)
# ---------------------------------------------------------------------------

# Named feature subsets for the ablation study
_ABLATION_GROUPS: dict[str, list[str]] = {
    "all_9":        ALL_FEATURES,
    "map_only":     MAP_FEATURES,
    "traj_only":    TRAJ_FEATURES,
    "spectral_only": ["information_index", "spatial_diversity_index", "localization_index"],
    "drop_spectral": [
        "gradient_rms", "localization_bound_m", "correlation_length_m",
        "speed_m_s", "total_distance_m", "sigma_init_m",
    ],
}


def cmd_ablation(args: argparse.Namespace) -> None:
    """Train XGBoost on feature subsets; report R² relative to baseline."""
    import json

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = _load_data(args.csv)
    print(f"  {len(df)} rows")

    y = np.log10(df[TARGET].values)

    results: list[dict] = []
    for group_name, feats in _ABLATION_GROUPS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  Skipping '{group_name}': missing columns {missing}")
            continue

        X = df[feats].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipe = build_xgb_pipeline()
        pipe.fit(X_train, y_train)
        r2 = r2_score(y_test, pipe.predict(X_test))
        mae = mean_absolute_error(y_test, pipe.predict(X_test))
        print(f"  {group_name:20s}  n_feats={len(feats):2d}  R²={r2:.4f}  MAE={mae:.4f}")
        results.append({"group": group_name, "n_features": len(feats), "r2": r2, "mae": mae})

    df_res = pd.DataFrame(results)
    csv_out = outdir / "ablation_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\nAblation results → {csv_out}")

    baseline_r2 = df_res.loc[df_res["group"] == "all_9", "r2"].values[0]
    df_res["delta_r2"] = df_res["r2"] - baseline_r2

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["steelblue" if g == "all_9" else "salmon" for g in df_res["group"]]
    ax.bar(df_res["group"], df_res["r2"], color=colors)
    ax.axhline(baseline_r2, color="gray", linestyle="--", linewidth=1)
    for idx, row in df_res.iterrows():
        ax.text(idx, row["r2"] + 0.003, f"{row['r2']:.3f}", ha="center", fontsize=9)
    ax.set_ylim(max(0, df_res["r2"].min() - 0.05), 1.01)
    ax.set_ylabel("Test R² (log₁₀ CEP)")
    ax.set_title("Feature ablation study — XGBoost")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(outdir / "ablation_bar.png", dpi=150)
    plt.close(fig)
    print(f"Ablation bar chart → {outdir / 'ablation_bar.png'}")


# ---------------------------------------------------------------------------
# Cross-validation bias subcommand (Task 5)
# ---------------------------------------------------------------------------


def cmd_crossval(args: argparse.Namespace) -> None:
    """Compare random vs stratified holdout to quantify interpolation/extrapolation gap."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = _load_data(args.csv)
    print(f"  {len(df)} rows")

    # Determine feature set
    use_extended = all(c in df.columns for c in INTERACTION_FEATURES)
    features = ALL_FEATURES_EXTENDED if use_extended else ALL_FEATURES

    X = df[features].values
    y = np.log10(df[TARGET].values)

    results: list[dict] = []

    # Random holdout — 5 seeds
    for seed in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        pipe = build_xgb_pipeline()
        pipe.fit(X_train, y_train)
        r2 = r2_score(y_test, pipe.predict(X_test))
        mae = mean_absolute_error(y_test, pipe.predict(X_test))
        results.append({"strategy": f"random_seed{seed}", "type": "interpolation", "r2": r2, "mae": mae})
        print(f"  random_seed{seed}:  R²={r2:.4f}  MAE={mae:.4f}")

    # Stratified holdout — withhold entire levels of key parameters
    stratifications: list[tuple[str, str, object]] = []
    if "dx_m" in df.columns:
        dx_max = df["dx_m"].max()
        stratifications.append(("dx_m", f"dx={dx_max:.0f}m", dx_max))
    if "beta" in df.columns:
        beta_max = df["beta"].max()
        stratifications.append(("beta", f"beta={beta_max}", beta_max))
    if "sigma_init_m" in df.columns:
        sigma_max = df["sigma_init_m"].max()
        stratifications.append(("sigma_init_m", f"sigma_init={sigma_max:.0f}m", sigma_max))

    for col, label, val in stratifications:
        mask_test = df[col] == val
        n_test = int(mask_test.sum())
        n_train = int((~mask_test).sum())
        if n_test < 10:
            print(f"  {label}: too few test samples ({n_test}), skipping")
            continue
        if n_train < 10:
            print(f"  {label}: only one level present — all {n_test} rows match, skipping")
            continue
        X_train, y_train = X[~mask_test.values], y[~mask_test.values]
        X_test, y_test = X[mask_test.values], y[mask_test.values]
        pipe = build_xgb_pipeline()
        pipe.fit(X_train, y_train)
        r2 = r2_score(y_test, pipe.predict(X_test))
        mae = mean_absolute_error(y_test, pipe.predict(X_test))
        results.append({"strategy": f"stratified_{label}", "type": "extrapolation", "r2": r2, "mae": mae})
        print(f"  stratified_{label}:  R²={r2:.4f}  MAE={mae:.4f}  ({mask_test.sum()} test samples)")

    df_res = pd.DataFrame(results)
    csv_out = outdir / "crossval_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\nCross-validation results → {csv_out}")

    mean_interp = df_res[df_res["type"] == "interpolation"]["r2"].mean()
    extrap = df_res[df_res["type"] == "extrapolation"]
    if not extrap.empty:
        mean_extrap = extrap["r2"].mean()
        print(f"\nMean R² (interpolation): {mean_interp:.4f}")
        print(f"Mean R² (extrapolation): {mean_extrap:.4f}")
        print(f"Generalization gap:      {mean_interp - mean_extrap:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if t == "interpolation" else "salmon"
              for t in df_res["type"]]
    ax.bar(df_res["strategy"], df_res["r2"], color=colors)
    ax.set_ylim(max(0, df_res["r2"].min() - 0.1), 1.01)
    ax.set_ylabel("Test R² (log₁₀ CEP)")
    ax.set_title("Cross-validation: interpolation vs. extrapolation")
    ax.tick_params(axis="x", rotation=25)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue", label="Interpolation (random)"),
        Patch(color="salmon", label="Extrapolation (stratified)"),
    ])
    fig.tight_layout()
    fig.savefig(outdir / "crossval_bar.png", dpi=150)
    plt.close(fig)
    print(f"Cross-validation bar chart → {outdir / 'crossval_bar.png'}")


# ---------------------------------------------------------------------------
# Quantile regression subcommand (Task 7)
# ---------------------------------------------------------------------------


def cmd_quantile(args: argparse.Namespace) -> None:
    """Train 5th/50th/95th quantile XGBoost models; report empirical coverage."""
    import pickle as pkl

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = _load_data(args.csv)
    print(f"  {len(df)} rows")

    use_extended = all(c in df.columns for c in INTERACTION_FEATURES)
    features = ALL_FEATURES_EXTENDED if use_extended else ALL_FEATURES

    X = df[features].values
    y = np.log10(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    quantiles = [0.05, 0.50, 0.95]
    pipes: dict[float, object] = {}
    for q in quantiles:
        print(f"  Training quantile={q} model ...")
        pipe = build_xgb_quantile_pipeline(q)
        pipe.fit(X_train, y_train)
        pipes[q] = pipe
        model_pkl = outdir / f"model_q{int(q * 100):02d}.pkl"
        with open(model_pkl, "wb") as f:
            pkl.dump(pipe, f)
        print(f"    Saved → {model_pkl}")

    # Predictions on test set
    y_lo = pipes[0.05].predict(X_test)
    y_med = pipes[0.50].predict(X_test)
    y_hi = pipes[0.95].predict(X_test)

    # Empirical coverage at 90%
    in_band = (y_test >= y_lo) & (y_test <= y_hi)
    coverage = float(in_band.mean())
    median_width = float(np.median(y_hi - y_lo))
    print(f"\n  Empirical 90% coverage: {coverage:.3f}  (target: 0.90)")
    print(f"  Median band width:      {median_width:.3f} log10-m")
    print(f"  Median band width:     ×{10**median_width:.2f} factor")

    # Coverage by CEP magnitude bin
    bins = [0, 1, 2, 3, 4, 5]
    print("\n  Coverage by actual log₁₀(CEP) bin:")
    print("  {:>12s}  {:>10s}  {:>10s}  {:>12s}".format(
        "bin [log-m]", "n", "coverage", "med_width"
    ))
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() == 0:
            continue
        cov_bin = float(in_band[mask].mean())
        width_bin = float(np.median((y_hi - y_lo)[mask]))
        print(f"  [{lo:3.0f}, {hi:3.0f})    {mask.sum():6d}  {cov_bin:10.3f}  {width_bin:12.3f}")

    # Predicted vs actual (panel A)
    sort_idx = np.argsort(y_med)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test[sort_idx], y_med[sort_idx], s=4, alpha=0.3,
               color="steelblue", label="Median pred")
    ax.fill_between(
        y_test[sort_idx],
        y_lo[sort_idx], y_hi[sort_idx],
        alpha=0.3, color="steelblue", label="90% interval"
    )
    lims = [y_test.min() - 0.1, y_test.max() + 0.1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual log₁₀(CEP) [m]")
    ax.set_ylabel("Predicted log₁₀(CEP) [m]")
    ax.set_title(f"Quantile regression  (90% coverage = {coverage:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "quantile_prediction.png", dpi=150)
    plt.close(fig)
    print(f"\nQuantile prediction plot → {outdir / 'quantile_prediction.png'}")

    # Coverage by CEP magnitude (panel B)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, in_band.astype(float) + np.random.default_rng(0).uniform(-0.02, 0.02, len(y_test)),
               s=4, alpha=0.2, color="steelblue")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() == 0:
            continue
        ax.plot((lo + hi) / 2, float(in_band[mask].mean()), "rs", markersize=10)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
    ax.set_xlabel("Actual log₁₀(CEP) [m]")
    ax.set_ylabel("In-band (0 or 1)")
    ax.set_title("Coverage by CEP magnitude  (red squares = bin mean)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "quantile_coverage_by_bin.png", dpi=150)
    plt.close(fig)
    print(f"Quantile coverage plot → {outdir / 'quantile_coverage_by_bin.png'}")


# ---------------------------------------------------------------------------
# Scaling law subcommand (Task 6)
# ---------------------------------------------------------------------------


def cmd_scaling_law(args: argparse.Namespace) -> None:
    """Fit power-law CEP ~ C * sigma0^alpha / (g_rms * d_total)^gamma; compare to XGBoost."""
    from scipy.optimize import curve_fit

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = _load_data(args.csv)
    print(f"  {len(df)} rows")

    required = ["gradient_rms", "total_distance_m", "sigma_init_m", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for scaling law: {missing}")

    X_full = df[ALL_FEATURES].values
    y = np.log10(df[TARGET].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )

    # XGBoost baseline
    print("Training XGBoost baseline ...")
    pipe_xgb = build_xgb_pipeline()
    pipe_xgb.fit(X_train, y_train)
    r2_xgb = r2_score(y_test, pipe_xgb.predict(X_test))
    print(f"  XGBoost R² = {r2_xgb:.4f}")

    # Log-linear scaling law: log10(CEP) = log10(C) + alpha*log10(sigma0) - gamma*log10(g_rms*d_total)
    g_rms = df["gradient_rms"].values
    d_total = df["total_distance_m"].values
    sigma_init = df["sigma_init_m"].values
    cep = df[TARGET].values

    mask = (g_rms > 0) & (d_total > 0) & (sigma_init > 0) & (cep > 0)
    log_cep = np.log10(cep[mask])
    log_gd = np.log10(g_rms[mask] * d_total[mask])
    log_sigma = np.log10(sigma_init[mask])

    # Linear regression in log space: log10_cep = a + alpha*log_sigma + gamma*log_gd
    A = np.column_stack([np.ones(mask.sum()), log_sigma, log_gd])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, log_cep, rcond=None)
    log_C, alpha, gamma = coeffs

    y_pred_sl = A @ coeffs
    r2_sl = r2_score(log_cep, y_pred_sl)
    print(f"\nScaling law:  CEP ≈ {10**log_C:.4g} × σ₀^{alpha:.3f} × (g_rms · d)^{gamma:.3f}")
    print(f"  Scaling law R² (all data) = {r2_sl:.4f}")

    # Evaluate scaling law on test set
    test_mask = np.zeros(len(df), dtype=bool)
    test_indices = np.where(train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)[1])[0]
    test_mask_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)[1]
    df_test = df.iloc[test_mask_idx]
    mask_test = (df_test["gradient_rms"].values > 0) & (df_test["total_distance_m"].values > 0)
    log_gd_test = np.log10(df_test["gradient_rms"].values[mask_test] * df_test["total_distance_m"].values[mask_test])
    log_sigma_test = np.log10(df_test["sigma_init_m"].values[mask_test])
    A_test = np.column_stack([np.ones(mask_test.sum()), log_sigma_test, log_gd_test])
    y_pred_sl_test = A_test @ coeffs
    y_true_test = np.log10(df_test[TARGET].values[mask_test])
    r2_sl_test = r2_score(y_true_test, y_pred_sl_test)
    print(f"  Scaling law R² (test set) = {r2_sl_test:.4f}")
    print(f"  XGBoost    R² (test set) = {r2_xgb:.4f}")
    print(f"  XGBoost / scaling-law ratio: {r2_xgb / r2_sl_test:.3f}")

    # Log-log scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    x_plot = g_rms[mask] * d_total[mask]
    ax.scatter(x_plot, cep[mask], s=4, alpha=0.15, rasterized=True, label="Data")
    x_sorted = np.sort(x_plot)
    sigma_median = np.median(sigma_init[mask])
    y_fit = 10 ** (log_C + alpha * np.log10(sigma_median) + gamma * np.log10(x_sorted))
    ax.plot(x_sorted, y_fit, "r-", linewidth=2,
            label=f"Scaling law (σ₀={sigma_median:.0f}m, R²={r2_sl_test:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("g_rms × d_total  [f.u./m × m]")
    ax.set_ylabel("PCRB CEP [m]")
    ax.set_title("Empirical scaling law:  CEP vs. g_rms · d_total")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "scaling_law.png", dpi=150)
    plt.close(fig)
    print(f"\nScaling law plot → {outdir / 'scaling_law.png'}")

    import json
    with open(outdir / "scaling_law_coeffs.json", "w") as f:
        json.dump({"C": float(10 ** log_C), "alpha": float(alpha), "gamma": float(gamma),
                   "r2_test": float(r2_sl_test), "r2_xgb_test": float(r2_xgb)}, f, indent=2)
    print(f"Scaling law coefficients → {outdir / 'scaling_law_coeffs.json'}")


# ---------------------------------------------------------------------------
# Predict helpers
# ---------------------------------------------------------------------------

# All named presets mirrored from fetch_terrain — (lat, lon, human label)
PRESETS: dict[str, tuple[float, float, str]] = {
    "big-sur":      ( 36.5, -121.5, "Big Sur, CA"),
    "alps":         ( 47.0,   10.5, "Alps, Austria/Italy"),
    "grand-canyon": ( 36.1, -112.1, "Grand Canyon, AZ"),
    "everest":      ( 27.9,   86.9, "Everest region, Nepal"),
    "iceland":      ( 64.0,  -16.0, "Vatnajökull glacier, Iceland"),
    "amazon":       ( -3.5,  -62.0, "Amazon basin, Brazil"),
    "sahara":       ( 23.0,   10.0, "Sahara, Libya"),
}


def _load_terrain_file(path: str | Path) -> np.ndarray:
    """Load terrain from .npy and return a normalised float array."""
    p = Path(path)
    if p.suffix != ".npy":
        raise ValueError(f"Unsupported terrain format: {p.suffix}  (expect .npy)")
    arr = np.load(p).astype(float)
    std = arr.std()
    arr = (arr - arr.mean()) / std if std > 0 else arr - arr.mean()
    return arr


def _load_model(model_path: str | Path):
    import json
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found: {p}\n"
            "Run `analyze-results analyze` first to train the model."
        )
    with open(p, "rb") as f:
        pipe = pickle.load(f)
    # Load saved feature list from companion metadata file if present
    meta_path = p.with_name(p.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path) as fj:
            meta = json.load(fj)
        features = meta.get("features", None)
        pipe._saved_features = features  # stash for use by callers
    else:
        pipe._saved_features = None
    return pipe


def _predict_and_plot(
    *,
    terrain: np.ndarray,
    dx: float,
    sigma_meas: float,
    sigma_init: float,
    pipe,
    outdir: Path,
    terrain_name: str,
    terrain_label: str,
    speeds: np.ndarray,
    durations: np.ndarray,
    plot_slices: bool,
) -> np.ndarray:
    """
    Core prediction + plotting logic.  Returns the (n_speeds, n_durations)
    CEP prediction array.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Map feature extraction
    print(f"\n[{terrain_label}]  shape={terrain.shape}  dx={dx:.1f} m  σ_meas={sigma_meas}")
    map_metrics = map_information_content(terrain, dx=dx, sigma_meas=sigma_meas)

    # Use the feature list saved with the model (handles models trained without
    # spatial_diversity_index or other features not present in training data).
    saved_features: list[str] | None = getattr(pipe, "_saved_features", None)
    if saved_features is not None:
        map_feat_names = [f for f in saved_features if f in MAP_FEATURES]
        traj_feat_names = [f for f in saved_features if f in TRAJ_FEATURES]
    else:
        map_feat_names = MAP_FEATURES
        traj_feat_names = TRAJ_FEATURES

    for k in map_feat_names:
        val = map_metrics.get(k, float("nan"))
        print(f"  {k}: {val:.4g}")
    map_feat_values = [map_metrics.get(k, float("nan")) for k in map_feat_names]

    # Build speed × duration prediction grid
    spd_grid, dur_grid = np.meshgrid(speeds, durations, indexing="ij")
    dist_grid = spd_grid * dur_grid
    n_speeds, n_durations = spd_grid.shape
    predictions = np.empty((n_speeds, n_durations))

    for i in range(n_speeds):
        for j in range(n_durations):
            traj_vals = [
                float(spd_grid[i, j]),      # speed_m_s
                float(dist_grid[i, j]),     # total_distance_m
                float(sigma_init),          # sigma_init_m
            ]
            # Only include trajectory features that the model was trained on
            traj_row = [traj_vals[["speed_m_s", "total_distance_m", "sigma_init_m"].index(f)]
                        for f in traj_feat_names]
            row = map_feat_values + traj_row
            predictions[i, j] = 10 ** pipe.predict(np.array([row]))[0]

    # ── Heatmap ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(durations, speeds, predictions, cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=ax, label="Predicted CEP (m)")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(
        f"Predicted fix CEP — {terrain_label}\n"
        f"σ₀={sigma_init:.0f} m  dx={dx:.0f} m  σ_meas={sigma_meas}"
    )
    # Cell annotations
    threshold = predictions.max() * 0.6
    for i, spd in enumerate(speeds):
        for j, dur in enumerate(durations):
            val = predictions[i, j]
            txt = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
            ax.text(dur, spd, txt, ha="center", va="center",
                    fontsize=8, color="white" if val < threshold else "black")
    fig.tight_layout()
    heatmap_png = outdir / f"{terrain_name}_sigma{int(sigma_init)}.png"
    fig.savefig(heatmap_png, dpi=150)
    plt.close(fig)
    print(f"  Heatmap → {heatmap_png}")

    # ── Slice plots (combined 2-panel) ────────────────────────────────────
    if plot_slices:
        tag = f"{terrain_name}_sigma{int(sigma_init)}"
        fig, (ax_dur, ax_spd) = plt.subplots(1, 2, figsize=(13, 5))

        for i, spd in enumerate(speeds):
            ax_dur.plot(durations, predictions[i], marker="o", label=f"{spd:.0f} m/s")
        ax_dur.set_xlabel("Duration (s)")
        ax_dur.set_ylabel("Predicted CEP (m)")
        ax_dur.set_yscale("log")
        ax_dur.set_title(f"CEP vs duration — {terrain_label}")
        ax_dur.legend(title="Speed", fontsize=8)
        ax_dur.grid(True, alpha=0.3)

        for j, dur in enumerate(durations):
            ax_spd.plot(speeds, predictions[:, j], marker="s", label=f"{dur:.0f} s")
        ax_spd.set_xlabel("Speed (m/s)")
        ax_spd.set_ylabel("Predicted CEP (m)")
        ax_spd.set_yscale("log")
        ax_spd.set_title(f"CEP vs speed — {terrain_label}")
        ax_spd.legend(title="Duration", fontsize=8)
        ax_spd.grid(True, alpha=0.3)

        fig.suptitle(f"σ₀={sigma_init:.0f} m  Δx={dx:.0f} m", fontsize=10)
        fig.tight_layout()
        slices_png = outdir / f"{tag}_slices.png"
        fig.savefig(slices_png, dpi=150)
        plt.close(fig)
        print(f"  Slices → {slices_png}")

    # ── Summary table ─────────────────────────────────────────────────────
    dur_hdr = "  ".join(f"{d:>8.0f}s" for d in durations)
    print(f"\n  Predicted CEP (m):")
    print(f"  {'speed':>7}  {dur_hdr}")
    print("  " + "-" * (9 + 10 * len(durations)))
    for i, spd in enumerate(speeds):
        row_str = "  ".join(f"{predictions[i, j]:>9.1f}" for j in range(n_durations))
        print(f"  {spd:>6.1f}  {row_str}")

    return predictions


# ---------------------------------------------------------------------------
# Predict subcommand
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    pipe = _load_model(args.model)
    print(f"Loaded model from {args.model}")

    speeds = np.array(args.speeds if args.speeds else DEFAULT_SPEEDS)
    durations = np.array(args.durations if args.durations else DEFAULT_DURATIONS)

    if args.preset:
        # Fetch directly from SRTM — dx is auto-computed at the tile's latitude
        from geo_fixability.terrain import load_terrain_patch
        lat, lon, label = PRESETS[args.preset]
        terrain, dx = load_terrain_patch(lat, lon, size_km=args.size_km)
        terrain_name = args.preset
    else:
        terrain = _load_terrain_file(args.terrain)
        dx = args.dx
        terrain_name = Path(args.terrain).stem
        label = terrain_name

    _predict_and_plot(
        terrain=terrain,
        dx=dx,
        sigma_meas=args.sigma_meas,
        sigma_init=args.sigma_init,
        pipe=pipe,
        outdir=outdir,
        terrain_name=terrain_name,
        terrain_label=label,
        speeds=speeds,
        durations=durations,
        plot_slices=args.plot_slices,
    )


# ---------------------------------------------------------------------------
# Predict-presets subcommand
# ---------------------------------------------------------------------------


def cmd_predict_presets(args: argparse.Namespace) -> None:
    """Download all (or selected) SRTM presets and generate prediction plots."""
    from geo_fixability.terrain import load_terrain_patch

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pipe = _load_model(args.model)
    print(f"Loaded model from {args.model}")

    speeds = np.array(args.speeds if args.speeds else DEFAULT_SPEEDS)
    durations = np.array(args.durations if args.durations else DEFAULT_DURATIONS)

    names = args.presets if args.presets else list(PRESETS.keys())
    failures: list[str] = []

    for name in names:
        if name not in PRESETS:
            print(f"  Unknown preset '{name}' — skipping.")
            continue
        lat, lon, label = PRESETS[name]
        print(f"\n{'='*60}")
        print(f"  Preset: {name}  ({label})")
        print(f"{'='*60}")
        try:
            terrain, dx = load_terrain_patch(lat, lon, size_km=args.size_km)
            _predict_and_plot(
                terrain=terrain,
                dx=dx,
                sigma_meas=args.sigma_meas,
                sigma_init=args.sigma_init,
                pipe=pipe,
                outdir=outdir,
                terrain_name=name,
                terrain_label=label,
                speeds=speeds,
                durations=durations,
                plot_slices=args.plot_slices,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR for {name}: {exc}")
            failures.append(name)

    print(f"\n{'='*60}")
    print(f"Done.  {len(names) - len(failures)}/{len(names)} presets succeeded.")
    if failures:
        print(f"Failed: {failures}")
    print(f"Outputs in: {outdir}")


# ---------------------------------------------------------------------------
# Spatial fixability map subcommand
# ---------------------------------------------------------------------------


def _hillshade(terrain: np.ndarray) -> np.ndarray:
    """Hillshade from normalised terrain for use as a plot background."""
    gy, gx = np.gradient(terrain)
    # Light from upper-left (NW) at ~63° altitude
    lx, ly, lz = -1.0, -1.0, 2.0
    norm_light = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    shade = (-gx * lx - gy * ly + lz) / (np.sqrt(gx ** 2 + gy ** 2 + 1.0) * norm_light)
    return np.clip(shade, 0.0, 1.0)


def _analytical_maps(
    terrain: np.ndarray, dx: float, sigma_meas: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel gradient magnitude [f.u./m] and single-measurement CRB [m]."""
    gy, gx = np.gradient(terrain)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) / dx
    with np.errstate(divide="ignore", invalid="ignore"):
        loc_bound = np.where(grad_mag > 0.0, sigma_meas / grad_mag, np.nan)
    return grad_mag, loc_bound


def _fixability_raster(
    terrain: np.ndarray,
    dx: float,
    sigma_meas: float,
    sigma_init: float,
    pipe,
    speed: float,
    duration: float,
    n_arms: int = 8,
    stride_px: int | None = None,
    min_window_px: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window ML prediction of fix CEP across the terrain.

    For each point on a regular coarse grid a local patch is extracted whose
    size matches the area swept by a star trajectory of the given speed and
    duration (arm length D = speed * duration / (2 * n_arms)).  Map metrics
    are computed on the patch and fed to the trained model.

    Parameters
    ----------
    terrain : ndarray (ny, nx)
        Normalised terrain (zero-mean, unit-variance).
    dx : float
        Pixel size [m].
    sigma_meas, sigma_init : float
        Measurement noise and initial position uncertainty [m].
    pipe : sklearn Pipeline
        Trained model.
    speed : float
        Vehicle speed [m/s].
    duration : float
        Trajectory duration [s].
    n_arms : int
        Star arm count (default 8).
    stride_px : int or None
        Grid stride in pixels.  Default: window_px // 4.
    min_window_px : int
        Minimum patch side [px] for reliable spectral analysis.

    Returns
    -------
    cep : ndarray (n_y, n_x)   — predicted CEP [m]
    row_px : ndarray (n_y,)    — pixel row index of each prediction
    col_px : ndarray (n_x,)    — pixel col index of each prediction
    """
    ny, nx = terrain.shape
    D = speed * duration / (2.0 * n_arms)
    window_px = max(min_window_px, int(round(2.0 * D / dx)))
    half_w = window_px // 2
    if stride_px is None:
        stride_px = max(4, window_px // 4)

    rows = np.arange(half_w, ny - half_w, stride_px)
    cols = np.arange(half_w, nx - half_w, stride_px)
    total_dist = speed * duration
    cep = np.full((len(rows), len(cols)), np.nan)

    n_total = len(rows) * len(cols)
    report_at = set(range(0, n_total, max(1, n_total // 10)))
    done = 0

    print(f"  patch={window_px}px ({window_px * dx / 1000:.1f} km)  "
          f"stride={stride_px}px ({stride_px * dx:.0f} m)  "
          f"grid={len(rows)}×{len(cols)} ({n_total} pts)")

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            patch = terrain[r - half_w:r + half_w, c - half_w:c + half_w].copy()
            std = patch.std()
            if std < 1e-10:
                done += 1
                continue
            patch = (patch - patch.mean()) / std
            m = map_information_content(patch, dx=dx, sigma_meas=sigma_meas)
            saved = getattr(pipe, "_saved_features", None)
            if saved is not None:
                map_f = [f for f in saved if f in MAP_FEATURES]
                traj_f = [f for f in saved if f in TRAJ_FEATURES]
                traj_lookup = {"speed_m_s": speed,
                               "total_distance_m": total_dist,
                               "sigma_init_m": sigma_init}
                feat = [m.get(k, float("nan")) for k in map_f] + \
                       [traj_lookup[k] for k in traj_f]
            else:
                feat = [m[k] for k in MAP_FEATURES] + [speed, total_dist, sigma_init]
            cep[i, j] = 10.0 ** pipe.predict(np.array([feat]))[0]
            done += 1
            if done in report_at:
                print(f"    {done}/{n_total} ({100 * done // n_total}%)",
                      end="\r", flush=True)

    print(f"    {n_total}/{n_total} (100%)   ")
    return cep, rows, cols


def cmd_map(args: argparse.Namespace) -> None:
    """Generate spatial fixability maps overlaid on terrain hillshade."""
    from geo_fixability.terrain import load_terrain_patch

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pipe = _load_model(args.model)
    print(f"Loaded model from {args.model}")

    # Load terrain ─────────────────────────────────────────────────────────
    if args.preset:
        lat, lon, label = PRESETS[args.preset]
        terrain, dx = load_terrain_patch(lat, lon, size_km=args.size_km)
        terrain_name = args.preset
    else:
        terrain = _load_terrain_file(args.terrain)
        dx = args.dx
        terrain_name = Path(args.terrain).stem
        label = terrain_name

    ny, nx = terrain.shape
    x_max_km = nx * dx / 1000.0
    y_max_km = ny * dx / 1000.0
    print(f"Terrain: {ny}×{nx} px  dx={dx:.1f} m  "
          f"{x_max_km:.1f}×{y_max_km:.1f} km  [{label}]")

    speeds = np.array(args.speeds if args.speeds else DEFAULT_SPEEDS)
    duration = args.duration

    # Shared axis helpers ──────────────────────────────────────────────────
    # Work in pixel coordinates throughout; label axes in km.
    def _km_ticks(n_px: int, n_ticks: int = 5) -> tuple[np.ndarray, list[str]]:
        px = np.linspace(0, n_px - 1, n_ticks)
        km = px * dx / 1000.0
        return px, [f"{v:.0f}" for v in km]

    xtick_px, xtick_lbl = _km_ticks(nx)
    ytick_px, ytick_lbl = _km_ticks(ny)

    shade = _hillshade(terrain)
    grad_mag, loc_bound = _analytical_maps(terrain, dx=dx, sigma_meas=args.sigma_meas)

    # ── Combined terrain diagnostics (3-panel: hillshade + gradient + CRB) ─
    def _decorate(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=9)
        ax.set_xticks(xtick_px, xtick_lbl)
        ax.set_yticks(ytick_px, ytick_lbl)
        ax.set_xlabel("East (km)", fontsize=8)
        ax.set_ylabel("South (km)", fontsize=8)
        ax.tick_params(labelsize=7)

    gm_log = np.log10(np.clip(grad_mag, 1e-6, None))
    lb_log = np.log10(np.clip(loc_bound, 0.1, 1e5))
    vmin_lb, vmax_lb = float(np.nanmin(lb_log)), float(np.nanmax(lb_log))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    axes[0].imshow(shade, cmap="gray", aspect="equal")
    _decorate(axes[0], f"Hillshade — {label}  (dx={dx:.0f} m)")

    axes[1].imshow(shade, cmap="gray", aspect="equal", alpha=0.45)
    im2 = axes[1].imshow(gm_log, cmap="plasma", aspect="equal", alpha=0.75)
    cb2 = plt.colorbar(im2, ax=axes[1], shrink=0.85, pad=0.02)
    cb2.set_label("log₁₀ |∇h| (f.u./m)", fontsize=8)
    _decorate(axes[1], f"Gradient magnitude — {label}")

    axes[2].imshow(shade, cmap="gray", aspect="equal", alpha=0.45)
    im3 = axes[2].imshow(lb_log, cmap="RdYlGn_r", aspect="equal", alpha=0.75,
                         vmin=vmin_lb, vmax=vmax_lb)
    cb3 = plt.colorbar(im3, ax=axes[2], shrink=0.85, pad=0.02)
    tick_m = [t for t in [1, 5, 10, 50, 100, 500, 1000, 5000]
              if vmin_lb <= np.log10(t) <= vmax_lb]
    if tick_m:
        cb3.set_ticks(np.log10(tick_m))
        cb3.set_ticklabels([f"{t} m" for t in tick_m])
    cb3.set_label("Single-meas. CRB position error (m)", fontsize=8)
    _decorate(axes[2], f"Localization bound — {label}")

    fig.tight_layout()
    diag_png = outdir / f"{terrain_name}_diagnostics.png"
    fig.savefig(diag_png, dpi=150)
    plt.close(fig)
    print(f"Saved {diag_png}")

    # ── Per-speed CEP prediction maps (multi-panel, one panel per speed) ─
    CEP_CONTOURS = [1, 2, 5, 10, 25, 50, 100, 250, 500]

    n_speeds = len(speeds)
    all_cep: list[np.ndarray] = []
    all_row_px: list[np.ndarray] = []
    all_col_px: list[np.ndarray] = []

    for speed in speeds:
        print(f"\nv = {speed:.0f} m/s   T = {duration:.0f} s ...")
        cep, row_px_s, col_px_s = _fixability_raster(
            terrain=terrain, dx=dx,
            sigma_meas=args.sigma_meas, sigma_init=args.sigma_init,
            pipe=pipe, speed=speed, duration=duration,
            stride_px=args.stride_px, min_window_px=args.min_window_px,
        )
        all_cep.append(cep)
        all_row_px.append(row_px_s)
        all_col_px.append(col_px_s)

    # Shared colour scale across all speeds
    all_valid = np.concatenate([
        np.log10(np.clip(c[np.isfinite(c)], 0.1, None)) for c in all_cep if np.any(np.isfinite(c))
    ])
    vmin_cep = float(all_valid.min()) if all_valid.size else 0.0
    vmax_cep = float(all_valid.max()) if all_valid.size else 3.0

    ncols = min(4, n_speeds)
    nrows = (n_speeds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=150,
                             squeeze=False)
    for k, (speed, cep, row_px_s, col_px_s) in enumerate(
            zip(speeds, all_cep, all_row_px, all_col_px)):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(shade, cmap="gray", aspect="equal")
        log_cep = np.log10(np.where(np.isfinite(cep), np.clip(cep, 0.1, None), np.nan))
        if np.any(np.isfinite(log_cep)):
            pcm = ax.pcolormesh(
                col_px_s, row_px_s, log_cep,
                cmap="RdYlGn_r", alpha=0.72, shading="nearest",
                vmin=vmin_cep, vmax=vmax_cep,
            )
            shown_ticks = [t for t in CEP_CONTOURS
                           if vmin_cep < np.log10(t) < vmax_cep]
            if shown_ticks:
                ax.contour(col_px_s, row_px_s, log_cep,
                           levels=np.log10(shown_ticks),
                           colors="white", linewidths=0.65, alpha=0.8)
        ax.set_xticks(xtick_px, xtick_lbl, fontsize=7)
        ax.set_yticks(ytick_px, ytick_lbl, fontsize=7)
        ax.set_xlabel("East (km)", fontsize=8)
        ax.set_ylabel("South (km)", fontsize=8)
        D_arm = speed * duration / 16.0
        ax.set_title(
            f"v={speed:.0f} m/s  D={D_arm:.0f} m",
            fontsize=9,
        )
    # Hide unused axes
    for k in range(n_speeds, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    # Shared colourbar
    if all_valid.size > 0:
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                   norm=plt.Normalize(vmin=vmin_cep, vmax=vmax_cep))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        shown_ticks = [t for t in CEP_CONTOURS if vmin_cep < np.log10(t) < vmax_cep]
        if shown_ticks:
            cb.set_ticks(np.log10(shown_ticks))
            cb.set_ticklabels([f"{t} m" for t in shown_ticks])
        cb.set_label("Predicted CEP (m)")
    fig.suptitle(
        f"Predicted fix CEP — {label}  T={duration:.0f} s  σ₀={args.sigma_init:.0f} m",
        fontsize=10,
    )
    fig.tight_layout()
    maps_png = (outdir /
                f"{terrain_name}_fixability_T{int(duration)}_sigma{int(args.sigma_init)}.png")
    fig.savefig(maps_png, dpi=150)
    plt.close(fig)
    print(f"Saved {maps_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and predict fix accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- analyze subcommand ---
    p_analyze = sub.add_parser(
        "analyze",
        help="Train model from experiment CSV and generate EDA plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_analyze.add_argument(
        "--csv", default="data/experiments/star_trajectory_results.csv",
        help="Path to experiment results CSV.",
    )
    p_analyze.add_argument(
        "--outdir", default="data/analysis",
        help="Directory for output plots and model.",
    )
    p_analyze.add_argument(
        "--model-type", choices=["gbr", "xgb"], default="xgb", dest="model_type",
        help="Regression model type: 'gbr' (GradientBoosting) or 'xgb' (XGBoost). "
             "XGBoost also generates a SHAP summary plot.",
    )

    # --- predict subcommand ---
    p_predict = sub.add_parser(
        "predict",
        help="Predict fix accuracy for a terrain file or named preset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    terrain_src = p_predict.add_mutually_exclusive_group(required=True)
    terrain_src.add_argument(
        "terrain", nargs="?", default=None,
        help="Path to terrain .npy file.",
    )
    terrain_src.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None,
        help="Named SRTM location preset (downloads automatically).",
    )
    p_predict.add_argument(
        "--dx", type=float, default=30.0,
        help="Pixel size [m/px]. Ignored when --preset is used (auto-computed).",
    )
    p_predict.add_argument(
        "--size-km", type=float, default=50.0, dest="size_km",
        help="Patch side length [km] for --preset downloads.",
    )
    p_predict.add_argument(
        "--sigma-meas", type=float, default=0.1, dest="sigma_meas",
        help="Measurement noise std [field units].",
    )
    p_predict.add_argument(
        "--sigma-init", type=float, default=500.0, dest="sigma_init",
        help="Initial position uncertainty 1-sigma [m].",
    )
    p_predict.add_argument(
        "--model", default="data/theoretical/analysis/model.pkl",
        help="Path to trained model pickle.",
    )
    p_predict.add_argument(
        "--speeds", nargs="+", type=float, default=None,
        help="Speed values to sweep [m/s]. Default: 2 5 10 20",
    )
    p_predict.add_argument(
        "--durations", nargs="+", type=float, default=None,
        help="Duration values to sweep [s]. Default: 300 600 1200 2400",
    )
    p_predict.add_argument(
        "--outdir", default="data/predictions",
        help="Directory for output plots.",
    )
    p_predict.add_argument(
        "--plot-slices", action="store_true", dest="plot_slices",
        help="Also generate 1D slice plots (CEP vs speed, CEP vs duration).",
    )

    # --- predict-presets subcommand ---
    p_presets = sub.add_parser(
        "predict-presets",
        help="Run predictions for all (or selected) named SRTM presets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_presets.add_argument(
        "--presets", nargs="+", choices=list(PRESETS.keys()), default=None,
        help="Subset of presets to run. Default: all.",
    )
    p_presets.add_argument(
        "--size-km", type=float, default=50.0, dest="size_km",
        help="Patch side length [km] for downloads.",
    )
    p_presets.add_argument(
        "--sigma-meas", type=float, default=0.1, dest="sigma_meas",
        help="Measurement noise std [field units].",
    )
    p_presets.add_argument(
        "--sigma-init", type=float, default=500.0, dest="sigma_init",
        help="Initial position uncertainty 1-sigma [m].",
    )
    p_presets.add_argument(
        "--model", default="data/theoretical/analysis/model.pkl",
        help="Path to trained model pickle.",
    )
    p_presets.add_argument(
        "--speeds", nargs="+", type=float, default=None,
        help="Speed values to sweep [m/s]. Default: 2 5 10 20",
    )
    p_presets.add_argument(
        "--durations", nargs="+", type=float, default=None,
        help="Duration values to sweep [s]. Default: 300 600 1200 2400",
    )
    p_presets.add_argument(
        "--outdir", default="data/predictions",
        help="Directory for output plots.",
    )
    p_presets.add_argument(
        "--plot-slices", action="store_true", dest="plot_slices",
        help="Also generate 1D slice plots.",
    )

    # --- map subcommand ---
    p_map = sub.add_parser(
        "map",
        help="Generate spatial fixability maps overlaid on terrain hillshade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    terrain_src3 = p_map.add_mutually_exclusive_group(required=True)
    terrain_src3.add_argument(
        "terrain", nargs="?", default=None,
        help="Path to terrain .npy file.",
    )
    terrain_src3.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None,
        help="Named SRTM preset (downloads automatically).",
    )
    p_map.add_argument(
        "--dx", type=float, default=30.0,
        help="Pixel size [m/px].  Ignored when --preset is used.",
    )
    p_map.add_argument(
        "--size-km", type=float, default=50.0, dest="size_km",
        help="Patch side length [km] for --preset downloads.",
    )
    p_map.add_argument(
        "--sigma-meas", type=float, default=0.1, dest="sigma_meas",
        help="Measurement noise std [field units].",
    )
    p_map.add_argument(
        "--sigma-init", type=float, default=500.0, dest="sigma_init",
        help="Initial position uncertainty 1-sigma [m].",
    )
    p_map.add_argument(
        "--model", default="data/theoretical/analysis/model.pkl",
        help="Path to trained model pickle.",
    )
    p_map.add_argument(
        "--speeds", nargs="+", type=float, default=None,
        help="Speeds to map [m/s].  Default: 2 5 10 20",
    )
    p_map.add_argument(
        "--duration", type=float, default=600.0,
        help="Trajectory duration [s] used for the prediction.",
    )
    p_map.add_argument(
        "--stride-px", type=int, default=None, dest="stride_px",
        help="Prediction grid stride [px].  Default: patch_px // 4  (~25%% overlap).",
    )
    p_map.add_argument(
        "--min-window-px", type=int, default=64, dest="min_window_px",
        help="Minimum patch side [px] for spectral metric reliability.",
    )
    p_map.add_argument(
        "--outdir", default="data/predictions",
        help="Output directory for plots.",
    )

    # --- ablation subcommand ---
    p_ablation = sub.add_parser(
        "ablation",
        help="Feature ablation study — train XGBoost on feature subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_ablation.add_argument(
        "--csv", default="data/experiments/star_trajectory_results.csv",
        help="Path to experiment results CSV.",
    )
    p_ablation.add_argument(
        "--outdir", default="data/theoretical/analysis",
        help="Directory for output CSV and figure.",
    )

    # --- crossval subcommand ---
    p_crossval = sub.add_parser(
        "crossval",
        help="Cross-validation bias analysis — random vs. stratified holdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_crossval.add_argument(
        "--csv", default="data/experiments/star_trajectory_results.csv",
        help="Path to experiment results CSV.",
    )
    p_crossval.add_argument(
        "--outdir", default="data/theoretical/analysis",
        help="Directory for output CSV and figure.",
    )

    # --- quantile subcommand ---
    p_quantile = sub.add_parser(
        "quantile",
        help="Train 5th/50th/95th quantile XGBoost models for uncertainty bounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_quantile.add_argument(
        "--csv", default="data/experiments/star_trajectory_results.csv",
        help="Path to experiment results CSV.",
    )
    p_quantile.add_argument(
        "--outdir", default="data/theoretical/analysis",
        help="Directory for output models and figures.",
    )

    # --- scaling-law subcommand ---
    p_sl = sub.add_parser(
        "scaling-law",
        help="Fit empirical power-law CEP ~ C * sigma0^a / (g_rms*d)^g.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_sl.add_argument(
        "--csv", default="data/experiments/star_trajectory_results.csv",
        help="Path to experiment results CSV.",
    )
    p_sl.add_argument(
        "--outdir", default="data/theoretical/analysis",
        help="Directory for output figures and coefficient JSON.",
    )

    args = parser.parse_args()
    if args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "predict-presets":
        cmd_predict_presets(args)
    elif args.cmd == "map":
        cmd_map(args)
    elif args.cmd == "ablation":
        cmd_ablation(args)
    elif args.cmd == "crossval":
        cmd_crossval(args)
    elif args.cmd == "quantile":
        cmd_quantile(args)
    elif args.cmd == "scaling-law":
        cmd_scaling_law(args)


if __name__ == "__main__":
    main()
