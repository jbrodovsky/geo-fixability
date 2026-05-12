"""
Validate the PCRB prediction model on real SRTM terrain patches.

For each named preset location, downloads the terrain (or uses cached tiles),
extracts random sub-patches, computes the true PCRB CEP, and compares it to
the model's predicted CEP.  Results are saved as a CSV and a scatter plot with
R² annotated — filling the real-terrain validation gap in the paper (§V-B).

Usage
-----
    uv run validate-terrain
    uv run validate-terrain --presets big-sur alps --n-seeds 5
    uv run validate-terrain --model data/theoretical/analysis/model.pkl \\
                             --outdir data/validation --n-seeds 20
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler

from geo_fixability.navigation import pcrb_trajectory, navigation_improvement
from geo_fixability.trajectory import star_trajectory
from geo_fixability.terrain import load_terrain_patch
from geo_fixability.metrics import map_information_content
from geo_fixability.features import patch_heterogeneity


# ---------------------------------------------------------------------------
# Named terrain presets (mirrors fetch_terrain and analyze_results)
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[float, float, str]] = {
    "big-sur":      ( 36.5, -121.5, "Big Sur, CA"),
    "alps":         ( 47.0,   10.5, "Alps, Austria/Italy"),
    "grand-canyon": ( 36.1, -112.1, "Grand Canyon, AZ"),
    "everest":      ( 27.9,   86.9, "Everest region, Nepal"),
    "iceland":      ( 64.0,  -16.0, "Vatnajökull glacier, Iceland"),
    "amazon":       ( -3.5,  -62.0, "Amazon basin, Brazil"),
    "sahara":       ( 23.0,   10.0, "Sahara, Libya"),
}

# Feature columns the model expects (must match model_meta.json)
_FALLBACK_FEATURES = [
    "gradient_rms", "information_index", "spatial_diversity_index",
    "localization_index", "localization_bound_m", "correlation_length_m",
    "speed_m_s", "total_distance_m", "sigma_init_m",
]


def _load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _load_feature_names(model_path: Path) -> list[str]:
    meta_path = model_path.with_name("model_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)["features"]
    return _FALLBACK_FEATURES


def run_pcrb_on_patch(
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
    speed: float,
    duration: float,
    n_arms: int,
    sigma_init: float,
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Compute PCRB CEP for one (field, trajectory) scenario.

    Parameters
    ----------
    field : ndarray (ny, nx)
        Normalised terrain patch (zero-mean, unit-variance).
    dx : float
        Pixel size [m/pixel].
    sigma_meas : float
        Measurement noise std [field units].
    speed : float
        Vehicle speed [m/s].
    duration : float
        Trajectory duration [s].
    n_arms : int
        Number of star arms.
    sigma_init : float
        Initial position uncertainty 1-sigma [m].
    dt : float
        Time step [s]. Default 1.0.

    Returns
    -------
    dict with map metric keys plus:
        ``speed_m_s``, ``total_distance_m``, ``sigma_init_m``,
        ``pcrb_cep_actual_m`` (true PCRB CEP).
    """
    D = speed * duration / (2.0 * n_arms)
    positions, times = star_trajectory(D=D, speed=speed, dt=dt, n_arms=n_arms)

    P_init = np.eye(2) * sigma_init ** 2
    Q = np.eye(2) * 1.0   # minimal process noise for bound computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_traj = pcrb_trajectory(positions, field, dx, sigma_meas, P_init, Q)

    metrics = navigation_improvement(P_traj)
    map_info = map_information_content(field, dx=dx, sigma_meas=sigma_meas)

    return {
        **map_info,
        "speed_m_s": speed,
        "total_distance_m": speed * duration,
        "sigma_init_m": sigma_init,
        "pcrb_cep_actual_m": metrics["cep_final_m"],
    }


def validate_preset(
    preset_name: str,
    lat: float,
    lon: float,
    size_km: float,
    speeds: list[float],
    durations: list[float],
    sigma_init: float,
    sigma_meas: float,
    model,
    feature_names: list[str],
    n_seeds: int,
    n_arms: int,
    dt: float,
    verbose: bool,
) -> pd.DataFrame:
    """
    Run PCRB validation for one named terrain preset.

    For each (speed, duration, seed) combination a sub-patch is extracted
    with a random ±10% offset from the tile centre, the true PCRB CEP is
    computed, and the model prediction is recorded.

    Returns a DataFrame with columns including ``pcrb_cep_actual_m`` and
    ``pcrb_cep_predicted_m``.
    """
    if verbose:
        print(f"\n  Downloading terrain for '{preset_name}' ({lat:.1f}, {lon:.1f}) ...")
    try:
        field_full, dx = load_terrain_patch(lat, lon, size_km=size_km, verbose=verbose)
    except Exception as exc:
        print(f"  ERROR loading terrain for {preset_name}: {exc}")
        return pd.DataFrame()

    ny, nx = field_full.shape
    rng = np.random.default_rng(0)
    records: list[dict] = []

    for speed in speeds:
        for duration in durations:
            for seed in range(n_seeds):
                # Random ±10% sub-patch offset
                rng_seed = np.random.default_rng(seed * 1000 + int(speed) * 100 + int(duration))
                dy = int(rng_seed.integers(-ny // 10, ny // 10 + 1))
                dx_offset = int(rng_seed.integers(-nx // 10, nx // 10 + 1))
                cy = np.clip(ny // 2 + dy, ny // 4, 3 * ny // 4)
                cx = np.clip(nx // 2 + dx_offset, nx // 4, 3 * nx // 4)

                half = min(cy, nx // 2 - abs(cx - nx // 2), ny // 2 - abs(cy - ny // 2))
                half = max(half, ny // 4)
                patch = field_full[cy - half:cy + half, cx - half:cx + half].copy()
                std = patch.std()
                if std < 1e-9:
                    continue
                patch = (patch - patch.mean()) / std

                try:
                    row = run_pcrb_on_patch(
                        patch, dx, sigma_meas, speed, duration,
                        n_arms, sigma_init, dt,
                    )
                except Exception as exc:
                    if verbose:
                        print(f"    skip ({speed},{duration},{seed}): {exc}")
                    continue

                # Model prediction — use only features the model knows
                feat_vec = [row.get(f, np.nan) for f in feature_names]
                if any(np.isnan(v) for v in feat_vec):
                    continue
                log_cep_pred = float(model.predict(np.array([feat_vec]))[0])
                cep_pred = 10.0 ** log_cep_pred

                records.append({
                    "preset": preset_name,
                    "lat": lat,
                    "lon": lon,
                    "speed_m_s": speed,
                    "duration_s": duration,
                    "seed": seed,
                    "pcrb_cep_actual_m": row["pcrb_cep_actual_m"],
                    "pcrb_cep_predicted_m": cep_pred,
                })

    return pd.DataFrame(records)


def _plot_validation(df: pd.DataFrame, outdir: Path) -> None:
    """Scatter plot of predicted vs actual PCRB CEP with R² annotated."""
    if df.empty:
        print("  No data to plot.")
        return

    actual = df["pcrb_cep_actual_m"].values
    predicted = df["pcrb_cep_predicted_m"].values

    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0) & (predicted > 0)
    if mask.sum() < 2:
        print("  Not enough valid points to compute R².")
        return

    log_actual = np.log10(actual[mask])
    log_pred = np.log10(predicted[mask])
    r2 = r2_score(log_actual, log_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    presets = df["preset"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(presets)))
    for color, preset in zip(colors, presets):
        sel = df["preset"] == preset
        ax.scatter(
            actual[sel.values & mask],
            predicted[sel.values & mask],
            s=15, alpha=0.6, label=preset, color=color,
        )

    lims_lo = min(actual[mask].min(), predicted[mask].min()) * 0.8
    lims_hi = max(actual[mask].max(), predicted[mask].max()) * 1.2
    ax.plot([lims_lo, lims_hi], [lims_lo, lims_hi], "k--", linewidth=1.2, label="Perfect fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual PCRB CEP [m]")
    ax.set_ylabel("Predicted PCRB CEP [m]")
    ax.set_title(
        f"Real-terrain PCRB validation\n"
        f"R² (log₁₀) = {r2:.3f}   n = {mask.sum()}"
    )
    ax.legend(fontsize=8, title="Terrain preset")
    fig.tight_layout()
    plot_path = outdir / "validation_scatter.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {plot_path}")
    print(f"\n  Validation R² (log₁₀ CEP) = {r2:.4f}  ({mask.sum()} points)")


def _compute_terrain_diagnostics(field: np.ndarray) -> dict[str, float]:
    """
    Compute the three terrain diagnostics used by the correction framework.

    Parameters
    ----------
    field : ndarray
        Normalised terrain patch (zero-mean, unit-variance).

    Returns
    -------
    dict with keys:
        ``skewness``         — scipy skewness of pixel values
        ``heterogeneity``    — patch_heterogeneity() coefficient of variation
        ``powerlaw_r2``      — R² of log-log linear fit to radial PSD (1 = pure power law)
    """
    skew = float(scipy.stats.skew(field.ravel()))

    het = patch_heterogeneity(field, n_subdivisions=4)

    # Radial PSD fit
    ny, nx = field.shape
    F = np.fft.fftshift(np.fft.fft2(field))
    psd = np.abs(F) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    FX, FY = np.meshgrid(fx, fy)
    freq_r = np.sqrt(FX ** 2 + FY ** 2).ravel()
    psd_r = psd.ravel()
    mask = (freq_r > 0) & (psd_r > 0)
    if mask.sum() > 10:
        log_f = np.log10(freq_r[mask])
        log_p = np.log10(psd_r[mask])
        coeffs = np.polyfit(log_f, log_p, 1)
        log_p_fit = np.polyval(coeffs, log_f)
        ss_res = np.sum((log_p - log_p_fit) ** 2)
        ss_tot = np.sum((log_p - log_p.mean()) ** 2)
        powerlaw_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        powerlaw_r2 = 0.0

    return {"skewness": skew, "heterogeneity": het, "powerlaw_r2": powerlaw_r2}


def cmd_correction(args: argparse.Namespace) -> None:
    """
    Diagnostic-driven correction framework (Task 15).

    Loads cached terrain tiles, re-runs PCRB validation to collect per-region
    diagnostics (skewness, heterogeneity, power-law fit quality), fits a
    log-linear regression to predict the correction factor, and saves the
    model + per-region summary table.
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = _load_model(model_path)
    feature_names = _load_feature_names(model_path)
    print(f"Loaded model from {model_path}")

    preset_names = args.presets or list(PRESETS.keys())
    speeds = args.speeds
    durations = args.durations
    n_seeds = args.n_seeds

    # ------------------------------------------------------------------
    # Phase 1: collect per-region diagnostics + correction factors
    # ------------------------------------------------------------------
    region_records: list[dict] = []
    print("\n=== Phase 1: collecting diagnostics per region ===")

    for name in preset_names:
        if name not in PRESETS:
            continue
        lat, lon, label = PRESETS[name]
        print(f"\n  {name}  ({label})")
        try:
            field_full, dx = load_terrain_patch(lat, lon, size_km=args.size_km, verbose=False)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        diag = _compute_terrain_diagnostics(field_full)
        print(f"    skewness={diag['skewness']:.3f}  het={diag['heterogeneity']:.3f}  "
              f"powerlaw_r2={diag['powerlaw_r2']:.3f}")

        # Run PCRB + predictions to collect correction factors
        ny, nx = field_full.shape
        for speed in speeds:
            for duration in durations:
                for seed in range(n_seeds):
                    rng_seed = np.random.default_rng(seed * 1000 + int(speed) * 100 + int(duration))
                    dy = int(rng_seed.integers(-ny // 10, ny // 10 + 1))
                    dx_off = int(rng_seed.integers(-nx // 10, nx // 10 + 1))
                    cy = np.clip(ny // 2 + dy, ny // 4, 3 * ny // 4)
                    cx = np.clip(nx // 2 + dx_off, nx // 4, 3 * nx // 4)
                    half = min(cy, nx // 2 - abs(cx - nx // 2), ny // 2 - abs(cy - ny // 2))
                    half = max(half, ny // 4)
                    patch = field_full[cy - half:cy + half, cx - half:cx + half].copy()
                    std = patch.std()
                    if std < 1e-9:
                        continue
                    patch = (patch - patch.mean()) / std

                    try:
                        row = run_pcrb_on_patch(
                            patch, dx, args.sigma_meas, speed, duration,
                            args.n_arms, args.sigma_init, args.dt,
                        )
                    except Exception:
                        continue

                    feat_vec = [row.get(f, np.nan) for f in feature_names]
                    if any(np.isnan(v) for v in feat_vec):
                        continue
                    cep_pred = 10.0 ** float(model.predict(np.array([feat_vec]))[0])
                    cep_actual = row["pcrb_cep_actual_m"]
                    if cep_pred <= 0 or cep_actual <= 0:
                        continue

                    region_records.append({
                        "preset": name,
                        "speed_m_s": speed,
                        "duration_s": duration,
                        "seed": seed,
                        "cep_actual": cep_actual,
                        "cep_predicted": cep_pred,
                        "correction_factor": cep_actual / cep_pred,
                        **diag,
                    })

    if not region_records:
        print("No records collected — check model and terrain access.")
        return

    df_corr = pd.DataFrame(region_records)
    df_corr.to_csv(outdir / "correction_data.csv", index=False)

    # ------------------------------------------------------------------
    # Phase 2: per-region summary
    # ------------------------------------------------------------------
    print("\n=== Phase 2: per-region correction factor summary ===")
    summary_rows: list[dict] = []
    for name in df_corr["preset"].unique():
        sub = df_corr[df_corr["preset"] == name]
        factors = sub["correction_factor"].values
        diag_mean = sub[["skewness", "heterogeneity", "powerlaw_r2"]].mean()
        row_sum = {
            "preset": name,
            "n": len(sub),
            "median_factor": float(np.median(factors)),
            "mean_factor": float(np.mean(factors)),
            "p25_factor": float(np.percentile(factors, 25)),
            "p75_factor": float(np.percentile(factors, 75)),
            **diag_mean.to_dict(),
        }
        summary_rows.append(row_sum)
        print(f"  {name:15s}  n={len(sub):4d}  "
              f"median_factor={row_sum['median_factor']:.2f}×  "
              f"het={diag_mean['heterogeneity']:.3f}  "
              f"skew={diag_mean['skewness']:.2f}")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(outdir / "correction_summary.csv", index=False)
    print(f"\nPer-region summary → {outdir / 'correction_summary.csv'}")

    # ------------------------------------------------------------------
    # Phase 3: fit log-linear correction model
    # ------------------------------------------------------------------
    print("\n=== Phase 3: fitting diagnostic-driven correction model ===")
    log_factor = np.log10(df_corr["correction_factor"].values)
    diag_matrix = df_corr[["skewness", "heterogeneity", "powerlaw_r2"]].values
    reg = LinearRegression()
    reg.fit(diag_matrix, log_factor)
    log_factor_pred = reg.predict(diag_matrix)
    ss_res = np.sum((log_factor - log_factor_pred) ** 2)
    ss_tot = np.sum((log_factor - log_factor.mean()) ** 2)
    corr_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    print(f"  Correction model R² = {corr_r2:.4f}")
    print(f"  Intercept:  {10 ** reg.intercept_:.4f}×")
    feature_names_diag = ["skewness", "heterogeneity", "powerlaw_r2"]
    for fname, coef in zip(feature_names_diag, reg.coef_):
        print(f"  coef[{fname}] = {coef:.4f}  (log₁₀ space)")

    # Save correction model
    corr_model_path = outdir / "correction_model.pkl"
    with open(corr_model_path, "wb") as f:
        pickle.dump({"model": reg, "features": feature_names_diag,
                     "r2": corr_r2, "intercept": float(reg.intercept_),
                     "coef": reg.coef_.tolist()}, f)
    print(f"  Correction model → {corr_model_path}")

    # ------------------------------------------------------------------
    # Phase 4: compare standard vs robust normalization on whole tiles
    # ------------------------------------------------------------------
    print("\n=== Phase 4: normalization sensitivity (StandardScaler vs RobustScaler) ===")
    norm_records: list[dict] = []
    for name in preset_names[:min(len(preset_names), 7)]:  # all presets
        if name not in PRESETS:
            continue
        lat, lon, label = PRESETS[name]
        try:
            field_full, dx = load_terrain_patch(lat, lon, size_km=args.size_km, verbose=False)
        except Exception:
            continue

        for norm_name, scaler_cls in [("standard", StandardScaler), ("robust", RobustScaler)]:
            sc = scaler_cls()
            field_norm = sc.fit_transform(field_full.reshape(-1, 1)).reshape(field_full.shape)
            map_info = map_information_content(field_norm, dx=dx, sigma_meas=args.sigma_meas)
            feat_vals = [map_info.get(f, np.nan) for f in feature_names
                         if f not in ("speed_m_s", "total_distance_m", "sigma_init_m")]
            # Use median speed/duration/sigma for comparison
            speed_med = float(np.median(speeds))
            dur_med = float(np.median(durations))
            feat_vec = feat_vals + [speed_med, speed_med * dur_med, args.sigma_init]
            if len(feat_vec) != len(feature_names) or any(np.isnan(v) for v in feat_vec):
                continue
            cep_pred = 10.0 ** float(model.predict(np.array([feat_vec]))[0])
            norm_records.append({"preset": name, "normalizer": norm_name, "cep_pred_m": cep_pred,
                                  "gradient_rms": map_info.get("gradient_rms", np.nan)})

    if norm_records:
        df_norm = pd.DataFrame(norm_records)
        df_norm_wide = df_norm.pivot_table(index="preset", columns="normalizer", values="cep_pred_m")
        print("\n  Predicted CEP by normalizer (median speed/duration):")
        print(df_norm_wide.to_string(float_format="{:.1f}".format))
        df_norm.to_csv(outdir / "normalizer_comparison.csv", index=False)
        print(f"\n  Normalization comparison → {outdir / 'normalizer_comparison.csv'}")

    # ------------------------------------------------------------------
    # Correction factor scatter plots (one file per diagnostic)
    # ------------------------------------------------------------------
    diag_labels = [("heterogeneity", "Patch heterogeneity (CV)", "correction_scatter_heterogeneity.png"),
                   ("skewness", "Terrain skewness", "correction_scatter_skewness.png"),
                   ("powerlaw_r2", "Power-law PSD fit R²", "correction_scatter_powerlaw_r2.png")]
    presets_list = df_corr["preset"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(presets_list)))

    for first, (diag_col, diag_label, fname) in enumerate(diag_labels):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for color, pname in zip(colors, presets_list):
            sub = df_corr[df_corr["preset"] == pname]
            ax.scatter(sub[diag_col], sub["correction_factor"], s=6, alpha=0.4,
                       color=color, label=pname)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(diag_label)
        ax.set_ylabel("Correction factor (actual/predicted)")
        ax.set_yscale("log")
        ax.set_title(
            f"Correction factor vs {diag_label}\n"
            f"(correction model R²={corr_r2:.3f})"
        )
        ax.grid(True, alpha=0.3)
        if first == 0:
            ax.legend(fontsize=7, title="Region")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
        print(f"\nCorrection scatter → {outdir / fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PCRB model on real SRTM terrain patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--presets", nargs="+", choices=list(PRESETS.keys()), default=None,
        help="Terrain presets to validate. Default: all.",
    )
    parser.add_argument(
        "--size-km", type=float, default=50.0, dest="size_km",
        help="Terrain patch side length [km].",
    )
    parser.add_argument(
        "--speeds", nargs="+", type=float, default=[5.0, 10.0, 20.0],
        help="Vehicle speeds [m/s].",
    )
    parser.add_argument(
        "--durations", nargs="+", type=float, default=[600.0, 1200.0, 2400.0],
        help="Trajectory durations [s].",
    )
    parser.add_argument(
        "--sigma-init", type=float, default=500.0, dest="sigma_init",
        help="Initial position uncertainty 1-sigma [m].",
    )
    parser.add_argument(
        "--sigma-meas", type=float, default=0.1, dest="sigma_meas",
        help="Measurement noise std [field units].",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10, dest="n_seeds",
        help="Number of random sub-patch offsets per (speed, duration) pair.",
    )
    parser.add_argument(
        "--n-arms", type=int, default=8, dest="n_arms",
        help="Number of star trajectory arms.",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0,
        help="Trajectory time step [s].",
    )
    parser.add_argument(
        "--model", default="data/theoretical/analysis/model.pkl",
        help="Path to trained model pickle.",
    )
    parser.add_argument(
        "--outdir", default="data/validation",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-tile download messages.",
    )

    # --- correction subcommand ---
    p_corr = parser.add_subparsers(dest="subcmd")
    p_correction = p_corr.add_parser(
        "correction",
        help="Fit diagnostic-driven correction framework and compare normalizers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_correction.add_argument(
        "--presets", nargs="+", choices=list(PRESETS.keys()), default=None,
    )
    p_correction.add_argument("--size-km", type=float, default=50.0, dest="size_km")
    p_correction.add_argument("--speeds", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    p_correction.add_argument("--durations", nargs="+", type=float, default=[600.0, 1200.0, 2400.0])
    p_correction.add_argument("--sigma-init", type=float, default=500.0, dest="sigma_init")
    p_correction.add_argument("--sigma-meas", type=float, default=0.1, dest="sigma_meas")
    p_correction.add_argument("--n-seeds", type=int, default=5, dest="n_seeds")
    p_correction.add_argument("--n-arms", type=int, default=8, dest="n_arms")
    p_correction.add_argument("--dt", type=float, default=1.0)
    p_correction.add_argument("--model", default="data/theoretical/analysis/model.pkl")
    p_correction.add_argument("--outdir", default="data/validation")

    args = parser.parse_args()

    # Route to correction subcommand
    if hasattr(args, "subcmd") and args.subcmd == "correction":
        cmd_correction(args)
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `analyze-results analyze` first."
        )
    model = _load_model(model_path)
    feature_names = _load_feature_names(model_path)
    print(f"Loaded model from {model_path}")
    print(f"  Feature names: {feature_names}")

    preset_names = args.presets or list(PRESETS.keys())
    all_frames: list[pd.DataFrame] = []

    for name in preset_names:
        lat, lon, label = PRESETS[name]
        print(f"\n{'='*60}")
        print(f"  {name}  ({label})")
        print(f"{'='*60}")
        df_preset = validate_preset(
            preset_name=name,
            lat=lat,
            lon=lon,
            size_km=args.size_km,
            speeds=args.speeds,
            durations=args.durations,
            sigma_init=args.sigma_init,
            sigma_meas=args.sigma_meas,
            model=model,
            feature_names=feature_names,
            n_seeds=args.n_seeds,
            n_arms=args.n_arms,
            dt=args.dt,
            verbose=not args.quiet,
        )
        if not df_preset.empty:
            all_frames.append(df_preset)
            n = len(df_preset)
            actual = df_preset["pcrb_cep_actual_m"].values
            predicted = df_preset["pcrb_cep_predicted_m"].values
            mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
            if mask.sum() >= 2:
                r2 = r2_score(
                    np.log10(actual[mask]),
                    np.log10(predicted[mask]),
                )
                print(f"  {n} scenarios  R²={r2:.3f}")
            else:
                print(f"  {n} scenarios  (insufficient data for R²)")

    if not all_frames:
        print("\nNo validation data collected.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)
    csv_path = outdir / "validation_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}  ({len(df_all)} rows)")

    _plot_validation(df_all, outdir)


if __name__ == "__main__":
    main()
