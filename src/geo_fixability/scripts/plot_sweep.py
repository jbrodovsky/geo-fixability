"""
plot_sweep.py
=============
Publication-quality figures from a completed geo-fixability parameter sweep CSV.

Each figure is saved individually as both PDF (LaTeX-ready) and PNG (preview).
Figures are designed at 7 × 5 inches (double-column width) unless noted.

Figures produced
----------------
fig_scenario_illustration  — Hero figure: synthetic map + star trajectory +
                             shrinking PCRB uncertainty ellipses.
fig_uncertainty_evolution  — PCRB uncertainty vs time for representative
                             scenarios spanning the information-index range.
fig_map_quality_space      — gradient_rms vs information_index coloured by
                             CEP improvement, showing what makes a map useful.
fig_improvement_heatmap_*  — Four 2-D heatmaps for key parameter pairs:
                               sigma_init × duration
                               sigma_meas × beta
                               correlation_length × dx
                               ins_drift_rate × sigma_meas
fig_marginal_effects       — Violin plots of log10(improvement) per value of
                             each of the eight swept dimensions.
fig_scaling_law            — Log-log: improvement vs cumulative FIM trace with
                             power-law fit and 90 % prediction band.
fig_feature_importance     — Spearman |ρ| bar chart for all input / derived
                             variables vs log10(cep_improvement_ratio).
fig_initial_vs_final_cep   — Log-log scatter: initial vs final CEP with
                             improvement-ratio isolines.
fig_pf_vs_pcrb             — (skipped if no PF data) PF vs PCRB improvement
                             scatter, coloured by measurement noise.

Usage
-----
    uv run plot-sweep
    uv run plot-sweep --csv data/sweep/sweep_results.csv --outdir paper/figures
    uv run plot-sweep --no-illustration   # skip the slow scenario illustration
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Global style — overrides on top of seaborn whitegrid
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
        "lines.linewidth": 1.8,
    }
)

_DPI_SAVE = 300          # print-quality output
_FIG_W = 7.0             # inches — double-column width
_FIG_H = 5.0

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CSV = REPO_ROOT / "data" / "sweep" / "sweep_results.csv"
_DEFAULT_OUT = REPO_ROOT / "paper" / "figures"

# Colour constants shared across figures
_CMAP_DIV = "RdYlGn"     # red (bad) → yellow → green (good)
_CMAP_SEQ = "viridis"
_CMAP_HEAT = "plasma"

# Human-readable labels for the swept parameters
_PARAM_LABELS: dict[str, str] = {
    "sigma_init":         r"$\sigma_0$ [m]",
    "speed":              "Speed [m/s]",
    "duration":           "Duration [s]",
    "ins_drift_rate":     "INS drift [m/s]",
    "sigma_meas":         r"$\sigma_\mathrm{meas}$ [f.u.]",
    "beta":               r"$\beta$ (spectral slope)",
    "correlation_length": r"$\lambda_c$ [m]",
    "dx":                 r"$\Delta x$ [m/px]",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{stem}.png"
    fig.savefig(p, dpi=_DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ok = df["status"].isin(["ok", "warn_oob"])
    n_bad = (~ok).sum()
    if n_bad:
        print(f"  Dropping {n_bad} non-ok rows (status counts: "
              f"{df['status'].value_counts().to_dict()})")
    df = df[ok].copy()
    df["log10_improvement"] = np.log10(df["cep_improvement_ratio"].clip(lower=1e-3))
    df["log10_fim"] = np.log10(df["cumulative_fim_trace"].clip(lower=1e-6))
    df["log10_initial_cep"] = np.log10(df["initial_cep_m"].clip(lower=1e-3))
    df["log10_final_cep"] = np.log10(df["final_cep_m"].clip(lower=1e-3))
    return df


def _cov_ellipse(
    ax: plt.Axes,
    cov: np.ndarray,
    center: tuple[float, float],
    n_std: float = 2.0,
    **kw,
) -> Ellipse:
    """Draw a 2-D Gaussian confidence ellipse (n_std σ) on *ax*."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2.0 * n_std * np.sqrt(vals[::-1])
    ell = Ellipse(xy=center, width=w, height=h, angle=angle, **kw)
    ax.add_patch(ell)
    return ell


def _power_law_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit log y = a·log x + b.  Returns (slope, intercept, r²) in log-log space."""
    mask = np.isfinite(x) & np.isfinite(y)
    res = stats.linregress(x[mask], y[mask])
    return float(res.slope), float(res.intercept), float(res.rvalue ** 2)


def _median_iqr_by(
    df: pd.DataFrame, x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_vals, medians, q25, q75) grouped by unique values of x_col."""
    x_vals = np.sort(df[x_col].unique())
    medians, q25, q75 = [], [], []
    for x in x_vals:
        sub = df.loc[df[x_col] == x, y_col].dropna()
        medians.append(sub.median())
        q25.append(sub.quantile(0.25))
        q75.append(sub.quantile(0.75))
    return x_vals, np.array(medians), np.array(q25), np.array(q75)


# ---------------------------------------------------------------------------
# Figure 1: Scenario illustration
# ---------------------------------------------------------------------------


def fig_scenario_illustration(outdir: Path) -> None:
    """
    Hero figure: synthetic map + star trajectory + PCRB uncertainty ellipses.

    Runs one representative scenario inline (no CSV needed) so the figure
    shows the concrete pipeline inputs and outputs.
    """
    from geo_fixability.mapping import generate_field_spectral
    from geo_fixability.navigation import pcrb_trajectory
    from geo_fixability.trajectory import calibrate_sigma_a, star_trajectory

    # --- Parameters ---
    BETA = 2.0
    DX = 50.0            # m/pixel
    CORR_LEN_M = 1000.0  # m
    SIGMA_INIT = 1000.0  # m
    SIGMA_MEAS = 0.1
    SPEED = 5.0          # m/s
    MAP_NPX = 256
    D_ARM = 0.40 * MAP_NPX * DX  # arm length [m]

    print("  [scenario_illustration] calibrating INS …")
    sigma_a, Q = calibrate_sigma_a(
        target_drift_rate=0.27778, eval_time=3600.0, dt=1.0, n_mc=100, seed=42
    )

    print("  [scenario_illustration] generating map …")
    rng_state = np.random.get_state()
    np.random.seed(7)
    field = generate_field_spectral(
        shape=(MAP_NPX, MAP_NPX),
        beta=BETA,
        correlation_length=CORR_LEN_M / DX,
    )
    np.random.set_state(rng_state)

    print("  [scenario_illustration] running PCRB …")
    positions, times = star_trajectory(D=D_ARM, speed=SPEED, dt=1.0)

    P_init = SIGMA_INIT ** 2 * np.eye(2)
    P_traj = pcrb_trajectory(positions, field, DX, SIGMA_MEAS, P_init, Q)

    # Uncertainty time series: sqrt(trace(P))
    pos_unc = np.sqrt(np.trace(P_traj, axis1=1, axis2=2))

    # --- Layout: left = map panel, right = time series ---
    fig, (ax_map, ax_ts) = plt.subplots(
        1, 2, figsize=(_FIG_W, _FIG_H),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )

    # — Map panel —
    half = MAP_NPX * DX / 2.0
    extent = [-half, half, -half, half]
    im = ax_map.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        alpha=0.85,
        vmin=-2.5,
        vmax=2.5,
    )
    cb = fig.colorbar(im, ax=ax_map, shrink=0.72, pad=0.02)
    cb.set_label("Field amplitude [σ]", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Trajectory
    ax_map.plot(
        positions[:, 0], positions[:, 1],
        color="k", lw=0.8, alpha=0.7, zorder=3, label="Trajectory",
    )

    # Initial uncertainty ellipse (2σ) at origin
    _cov_ellipse(
        ax_map, P_traj[0], (0.0, 0.0), n_std=2,
        facecolor="royalblue", alpha=0.25, edgecolor="royalblue", lw=1.5,
        zorder=4, label=rf"Initial $2\sigma$ ({SIGMA_INIT:.0f} m)",
    )

    # Final uncertainty ellipse (2σ) at last position
    px_f, py_f = float(positions[-1, 0]), float(positions[-1, 1])
    _cov_ellipse(
        ax_map, P_traj[-1], (px_f, py_f), n_std=2,
        facecolor="crimson", alpha=0.6, edgecolor="crimson", lw=1.5,
        zorder=5, label=rf"Final $2\sigma$ ({pos_unc[-1]:.1f} m)",
    )
    # Mark start
    ax_map.plot(0, 0, "b^", ms=7, zorder=6)
    ax_map.set_xlim(-half, half)
    ax_map.set_ylim(-half, half)
    ax_map.set_xlabel("East [m]")
    ax_map.set_ylabel("North [m]")
    ax_map.set_title("Synthetic map + trajectory")
    ax_map.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Format axis ticks in km for readability
    km_ticks = np.linspace(-half, half, 5)
    ax_map.set_xticks(km_ticks)
    ax_map.set_xticklabels([f"{x/1000:.1f}" for x in km_ticks])
    ax_map.set_yticks(km_ticks)
    ax_map.set_yticklabels([f"{y/1000:.1f}" for y in km_ticks])
    ax_map.set_xlabel("East [km]")
    ax_map.set_ylabel("North [km]")

    # — Time series panel —
    t_min = times / 60.0  # convert to minutes for readability
    ax_ts.semilogy(t_min, pos_unc, color="#2166ac", lw=2.0, label="PCRB σ(t)")
    ax_ts.axhline(
        pos_unc[0], color="#d6604d", ls="--", lw=1.2,
        label=rf"Initial $\sigma_0 = {SIGMA_INIT:.0f}$ m",
    )
    ax_ts.axhline(
        pos_unc[-1], color="#4dac26", ls=":", lw=1.5,
        label=rf"Final $\sigma_f = {pos_unc[-1]:.1f}$ m",
    )

    ratio = pos_unc[0] / pos_unc[-1]
    ax_ts.text(
        0.97, 0.50,
        f"Improvement\n{ratio:.0f}×",
        transform=ax_ts.transAxes,
        ha="right", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax_ts.set_xlabel("Time [min]")
    ax_ts.set_ylabel("Position uncertainty σ [m]")
    ax_ts.set_title("PCRB uncertainty reduction")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.set_xlim(left=0)

    # Annotate map parameters
    param_text = (
        rf"$\beta={BETA}$, $\lambda_c={CORR_LEN_M:.0f}$ m"
        "\n"
        rf"$\Delta x={DX:.0f}$ m/px,  $\sigma_\mathrm{{meas}}={SIGMA_MEAS}$"
        "\n"
        rf"Speed $={SPEED:.0f}$ m/s"
    )
    fig.text(
        0.52, 0.01, param_text,
        ha="left", va="bottom", fontsize=8, color="0.4",
    )

    fig.suptitle(
        "Geophysical map-aided navigation: pipeline illustration",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig_scenario_illustration", outdir)


# ---------------------------------------------------------------------------
# Figure 2: Uncertainty evolution for representative scenarios
# ---------------------------------------------------------------------------


def fig_uncertainty_evolution(df: pd.DataFrame, outdir: Path) -> None:
    """
    Re-run PCRB for a handful of scenarios spanning the information_index range
    and overlay their uncertainty time series.
    """
    from geo_fixability.mapping import generate_field_spectral
    from geo_fixability.navigation import pcrb_trajectory
    from geo_fixability.trajectory import calibrate_sigma_a, star_trajectory

    # Pick 5 scenarios with spread in information_index
    df_sorted = df.sort_values("information_index").reset_index(drop=True)
    n = len(df_sorted)
    indices = [int(n * q) for q in [0.05, 0.25, 0.5, 0.75, 0.95]]
    indices = [min(i, n - 1) for i in indices]
    selected = df_sorted.iloc[indices]

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / 4) for i in range(5)]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    print("  [uncertainty_evolution] re-running PCRB for 5 representative scenarios …")

    # Cache calibrations
    cal_cache: dict[tuple, tuple] = {}

    for i, (_, row) in enumerate(selected.iterrows()):
        key = (float(row["ins_drift_rate"]), float(row["dt"]))
        if key not in cal_cache:
            sig_a, Q = calibrate_sigma_a(
                float(row["ins_drift_rate"]),
                eval_time=max(float(row["duration"]), 3600.0),
                dt=float(row["dt"]),
                n_mc=100, seed=42,
            )
            cal_cache[key] = (sig_a, Q)
        sigma_a, Q = cal_cache[key]

        map_npix = int(row["map_npix"])
        corr_px = float(row["correlation_length"]) / float(row["dx"])
        corr_px = min(corr_px, map_npix / 4.0)

        np.random.seed(int(row["seed"]))
        field = generate_field_spectral(
            (map_npix, map_npix),
            beta=float(row["beta"]),
            correlation_length=corr_px,
        )

        D = 0.4 * map_npix * float(row["dx"])
        positions, times = star_trajectory(D=D, speed=float(row["speed"]), dt=float(row["dt"]))

        P_init = float(row["sigma_init"]) ** 2 * np.eye(2)
        P_traj = pcrb_trajectory(positions, field, float(row["dx"]),
                                 float(row["sigma_meas"]), P_init, Q)

        pos_unc = np.sqrt(np.trace(P_traj, axis1=1, axis2=2))
        t_min = times / 60.0

        label = (
            rf"$I={row['information_index']:.2f}$,  "
            rf"$\sigma_0={row['sigma_init']:.0f}$ m,  "
            rf"{row['cep_improvement_ratio']:.0f}×"
        )
        ax.semilogy(t_min, pos_unc, color=colors[i], lw=1.8, label=label)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("PCRB position uncertainty σ [m]")
    ax.set_title("Uncertainty evolution for representative scenarios")
    ax.legend(fontsize=8, title="Info index / init σ / improvement", loc="upper right")
    ax.set_xlim(left=0)

    fig.tight_layout()
    _save(fig, "fig_uncertainty_evolution", outdir)


# ---------------------------------------------------------------------------
# Figure 3: Map quality space
# ---------------------------------------------------------------------------


def fig_map_quality_space(df: pd.DataFrame, outdir: Path) -> None:
    """
    gradient_rms (x, log) vs information_index (y) coloured by
    log10(cep_improvement_ratio).  Shows which part of map space is useful.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    sc = ax.scatter(
        df["gradient_rms"],
        df["information_index"],
        c=df["log10_improvement"],
        cmap=_CMAP_SEQ,
        s=10,
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$(CEP improvement ratio)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xscale("log")
    ax.set_xlabel(r"Gradient RMS $|\nabla h|$ [σ/m]")
    ax.set_ylabel("Information index [0–1]")
    ax.set_title("Map quality space coloured by navigation improvement")

    # Marginal rug on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale("log")
    ax2.set_xticklabels([])
    ax2.tick_params(length=0)

    fig.tight_layout()
    _save(fig, "fig_map_quality_space", outdir)


# ---------------------------------------------------------------------------
# Figure 4a–d: Parameter heatmaps
# ---------------------------------------------------------------------------


def _heatmap(
    df: pd.DataFrame,
    row_param: str,
    col_param: str,
    outdir: Path,
    stem: str,
    title: str,
) -> None:
    pivot = (
        df.groupby([row_param, col_param])["log10_improvement"]
        .median()
        .unstack(col_param)
    )

    fig, ax = plt.subplots(figsize=(_FIG_W, 0.9 * _FIG_H))

    vmin = df["log10_improvement"].quantile(0.05)
    vmax = df["log10_improvement"].quantile(0.95)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap=_CMAP_HEAT,
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"Median $\log_{10}$(CEP improvement)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=8)
    ax.set_xlabel(_PARAM_LABELS.get(col_param, col_param))
    ax.set_ylabel(_PARAM_LABELS.get(row_param, row_param))
    ax.set_title(title)

    # Annotate cells with numeric values
    for ri, row_v in enumerate(pivot.index):
        for ci, col_v in enumerate(pivot.columns):
            val = pivot.loc[row_v, col_v]
            if np.isfinite(val):
                text_color = "white" if val < (vmin + vmax) / 2 else "k"
                ax.text(
                    ci, ri, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

    fig.tight_layout()
    _save(fig, stem, outdir)


def fig_parameter_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    """Four heatmaps for key parameter pairs."""
    pairs = [
        ("sigma_init", "duration",
         "fig_heatmap_sigma_duration",
         r"Improvement: initial uncertainty $\sigma_0$ × mission duration"),
        ("sigma_meas", "beta",
         "fig_heatmap_sigmameas_beta",
         r"Improvement: sensor noise $\sigma_\mathrm{meas}$ × spectral slope $\beta$"),
        ("correlation_length", "dx",
         "fig_heatmap_corr_dx",
         r"Improvement: feature scale $\lambda_c$ × map resolution $\Delta x$"),
        ("ins_drift_rate", "sigma_meas",
         "fig_heatmap_drift_sigmameas",
         r"Improvement: INS drift × sensor noise $\sigma_\mathrm{meas}$"),
    ]
    for row_p, col_p, stem, title in pairs:
        if row_p in df.columns and col_p in df.columns:
            _heatmap(df, row_p, col_p, outdir, stem, title)


# ---------------------------------------------------------------------------
# Figure 5: Marginal effects
# ---------------------------------------------------------------------------


def fig_marginal_effects(df: pd.DataFrame, outdir: Path) -> None:
    """
    Violin plots of log10(CEP improvement) vs each swept parameter independently.
    4 × 2 grid, one subplot per parameter.
    """
    params = list(_PARAM_LABELS.keys())
    params = [p for p in params if p in df.columns]
    n = len(params)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_FIG_W * ncols * 0.6, _FIG_H * nrows * 0.55),
        squeeze=False,
    )

    for idx, param in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        vals = np.sort(df[param].unique())
        data = [df.loc[df[param] == v, "log10_improvement"].dropna().values for v in vals]
        parts = ax.violinplot(data, positions=range(len(vals)), showmedians=True, showextrema=False)

        for body in parts["bodies"]:
            body.set_alpha(0.65)
            body.set_facecolor("#4c72b0")
        parts["cmedians"].set_color("crimson")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(r"$\log_{10}$(improvement)", fontsize=8)
        ax.set_xlabel(_PARAM_LABELS.get(param, param), fontsize=8)
        ax.set_title(_PARAM_LABELS.get(param, param), fontsize=9)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Marginal effect of each swept parameter on CEP improvement",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig_marginal_effects", outdir)


# ---------------------------------------------------------------------------
# Figure 6: Scaling law
# ---------------------------------------------------------------------------


def fig_scaling_law(df: pd.DataFrame, outdir: Path) -> None:
    """
    Log-log scatter of CEP improvement ratio vs cumulative FIM trace with
    power-law fit and 90 % prediction interval.
    """
    x = df["log10_fim"].values
    y = df["log10_improvement"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    slope, intercept, r2 = _power_law_fit(x, y)

    # 90 % prediction interval via residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred
    se = residuals.std()
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    sc = ax.scatter(
        x, y,
        c=np.log10(df.loc[mask, "sigma_init"].values),
        cmap="coolwarm",
        s=8, alpha=0.5, edgecolors="none", rasterized=True, zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(r"$\log_{10}(\sigma_0)$ [m]", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.fill_between(
        x_line,
        y_line - 1.645 * se,
        y_line + 1.645 * se,
        alpha=0.18, color="#d6604d", label="90 % prediction band",
    )
    ax.plot(
        x_line, y_line,
        color="#d6604d", lw=2.0,
        label=(
            rf"Power law fit: slope = {slope:.2f}"
            "\n"
            rf"$R^2 = {r2:.3f}$"
        ),
    )

    ax.set_xlabel(r"$\log_{10}(\Sigma\,\mathrm{FIM\ trace})$  [f.u.$^2$/m$^2$]")
    ax.set_ylabel(r"$\log_{10}$(CEP improvement ratio)")
    ax.set_title("Scaling law: improvement vs cumulative Fisher information")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    _save(fig, "fig_scaling_law", outdir)


# ---------------------------------------------------------------------------
# Figure 7: Feature importance (Spearman |ρ|)
# ---------------------------------------------------------------------------


def fig_feature_importance(df: pd.DataFrame, outdir: Path) -> None:
    """
    Horizontal bar chart of |Spearman ρ| between each predictor and
    log10(cep_improvement_ratio), sorted by magnitude.
    """
    candidates = {
        "sigma_init":            r"$\sigma_0$ (initial uncertainty)",
        "speed":                 "Speed",
        "duration":              "Duration",
        "ins_drift_rate":        "INS drift rate",
        "sigma_meas":            r"$\sigma_\mathrm{meas}$ (sensor noise)",
        "beta":                  r"$\beta$ (spectral slope)",
        "correlation_length":    r"$\lambda_c$ (feature scale)",
        "dx":                    r"$\Delta x$ (map resolution)",
        "gradient_rms":          r"$|\nabla h|$ RMS",
        "information_index":     "Information index",
        "localization_bound_m":  "Localisation bound",
        "localization_index":    "Localisation index",
        "effective_dof":         "Effective DOF",
        "cumulative_fim_trace":  "Cumulative FIM trace",
    }
    target = df["log10_improvement"]

    rows = []
    for col, label in candidates.items():
        if col not in df.columns:
            continue
        x = df[col].dropna()
        idx = x.index.intersection(target.dropna().index)
        if len(idx) < 10:
            continue
        rho, pval = stats.spearmanr(x.loc[idx], target.loc[idx])
        rows.append({"label": label, "rho": rho, "abs_rho": abs(rho), "pval": pval})

    feat_df = pd.DataFrame(rows).sort_values("abs_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(_FIG_W, max(3.5, len(feat_df) * 0.35)))

    colors = [
        "#2166ac" if r > 0 else "#d6604d"
        for r in feat_df["rho"]
    ]
    bars = ax.barh(
        feat_df["label"], feat_df["abs_rho"],
        color=colors, edgecolor="none", alpha=0.85,
    )
    # Annotate with ρ value and significance
    for bar, row in zip(bars, feat_df.itertuples()):
        sig = "***" if row.pval < 0.001 else ("**" if row.pval < 0.01 else ("*" if row.pval < 0.05 else ""))
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{row.rho:+.2f}{sig}",
            va="center", ha="left", fontsize=8,
        )

    # Legend for sign
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#2166ac", alpha=0.85, label="Positive ρ (higher → more improvement)"),
            Patch(color="#d6604d", alpha=0.85, label="Negative ρ (higher → less improvement)"),
        ],
        fontsize=8, loc="lower right",
    )

    ax.set_xlabel(r"|Spearman $\rho$|  vs  $\log_{10}$(CEP improvement)")
    ax.set_title("Feature importance: Spearman rank correlation")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.3, color="0.6", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.7, color="0.4", lw=0.8, ls="--", alpha=0.5)
    ax.text(0.3, -0.5, "moderate", ha="center", fontsize=7, color="0.5")
    ax.text(0.7, -0.5, "strong", ha="center", fontsize=7, color="0.4")

    fig.tight_layout()
    _save(fig, "fig_feature_importance", outdir)


# ---------------------------------------------------------------------------
# Figure 8: Initial vs final CEP
# ---------------------------------------------------------------------------


def fig_initial_vs_final_cep(df: pd.DataFrame, outdir: Path) -> None:
    """
    Log-log scatter: initial CEP vs final CEP, coloured by improvement ratio,
    with diagonal isolines marking 10×, 100×, 1000×, 10000× improvement.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    sc = ax.scatter(
        df["initial_cep_m"],
        df["final_cep_m"],
        c=df["log10_improvement"],
        cmap=_CMAP_SEQ,
        s=10, alpha=0.65, edgecolors="none", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$(CEP improvement)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Improvement isolines
    x_range = np.array([df["initial_cep_m"].min(), df["initial_cep_m"].max()])
    for ratio, ls, lw in [(10, ":", 0.9), (100, "--", 1.0), (1000, "-.", 1.1), (10000, "-", 1.2)]:
        y_iso = x_range / ratio
        ax.plot(x_range, y_iso, color="0.5", lw=lw, ls=ls, alpha=0.7)
        # Label at the right end
        ax.text(
            x_range[-1] * 0.85, y_iso[-1] * 1.15,
            f"{ratio}×", fontsize=7, color="0.45", ha="right",
        )

    # Unity line
    ax.plot(x_range, x_range, "k--", lw=0.8, alpha=0.4, label="No improvement")

    ax.set_xlabel(r"Initial CEP $\sigma_0$ [m]")
    ax.set_ylabel(r"Final CEP $\sigma_f$ [m]")
    ax.set_title("Initial vs final position uncertainty (PCRB)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_initial_vs_final_cep", outdir)


# ---------------------------------------------------------------------------
# Figure 9: Resolution sensitivity
# ---------------------------------------------------------------------------


def fig_resolution_sensitivity(df: pd.DataFrame, outdir: Path) -> None:
    """
    Median log10(improvement) vs dx, broken out by beta.
    Shows the interaction between map resolution and spectral character.
    """
    if "dx" not in df.columns or "beta" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    betas = np.sort(df["beta"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(betas) - 1, 1)) for i in range(len(betas))]

    for beta, color in zip(betas, colors):
        sub = df[df["beta"] == beta]
        x_vals, medians, q25, q75 = _median_iqr_by(sub, "dx", "log10_improvement")
        ax.plot(
            x_vals, medians,
            marker="o", color=color, lw=1.6,
            label=rf"$\beta = {beta}$",
        )
        ax.fill_between(x_vals, q25, q75, alpha=0.12, color=color)

    ax.set_xscale("log")
    ax.set_xlabel(r"Map resolution $\Delta x$ [m/px]")
    ax.set_ylabel(r"Median $\log_{10}$(CEP improvement)")
    ax.set_title(r"Improvement vs resolution, broken out by $\beta$")
    ax.legend(fontsize=8, title=r"Spectral slope $\beta$", title_fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_resolution_sensitivity", outdir)


# ---------------------------------------------------------------------------
# Figure 10: INS drift vs sensor noise trade-off
# ---------------------------------------------------------------------------


def fig_ins_sensor_tradeoff(df: pd.DataFrame, outdir: Path) -> None:
    """
    2-D scatter of ins_drift_rate vs sigma_meas coloured by improvement.
    Highlights the boundary where sensor noise or INS drift becomes limiting.
    """
    if "ins_drift_rate" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    sc = ax.scatter(
        df["ins_drift_rate"] * 3600,    # convert to m/hr for display
        df["sigma_meas"],
        c=df["log10_improvement"],
        cmap=_CMAP_DIV,
        s=12, alpha=0.7, edgecolors="none", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$(CEP improvement)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("INS drift rate [m/hr]")
    ax.set_ylabel(r"Measurement noise $\sigma_\mathrm{meas}$ [f.u.]")
    ax.set_title("INS drift vs sensor noise: improvement trade-off")

    fig.tight_layout()
    _save(fig, "fig_ins_sensor_tradeoff", outdir)


# ---------------------------------------------------------------------------
# Figure 11: PF vs PCRB (conditional)
# ---------------------------------------------------------------------------


def fig_pf_vs_pcrb(df: pd.DataFrame, outdir: Path) -> None:
    """
    Scatter of PF improvement vs PCRB improvement, coloured by sigma_meas.
    Only generated when particle-filter columns contain non-NaN data.
    """
    pf_col = "pf_improvement_ratio"
    if pf_col not in df.columns:
        return
    pf_data = df[pf_col].dropna()
    if len(pf_data) < 5:
        print("  [pf_vs_pcrb] insufficient PF data — skipping.")
        return

    sub = df[[pf_col, "cep_improvement_ratio", "sigma_meas"]].dropna()

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    sc = ax.scatter(
        sub["cep_improvement_ratio"],
        sub[pf_col],
        c=np.log10(sub["sigma_meas"]),
        cmap="coolwarm",
        s=12, alpha=0.7, edgecolors="none", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(\sigma_\mathrm{meas})$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lim_lo = min(sub["cep_improvement_ratio"].min(), sub[pf_col].min()) * 0.8
    lim_hi = max(sub["cep_improvement_ratio"].max(), sub[pf_col].max()) * 1.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.0, label="PF = PCRB (ideal)")

    rho, _ = stats.spearmanr(sub["cep_improvement_ratio"], sub[pf_col])
    ax.text(
        0.05, 0.92, rf"Spearman $\rho = {rho:.2f}$",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PCRB improvement ratio")
    ax.set_ylabel("PF improvement ratio")
    ax.set_title("Particle filter vs PCRB improvement ratio")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_pf_vs_pcrb", outdir)


# ---------------------------------------------------------------------------
# Figure 12: Flag atlas (edge cases)
# ---------------------------------------------------------------------------


def fig_flag_atlas(df: pd.DataFrame, outdir: Path) -> None:
    """
    Show where the map_size_capped and correlation_clamped flags fire in
    the (correlation_length, dx) parameter space.
    """
    needed = {"map_size_capped", "correlation_clamped", "correlation_length", "dx"}
    if not needed.issubset(df.columns):
        return
    if not (df["map_size_capped"].any() or df["correlation_clamped"].any()):
        print("  [flag_atlas] no flagged rows — skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, 0.8 * _FIG_H), sharey=True)

    for ax, flag_col, title, color in [
        (axes[0], "map_size_capped",    "Map size capped (field truncated)",  "#d6604d"),
        (axes[1], "correlation_clamped", r"Correlation clamped ($\lambda_c > L/4$)", "#4393c3"),
    ]:
        flagged = df[df[flag_col] == True]
        ok_rows = df[df[flag_col] == False]

        ax.scatter(
            ok_rows["dx"], ok_rows["correlation_length"],
            c="0.85", s=6, alpha=0.5, label="Normal", edgecolors="none",
        )
        if len(flagged):
            ax.scatter(
                flagged["dx"], flagged["correlation_length"],
                c=color, s=20, alpha=0.85, label="Flagged", edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta x$ [m/px]")
        ax.set_ylabel(r"$\lambda_c$ [m]")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Parameter-space regions triggering edge-case flags",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_flag_atlas", outdir)


# ---------------------------------------------------------------------------
# Summary statistics table (printed to stdout + saved as CSV)
# ---------------------------------------------------------------------------


def print_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    """Print and save a LaTeX-ready summary statistics table."""
    params = [p for p in _PARAM_LABELS if p in df.columns]
    rows = []
    for p in params:
        vals = df[p].unique()
        label = _PARAM_LABELS[p]
        rho, pval = stats.spearmanr(df[p], df["log10_improvement"])
        rows.append({
            "Parameter": label,
            "N values": len(vals),
            "Min": f"{vals.min():g}",
            "Max": f"{vals.max():g}",
            r"Spearman |ρ|": f"{abs(rho):.3f}",
            "p-value": f"{pval:.2e}",
        })

    tbl = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(outdir / "table_feature_correlations.csv", index=False)
    print("\nParameter sensitivity summary:")
    print(tbl.to_string(index=False))
    print(f"\n  → {outdir / 'table_feature_correlations.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from a geo-fixability sweep CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=str(_DEFAULT_CSV),
        help="Path to sweep_results.csv",
    )
    parser.add_argument(
        "--outdir",
        default=str(_DEFAULT_OUT),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--no-illustration",
        action="store_true",
        help="Skip fig_scenario_illustration (runs inline PCRB — slow on first call).",
    )
    parser.add_argument(
        "--no-evolution",
        action="store_true",
        help="Skip fig_uncertainty_evolution (re-runs 5 PCRB trajectories).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        sys.exit(
            f"CSV not found: {csv_path}\n"
            "Run  uv run run-sweep  first to generate sweep data."
        )

    print(f"Loading {csv_path} …")
    df = _load_csv(csv_path)
    print(f"  {len(df)} valid rows, {df['status'].value_counts().to_dict()}")

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {outdir}\n")

    # --- Figures that need inline PCRB runs ---
    if not args.no_illustration:
        print("Figure: scenario illustration")
        fig_scenario_illustration(outdir)

    if not args.no_evolution:
        print("Figure: uncertainty evolution")
        fig_uncertainty_evolution(df, outdir)

    # --- CSV-only figures ---
    print("Figure: map quality space")
    fig_map_quality_space(df, outdir)

    print("Figure: parameter heatmaps")
    fig_parameter_heatmaps(df, outdir)

    print("Figure: marginal effects")
    fig_marginal_effects(df, outdir)

    print("Figure: scaling law")
    fig_scaling_law(df, outdir)

    print("Figure: feature importance")
    fig_feature_importance(df, outdir)

    print("Figure: initial vs final CEP")
    fig_initial_vs_final_cep(df, outdir)

    print("Figure: resolution sensitivity")
    fig_resolution_sensitivity(df, outdir)

    print("Figure: INS/sensor trade-off")
    fig_ins_sensor_tradeoff(df, outdir)

    print("Figure: PF vs PCRB")
    fig_pf_vs_pcrb(df, outdir)

    print("Figure: flag atlas")
    fig_flag_atlas(df, outdir)

    print("Summary table")
    print_summary_table(df, outdir)

    print(f"\nAll done.  {len(list(outdir.glob('*.png')))} PNG figures in {outdir}")


if __name__ == "__main__":
    main()
