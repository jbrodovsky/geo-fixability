"""
plot_pf_results.py
==================
Load the particle-filter experiment CSV and produce individual publication-quality
figures plus a structured statistics summary printed to stdout.

Each figure is saved as a single panel (no subfigures); LaTeX handles layout.
Figures are saved to data/analysis/pf/.

Usage
-----
    uv run plot-pf-results
    uv run python src/geo_fixability/scripts/plot_pf_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "experiments" / "pf_experiment_results.csv"
OUT_DIR = REPO_ROOT / "data" / "analysis" / "pf"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load and lightly validate the PF experiment CSV."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _shaded_median_iqr(
    ax: plt.Axes,
    x_vals: np.ndarray,
    df: pd.DataFrame,
    col: str,
    label: str,
    color: str,
) -> None:
    """Plot median line + IQR shaded band against sorted x_vals (keyed by dx_m)."""
    medians, q25, q75 = [], [], []
    for x in x_vals:
        sub = df.loc[df["dx_m"] == x, col].dropna()
        medians.append(sub.median())
        q25.append(sub.quantile(0.25))
        q75.append(sub.quantile(0.75))
    ax.plot(x_vals, np.array(medians), marker="o", label=label, color=color, linewidth=1.8)
    ax.fill_between(x_vals, np.array(q25), np.array(q75), alpha=0.20, color=color)


def _find_sweet_spot(df: pd.DataFrame, dx_vals: np.ndarray) -> tuple[float, float]:
    medians = [df.loc[df["dx_m"] == x, "pf_vs_pcrb_ratio"].median() for x in dx_vals]
    best_idx = int(np.argmin(medians))
    lo = dx_vals[max(0, best_idx - 1)]
    hi = dx_vals[min(len(dx_vals) - 1, best_idx + 1)]
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Figure 1: Resolution sweep  (4 individual panels)
# ---------------------------------------------------------------------------

def fig1_resolution_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    dx_vals = np.sort(df["dx_m"].unique())
    ss_lo, ss_hi = _find_sweet_spot(df, dx_vals)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1a — improvement ratio
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _shaded_median_iqr(ax, dx_vals, df, "pf_improvement_ratio", "PF improvement ratio", colors[0])
    _shaded_median_iqr(ax, dx_vals, df, "pcrb_improvement_ratio", "PCRB improvement ratio", colors[1])
    ax.axvspan(ss_lo, ss_hi, alpha=0.12, color="green", label="Sweet spot")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("Improvement ratio")
    ax.set_title("PF vs PCRB improvement ratio")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig1a_res_improvement_ratio.png")

    # 1b — pf_vs_pcrb_ratio
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _shaded_median_iqr(ax, dx_vals, df, "pf_vs_pcrb_ratio", "PF vs PCRB ratio", colors[2])
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Ideal (=1)")
    ax.axvspan(ss_lo, ss_hi, alpha=0.12, color="green", label="Sweet spot")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("PF / PCRB ratio")
    ax.set_title("PF efficiency vs PCRB")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig1b_res_pf_vs_pcrb.png")

    # 1c — N_eff_min
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _shaded_median_iqr(ax, dx_vals, df, "pf_n_eff_min", "N_eff_min", colors[3])
    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Degeneracy threshold (50)")
    ax.axvspan(ss_lo, ss_hi, alpha=0.12, color="green", label="Sweet spot")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("N_eff_min")
    ax.set_title("Filter health (N_eff_min)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig1c_res_neff_min.png")

    # 1d — CEP final
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    _shaded_median_iqr(ax, dx_vals, df, "pf_cep_final_m", "PF CEP final [m]", colors[0])
    _shaded_median_iqr(ax, dx_vals, df, "pcrb_cep_final_m", "PCRB CEP final [m]", colors[1])
    ax.axvspan(ss_lo, ss_hi, alpha=0.12, color="green", label="Sweet spot")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("CEP final [m]")
    ax.set_title("Final CEP: PF vs PCRB")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig1d_res_cep_final.png")


# ---------------------------------------------------------------------------
# Figure 2: Map parameter effects  (4 individual panels)
# ---------------------------------------------------------------------------

def _boxplot_panel(df: pd.DataFrame, group_col: str, y_col: str,
                   xlabel: str, ylabel: str, title: str,
                   palette_name: str, hline: float | None = None) -> plt.Figure:
    order = sorted(df[group_col].unique())
    palette = sns.color_palette(palette_name, len(order))
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.boxplot(data=df, x=group_col, y=y_col, hue=group_col, order=order,
                hue_order=order, palette=palette, ax=ax, legend=False,
                flierprops={"markersize": 2})
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1.2, label=f"Ideal (={hline})")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig2_map_params(df: pd.DataFrame, out_dir: Path) -> None:
    _save(
        _boxplot_panel(df, "beta", "pf_improvement_ratio",
                       "beta (PSD slope)", "PF improvement ratio",
                       "PF improvement ratio by beta", "viridis"),
        out_dir / "fig2a_map_beta_improvement.png",
    )
    _save(
        _boxplot_panel(df, "corr_len_m", "pf_improvement_ratio",
                       "Correlation length [m]", "PF improvement ratio",
                       "PF improvement ratio by correlation length", "viridis"),
        out_dir / "fig2b_map_corr_improvement.png",
    )
    _save(
        _boxplot_panel(df, "beta", "pf_vs_pcrb_ratio",
                       "beta (PSD slope)", "PF / PCRB ratio",
                       "PF vs PCRB ratio by beta", "viridis", hline=1.0),
        out_dir / "fig2c_map_beta_ratio.png",
    )
    _save(
        _boxplot_panel(df, "corr_len_m", "pf_vs_pcrb_ratio",
                       "Correlation length [m]", "PF / PCRB ratio",
                       "PF vs PCRB ratio by correlation length", "viridis", hline=1.0),
        out_dir / "fig2d_map_corr_ratio.png",
    )


# ---------------------------------------------------------------------------
# Figure 3: Trajectory parameter effects  (4 individual panels)
# ---------------------------------------------------------------------------

def fig3_trajectory_params(df: pd.DataFrame, out_dir: Path) -> None:
    _save(
        _boxplot_panel(df, "speed_m_s", "pf_improvement_ratio",
                       "Speed [m/s]", "PF improvement ratio",
                       "PF improvement ratio by speed", "magma"),
        out_dir / "fig3a_traj_speed_improvement.png",
    )
    _save(
        _boxplot_panel(df, "duration_s", "pf_improvement_ratio",
                       "Duration [s]", "PF improvement ratio",
                       "PF improvement ratio by duration", "magma"),
        out_dir / "fig3b_traj_dur_improvement.png",
    )
    _save(
        _boxplot_panel(df, "speed_m_s", "pf_vs_pcrb_ratio",
                       "Speed [m/s]", "PF / PCRB ratio",
                       "PF vs PCRB ratio by speed", "magma", hline=1.0),
        out_dir / "fig3c_traj_speed_ratio.png",
    )
    _save(
        _boxplot_panel(df, "duration_s", "pf_vs_pcrb_ratio",
                       "Duration [s]", "PF / PCRB ratio",
                       "PF vs PCRB ratio by duration", "magma", hline=1.0),
        out_dir / "fig3d_traj_dur_ratio.png",
    )


# ---------------------------------------------------------------------------
# Figure 4: PCRB vs PF scatter  (unchanged — already single panel)
# ---------------------------------------------------------------------------

def fig4_pcrb_vs_pf(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    dx_log = np.log10(df["dx_m"])
    norm = mcolors.Normalize(vmin=dx_log.min(), vmax=dx_log.max())
    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(
        df["pcrb_improvement_ratio"], df["pf_improvement_ratio"],
        c=dx_log, cmap=cmap, norm=norm, s=18, alpha=0.6, linewidths=0,
    )

    xlim_min = min(df["pcrb_improvement_ratio"].min(), df["pf_improvement_ratio"].min())
    xlim_max = max(df["pcrb_improvement_ratio"].max(), df["pf_improvement_ratio"].max())
    diag = np.array([xlim_min, xlim_max])
    ax.plot(diag, diag, "r--", linewidth=1.5, label="y = x (PF = PCRB)")

    valid = df[["pcrb_improvement_ratio", "pf_improvement_ratio"]].dropna()
    r, _ = stats.pearsonr(
        np.log10(valid["pcrb_improvement_ratio"].clip(lower=1e-9)),
        np.log10(valid["pf_improvement_ratio"].clip(lower=1e-9)),
    )
    ax.annotate(
        f"Pearson r² = {r**2:.3f} (log-log)",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PCRB improvement ratio", fontsize=12)
    ax.set_ylabel("PF improvement ratio", fontsize=12)
    ax.set_title("PCRB vs PF Improvement Ratio", fontsize=14)
    ax.legend()

    cbar = fig.colorbar(sc, ax=ax)
    dx_unique = np.sort(df["dx_m"].unique())
    cbar.set_ticks(np.log10(dx_unique))
    cbar.set_ticklabels([str(int(v)) for v in dx_unique])
    cbar.set_label("dx [m]", fontsize=11)

    fig.tight_layout()
    _save(fig, out_dir / "fig4_pcrb_vs_pf.png")


# ---------------------------------------------------------------------------
# Figure 5: Filter health heatmap  (unchanged — already single panel)
# ---------------------------------------------------------------------------

def fig5_filter_health(df: pd.DataFrame, out_dir: Path) -> None:
    beta_vals = sorted(df["beta"].unique())
    dx_vals = sorted(df["dx_m"].unique())

    matrix = np.full((len(beta_vals), len(dx_vals)), np.nan)
    for i, b in enumerate(beta_vals):
        for j, d in enumerate(dx_vals):
            sub = df.loc[(df["beta"] == b) & (df["dx_m"] == d), "pf_n_eff_min"].dropna()
            if len(sub):
                matrix[i, j] = sub.median()

    vmin = min(0.0, np.nanmin(matrix))
    vmax = max(100.0, np.nanmax(matrix))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=50.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(dx_vals)))
    ax.set_xticklabels([str(int(v)) for v in dx_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(beta_vals)))
    ax.set_yticklabels([str(v) for v in beta_vals])
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("beta")
    ax.set_title("Median N_eff_min by beta × dx")

    for i in range(len(beta_vals)):
        for j in range(len(dx_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.04)
    cbar.set_label("Median N_eff_min")
    cbar.ax.axhline(50.0, color="black", linewidth=1.0, linestyle="--")

    fig.tight_layout()
    _save(fig, out_dir / "fig5_filter_health.png")


# ---------------------------------------------------------------------------
# Figure 6: Information index vs PF improvement  (unchanged — already single panel)
# ---------------------------------------------------------------------------

def fig6_info_vs_improvement(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    beta_vals = sorted(df["beta"].unique())
    palette = sns.color_palette("tab10", len(beta_vals))

    for beta, color in zip(beta_vals, palette):
        sub = df[df["beta"] == beta].copy()
        sub = sub[sub["pf_improvement_ratio"] > 0].dropna(
            subset=["information_index", "pf_improvement_ratio"]
        )
        ax.scatter(sub["information_index"], sub["pf_improvement_ratio"],
                   color=color, s=15, alpha=0.5, label=f"beta={beta}", linewidths=0)

        x = sub["information_index"].values
        y = np.log10(sub["pf_improvement_ratio"].values)
        if len(x) >= 2:
            slope, intercept, *_ = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fit, 10 ** (slope * x_fit + intercept), color=color, linewidth=1.8)

    ax.set_yscale("log")
    ax.set_xlabel("Information index", fontsize=12)
    ax.set_ylabel("PF improvement ratio (log scale)", fontsize=12)
    ax.set_title("Information Index vs PF Improvement Ratio", fontsize=14)
    ax.legend(title="beta")

    fig.tight_layout()
    _save(fig, out_dir / "fig6_info_vs_improvement.png")


# ---------------------------------------------------------------------------
# Figure 7: CEP reduction fraction vs input parameters  (4 individual panels)
# ---------------------------------------------------------------------------

def fig7_reduction_vs_params(df: pd.DataFrame, out_dir: Path) -> None:
    col = "pf_cep_reduction_fraction"
    sub = df[df[col].notna() & df[col].between(0, 1)].copy()

    # 7a — vs sigma_init, colored by speed
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    speeds = sorted(sub["speed_m_s"].unique())
    palette = sns.color_palette("magma", len(speeds))
    for speed, c in zip(speeds, palette):
        s = sub[sub["speed_m_s"] == speed]
        ax.scatter(s["sigma_init_m"], s[col], color=c, s=12, alpha=0.5,
                   label=f"{speed:.0f} m/s", linewidths=0)
    ax.set_xlabel("Initial position uncertainty σ₀ [m]")
    ax.set_ylabel("CEP reduction fraction")
    ax.set_title("CEP reduction vs initial uncertainty")
    ax.legend(title="Speed", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig7a_reduction_vs_sigma_init.png")

    # 7b — vs speed, colored by sigma_init
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sigma_vals = sorted(sub["sigma_init_m"].unique())
    norm = mcolors.LogNorm(vmin=min(sigma_vals), vmax=max(sigma_vals))
    cmap = plt.get_cmap("plasma")
    sc = ax.scatter(sub["speed_m_s"], sub[col],
                    c=sub["sigma_init_m"], cmap=cmap, norm=norm,
                    s=12, alpha=0.5, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("σ₀ [m]", fontsize=10)
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("CEP reduction fraction")
    ax.set_title("CEP reduction vs speed")
    fig.tight_layout()
    _save(fig, out_dir / "fig7b_reduction_vs_speed.png")

    # 7c — vs information_index, colored by beta
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    beta_vals = sorted(sub["beta"].unique())
    palette = sns.color_palette("tab10", len(beta_vals))
    for beta, c in zip(beta_vals, palette):
        s = sub[sub["beta"] == beta].dropna(subset=["information_index"])
        ax.scatter(s["information_index"], s[col], color=c, s=12, alpha=0.5,
                   label=f"β={beta}", linewidths=0)
        x = s["information_index"].values
        y = s[col].values
        if len(x) >= 2:
            slope, intercept, *_ = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept, color=c, linewidth=1.5)
    ax.set_xlabel("Information index")
    ax.set_ylabel("CEP reduction fraction")
    ax.set_title("CEP reduction vs map information index")
    ax.legend(title="beta", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "fig7c_reduction_vs_info_index.png")

    # 7d — vs dx_m, median + IQR band
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    dx_vals = np.sort(sub["dx_m"].unique())
    medians, q25, q75 = [], [], []
    for x in dx_vals:
        s = sub.loc[sub["dx_m"] == x, col].dropna()
        medians.append(s.median())
        q25.append(s.quantile(0.25))
        q75.append(s.quantile(0.75))
    ax.plot(dx_vals, np.array(medians), marker="o", linewidth=1.8, label="Median")
    ax.fill_between(dx_vals, np.array(q25), np.array(q75), alpha=0.25, label="IQR")
    ax.set_xscale("log")
    ax.set_xlabel("dx [m/px]")
    ax.set_ylabel("CEP reduction fraction")
    ax.set_title("CEP reduction vs map spatial resolution")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "fig7d_reduction_vs_dx.png")


# ---------------------------------------------------------------------------
# Printed statistics
# ---------------------------------------------------------------------------

def print_statistics(df: pd.DataFrame) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PARTICLE FILTER EXPERIMENT — SUMMARY STATISTICS")
    print(sep)

    print("\n1. OVERALL STATS")
    print("-" * 72)
    cols = [
        "pcrb_improvement_ratio",
        "pf_improvement_ratio",
        "pf_cep_reduction_fraction",
        "pf_vs_pcrb_ratio",
        "pf_n_eff_min",
        "pf_cep_final_m",
    ]
    stat_rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        stat_rows.append({
            "metric": c,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "p5": s.quantile(0.05),
            "p95": s.quantile(0.95),
        })
    print(pd.DataFrame(stat_rows).set_index("metric").to_string(float_format="{:.4g}".format))

    print("\n2. MEDIAN pf_vs_pcrb_ratio BY dx_m (sorted)")
    print("-" * 72)
    by_dx = df.groupby("dx_m")["pf_vs_pcrb_ratio"].median().sort_values()
    best_dx = by_dx.idxmin()
    for dx, val in by_dx.items():
        marker = " <-- BEST" if dx == best_dx else ""
        print(f"  dx={int(dx):>5} m :  {val:.4f}{marker}")

    print("\n3. MEDIAN pf_improvement_ratio BY beta (sorted)")
    print("-" * 72)
    for b, val in df.groupby("beta")["pf_improvement_ratio"].median().sort_values(ascending=False).items():
        print(f"  beta={b:.1f} :  {val:.4f}")

    print("\n4. MEDIAN pf_improvement_ratio BY corr_len_m (sorted)")
    print("-" * 72)
    for c, val in df.groupby("corr_len_m")["pf_improvement_ratio"].median().sort_values(ascending=False).items():
        print(f"  corr_len={int(c):>5} m :  {val:.4f}")

    print("\n5. MEDIAN pf_cep_reduction_fraction BY sigma_init_m")
    print("-" * 72)
    if "pf_cep_reduction_fraction" in df.columns:
        for s, val in df.groupby("sigma_init_m")["pf_cep_reduction_fraction"].median().items():
            print(f"  σ₀={s:>7.0f} m :  {val:.4f}")

    print("\n6. FRACTION OF CONFIGS WITH pf_n_eff_min < 50 (by dx_m)")
    print("-" * 72)
    for dx in sorted(df["dx_m"].unique()):
        sub = df.loc[df["dx_m"] == dx, "pf_n_eff_min"].dropna()
        frac = (sub < 50).mean()
        bar = "#" * int(frac * 40)
        print(f"  dx={int(dx):>5} m :  {frac:.3f}  |{bar:<40}|")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {CSV_PATH} ...")
    df = load_data()
    print(f"  {len(df)} rows loaded (status=ok).")

    print("\nGenerating figures ...")
    fig1_resolution_sweep(df, OUT_DIR)
    fig2_map_params(df, OUT_DIR)
    fig3_trajectory_params(df, OUT_DIR)
    fig4_pcrb_vs_pf(df, OUT_DIR)
    fig5_filter_health(df, OUT_DIR)
    fig6_info_vs_improvement(df, OUT_DIR)
    fig7_reduction_vs_params(df, OUT_DIR)

    print_statistics(df)


if __name__ == "__main__":
    main()
