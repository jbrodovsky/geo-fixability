"""
plot_q_validation.py
====================
Validate the fixed-Q process-noise approximation against the true
σ_a·t^{3/2} INS drift model for the 2400 s runs in the parameter sweep.

Two figures (saved separately for LaTeX inclusion):
  fig_q_validation_drift     – INS drift from rest: MC empirical vs analytic
                               t^{3/2} vs fixed-Q model accumulated over time.
  fig_q_validation_unaided   – Total unaided position σ entering the aiding
                               interval, fixed-Q PCRB prior vs correct INS
                               formula, for each σ_init in SIGMA_INITS.

Vertical dashed lines mark the sweep durations.

Usage
-----
    uv run python src/geo_fixability/scripts/plot_q_validation.py

Output
------
    paper/figures/q-val/fig_q_validation_drift.{pdf,png}
    paper/figures/q-val/fig_q_validation_unaided.{pdf,png}
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

from geo_fixability.navigation import run_mc_ins
from geo_fixability.trajectory import calibrate_sigma_a

# ---------------------------------------------------------------------------
# Parameters — must match run_pf_experiment.py defaults
# ---------------------------------------------------------------------------
TARGET_DRIFT_RATE = 1000.0 / 3600.0  # 1 km/h in m/s
CALIB_EVAL_TIME = 3600.0  # s
DT = 1.0  # s
T_MAX = 3600.0  # longest sweep duration
N_MC = 500  # MC runs for empirical curve
SIGMA_INITS = [100.0, 250.0, 500.0, 750.0, 1000.0]  # m — sweep values
SWEEP_DURATIONS = [300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0, 2700.0, 3600.0]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "paper" / "figures" / "q-val"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _analytic_ins_sigma(t: np.ndarray, sigma_a: float) -> np.ndarray:
    """Total 2-D INS σ (sqrt of trace P) for velocity random walk from rest."""
    return sigma_a * np.sqrt(2.0 * t**3 / 3.0)


def _fixed_q_sigma(t: np.ndarray, sigma_a: float, sigma_init: float) -> np.ndarray:
    """Total 2-D unaided σ under fixed-Q model: √(2·(σ_init² + t·Q_diag))."""
    q_diag = sigma_a**2 * DT
    return np.sqrt(2.0 * (sigma_init**2 + t * q_diag))


def _correct_unaided_sigma(t: np.ndarray, sigma_a: float, sigma_init: float) -> np.ndarray:
    """Total 2-D unaided σ using the true INS t^{3/2} growth on top of P_init."""
    return np.sqrt(2.0 * (sigma_init**2 + sigma_a**2 * t**3 / 3.0))


def _sigma_colors(n: int) -> list:
    """Return n perceptually-uniform colors for the sigma_init curves."""
    cmap = plt.get_cmap("plasma")
    return [cmap(i / (n - 1)) for i in range(n)]


def _add_duration_lines(ax: plt.Axes) -> None:
    """Draw vertical dashed lines for each sweep duration."""
    for T_mark in SWEEP_DURATIONS:
        ax.axvline(T_mark, color="0.5", lw=0.8, ls="--")
        ax.text(
            T_mark + 25, 0.97, f"{T_mark:.0f} s",
            transform=ax.get_xaxis_transform(), fontsize=7, color="0.4", va="top",
        )


# ---------------------------------------------------------------------------
# Figure A: INS drift from rest
# ---------------------------------------------------------------------------


def _save_drift_figure(t: np.ndarray, mc_rmse: np.ndarray, mc_p16: np.ndarray,
                       mc_p84: np.ndarray, sigma_a: float) -> None:
    colors = {"mc": "#2166ac", "analytic": "#d6604d", "fixedq": "#4dac26"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    ax.fill_between(t, mc_p16, mc_p84, alpha=0.25, color=colors["mc"], label=None)
    ax.plot(t, mc_rmse, color=colors["mc"], lw=1.8,
            label=r"MC INS RMSE (16th–84th percentile band)")
    ax.plot(t, _analytic_ins_sigma(t, sigma_a),
            color=colors["analytic"], lw=2.0, ls="--",
            label=r"Analytic $\sigma_a \sqrt{2t^3/3}$")
    ax.plot(t, np.sqrt(2.0 * t * sigma_a**2 * DT),
            color=colors["fixedq"], lw=2.0, ls=":",
            label=r"Fixed-$Q$ accumulated: $\sqrt{2tQ_\mathrm{diag}}$")

    _add_duration_lines(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total 2-D position error σ [m]")
    ax.set_title("INS drift from rest")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_q_validation_drift.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: Unaided σ entering aiding interval
# ---------------------------------------------------------------------------


def _save_unaided_figure(t: np.ndarray, sigma_a: float) -> None:
    colors = _sigma_colors(len(SIGMA_INITS))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    for sigma_init, col in zip(SIGMA_INITS, colors):
        fq_curve = _fixed_q_sigma(t, sigma_a, sigma_init)
        cor_curve = _correct_unaided_sigma(t, sigma_a, sigma_init)

        ax.plot(t, fq_curve, color=col, lw=1.8, ls="-",
                label=rf"Fixed-$Q$, $\sigma_0={sigma_init:.0f}$ m")
        ax.plot(t, cor_curve, color=col, lw=1.8, ls="--",
                label=rf"Correct, $\sigma_0={sigma_init:.0f}$ m")

        fq_end = _fixed_q_sigma(T_MAX, sigma_a, sigma_init)
        cor_end = _correct_unaided_sigma(T_MAX, sigma_a, sigma_init)
        ax.annotate(
            "",
            xy=(T_MAX, cor_end), xytext=(T_MAX, fq_end),
            arrowprops=dict(arrowstyle="<->", color=col, lw=1.2),
        )
        mid = 0.5 * (fq_end + cor_end)
        pct = (cor_end / fq_end - 1.0) * 100.0
        ax.text(T_MAX - 80, mid, f"+{pct:.0f}%", fontsize=7.5, color=col,
                ha="right", va="center")

    _add_duration_lines(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total 2-D unaided σ [m]")
    ax.set_title(r"Unaided $\sigma$ entering aiding interval")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_q_validation_unaided.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Calibrating σ_a (same parameters as experiment sweep)...")
    sigma_a, Q = calibrate_sigma_a(
        TARGET_DRIFT_RATE,
        eval_time=CALIB_EVAL_TIME,
        dt=DT,
        n_mc=200,
        seed=42,
    )
    q_diag = Q[0, 0]
    print(f"  σ_a = {sigma_a:.6f} m/s^(3/2)   Q_diag = {q_diag:.3e} m²")

    N = int(T_MAX / DT) + 1
    positions_truth = np.zeros((N, 2))
    times = np.arange(N) * DT

    print(f"Running {N_MC} MC INS runs over [0, {T_MAX:.0f} s]...")
    mc = run_mc_ins(positions_truth, times, sigma_v=0.0, sigma_a=sigma_a, n_runs=N_MC, seed=0)

    per_run_total = np.sqrt(mc["errors_all"][:, :, 0] ** 2 + mc["errors_all"][:, :, 1] ** 2)
    mc_rmse = mc["rmse_total"]
    mc_p16 = np.percentile(per_run_total, 16, axis=0)
    mc_p84 = np.percentile(per_run_total, 84, axis=0)

    t = times

    # Summary numbers for the paper
    sigma_analytic = sigma_a * math.sqrt(2.0 * T_MAX**3 / 3.0)
    sigma_fixedq = math.sqrt(2.0 * T_MAX * q_diag)
    print(f"\nAt t = {T_MAX:.0f} s (from rest):")
    print(f"  Analytic t^3/2 σ (total 2D): {sigma_analytic:.1f} m")
    print(f"  Fixed-Q accumulated σ (total 2D): {sigma_fixedq:.3f} m")
    print(f"  MC empirical RMSE:            {mc_rmse[int(T_MAX / DT)]:.1f} m")

    for sigma_init in SIGMA_INITS:
        fq = _fixed_q_sigma(T_MAX, sigma_a, sigma_init)
        cor = _correct_unaided_sigma(T_MAX, sigma_a, sigma_init)
        print(f"\nσ_init = {sigma_init:.0f} m, t = {T_MAX:.0f} s:")
        print(f"  Fixed-Q unaided σ (total 2D): {fq:.1f} m")
        print(f"  Correct unaided σ (total 2D): {cor:.1f} m")
        print(f"  Relative overestimate of correct / fixed-Q: {cor / fq - 1:.1%}")

    print("\nGenerating figures...")
    _save_drift_figure(t, mc_rmse, mc_p16, mc_p84, sigma_a)
    _save_unaided_figure(t, sigma_a)


if __name__ == "__main__":
    main()
