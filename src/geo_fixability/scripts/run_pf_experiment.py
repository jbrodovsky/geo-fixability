"""
Eight-pointed star trajectory Particle Filter experiment sweep.

Runs both the PCRB recursion and a Bootstrap Particle Filter (SIR) for each
configuration in a reduced parameter grid. Outputs a CSV with PCRB and PF
metrics side-by-side for direct comparison.

Usage
-----
    uv run run-pf-experiment --config sweep_config.yaml   # use pf_verification section
    uv run run-pf-experiment --config sweep_config.yaml --quick
    uv run run-pf-experiment --n-particles 2000 --n-mc 20
    uv run run-pf-experiment --speeds 5 10 --betas 1.0 2.0
    uv run run-pf-experiment --workers 8 --outdir data/pf_experiments
    uv run run-pf-experiment --seed 0
"""

import argparse
import csv
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from geo_fixability.filtering import run_pf_mc
from geo_fixability.mapping import generate_field_spectral
from geo_fixability.metrics import map_information_content
from geo_fixability.navigation import navigation_improvement, pcrb_trajectory
from geo_fixability.trajectory import calibrate_sigma_a, star_trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reduced default parameter grid (used when --config is not supplied)
# ---------------------------------------------------------------------------

_DEFAULT_SPEEDS = [5.0, 10.0, 20.0]          # m/s
_DEFAULT_DURATIONS = [600.0, 1800.0, 3600.0]  # s
_DEFAULT_BETAS = [1.0, 2.0, 3.0]
_DEFAULT_CORR_LENS_M = [500.0, 1500.0, 3500.0]  # m
_DEFAULT_DX = 30.0                            # m/px
_DEFAULT_SIGMA_INITS = [100.0, 500.0, 1000.0]  # m
_DEFAULT_INS_DRIFT_RATES = [1000.0 / 3600.0]  # m/s  (≈ 1 km/hr)
_DEFAULT_SIGMA_MEAS = [0.1]
_DEFAULT_N_PARTICLES = 1000
_DEFAULT_N_MC = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CEP_FACTOR = np.sqrt(2.0 * np.log(2.0))  # ≈ 1.1774; CEP = sigma_per_axis * factor


def _pf_result_record(
    *,
    speed: float,
    duration: float,
    D: float,
    beta: float,
    corr_len: float,
    map_size: int,
    dx: float,
    extent_km: float,
    n_arms: int,
    dt: float,
    sigma_meas: float,
    sigma_v: float,
    sigma_a: float,
    sigma_init: float,
    ins_drift_rate: float,
    seed: int,
    map_metrics: dict,
    traj_n: int,
    traj_duration_actual: float,
    pcrb_metrics: dict,
    pf_mc_result: dict,
    pf_metrics: dict,
    status: str,
) -> dict:
    # PF uncertainty stats at final step from MC mean
    pf_cep_init = pf_metrics.get("cep_initial_m", float("nan"))
    pf_cep_final = pf_metrics.get("cep_final_m", float("nan"))
    pcrb_cep_final = pcrb_metrics.get("cep_final_m", float("nan"))
    pf_cep_reduction = (
        (pf_cep_init - pf_cep_final) / pf_cep_init
        if pf_cep_init and pf_cep_init == pf_cep_init and pf_cep_init > 0
        else float("nan")
    )

    if pcrb_cep_final and pcrb_cep_final == pcrb_cep_final and pcrb_cep_final > 0:
        pf_vs_pcrb = pf_cep_final / pcrb_cep_final
    else:
        pf_vs_pcrb = float("nan")

    pf_pos_unc_std_final = (
        float(pf_mc_result.get("pos_uncertainty_std", np.array([float("nan")]))[-1]) if pf_mc_result else float("nan")
    )

    rec = {
        # --- Input parameters ---
        "speed_m_s": speed,
        "duration_s": duration,
        "leg_length_m": D,
        "total_distance_m": 2 * n_arms * D,
        "beta": beta,
        "corr_len_m": corr_len * dx,
        "extent_km": extent_km,
        "grid_size": map_size,
        "dx_m": dx,
        "n_arms": n_arms,
        "dt_s": dt,
        "sigma_meas": sigma_meas,
        "sigma_v": sigma_v,
        "sigma_a": sigma_a,
        "ins_drift_rate": ins_drift_rate,
        "sigma_init_m": sigma_init,
        "seed": seed,
        # --- Map metrics ---
        "gradient_rms": map_metrics.get("gradient_rms", float("nan")),
        "correlation_length_m": map_metrics.get("correlation_length_m", float("nan")),
        "localization_bound_m": map_metrics.get("localization_bound_m", float("nan")),
        "information_index": map_metrics.get("information_index", float("nan")),
        "spatial_diversity_index": map_metrics.get("spatial_diversity_index", float("nan")),
        "localization_index": map_metrics.get("localization_index", float("nan")),
        # --- Trajectory metadata ---
        "n_trajectory_steps": traj_n,
        "trajectory_duration_actual_s": traj_duration_actual,
        # --- PCRB results ---
        "pcrb_initial_uncertainty_m": pcrb_metrics.get("initial_uncertainty_m", float("nan")),
        "pcrb_final_uncertainty_m": pcrb_metrics.get("final_uncertainty_m", float("nan")),
        "pcrb_improvement_ratio": pcrb_metrics.get("improvement_ratio", float("nan")),
        "pcrb_reduction_fraction": pcrb_metrics.get("reduction_fraction", float("nan")),
        "pcrb_cep_initial_m": pcrb_metrics.get("cep_initial_m", float("nan")),
        "pcrb_cep_final_m": pcrb_cep_final,
        "pcrb_cep_improvement_ratio": pcrb_metrics.get("cep_improvement_ratio", float("nan")),
        # --- PF results ---
        "pf_n_particles": pf_mc_result.get("n_particles", 0) if pf_mc_result else 0,
        "pf_n_mc": pf_mc_result.get("n_mc", 0) if pf_mc_result else 0,
        "pf_cep_initial_m": pf_cep_init,
        "pf_cep_final_m": pf_cep_final,
        "pf_improvement_ratio": pf_metrics.get("improvement_ratio", float("nan")),
        "pf_reduction_fraction": pf_metrics.get("reduction_fraction", float("nan")),
        "pf_cep_improvement_ratio": pf_metrics.get("cep_improvement_ratio", float("nan")),
        "pf_cep_reduction_fraction": pf_cep_reduction,
        "pf_n_eff_mean": pf_mc_result.get("n_eff_mean", float("nan")) if pf_mc_result else float("nan"),
        "pf_n_eff_min": pf_mc_result.get("n_eff_min", float("nan")) if pf_mc_result else float("nan"),
        "pf_pos_unc_std_final_m": pf_pos_unc_std_final,
        "pf_vs_pcrb_ratio": pf_vs_pcrb,
        # --- Status ---
        "status": status,
    }
    return {k: v.item() if hasattr(v, "item") else v for k, v in rec.items()}


# ---------------------------------------------------------------------------
# Per-configuration worker (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _run_single_pf_config(args: tuple) -> tuple[int, dict, str | None]:
    """
    Run one experiment configuration (PCRB + PF) and return (idx, result_record, traceback_or_none).

    All inputs are passed as a single tuple so the function is picklable.
    """
    (
        idx,
        n_total,
        speed,
        duration,
        beta,
        corr_len_m_val,
        dx_val,
        sigma_init,
        extent_km,
        max_D,
        n_arms,
        dt,
        sigma_meas,
        sigma_v,
        sigma_a,
        Q,
        ins_drift_rate,
        n_particles,
        n_mc,
        seed,
    ) = args

    D = speed * duration / (2.0 * n_arms)
    map_size = round(extent_km * 1000 / dx_val)
    corr_len = corr_len_m_val / dx_val

    # Boundary check
    if D >= max_D:
        rec = _pf_result_record(
            speed=speed,
            duration=duration,
            D=D,
            beta=beta,
            corr_len=corr_len,
            map_size=map_size,
            dx=dx_val,
            extent_km=extent_km,
            n_arms=n_arms,
            dt=dt,
            sigma_meas=sigma_meas,
            sigma_v=sigma_v,
            sigma_a=sigma_a,
            sigma_init=sigma_init,
            ins_drift_rate=ins_drift_rate,
            seed=seed,
            map_metrics={},
            traj_n=0,
            traj_duration_actual=0.0,
            pcrb_metrics={},
            pf_mc_result={},
            pf_metrics={},
            status="boundary_violation",
        )
        return idx, rec, None

    P_init = np.eye(2) * (sigma_init**2)

    try:
        map_seed = seed + idx
        np.random.seed(map_seed)
        field = generate_field_spectral(
            (map_size, map_size),
            beta=beta,
            correlation_length=corr_len,
        )

        map_metrics = map_information_content(field, dx=dx_val, sigma_meas=sigma_meas)

        positions, times = star_trajectory(D=D, speed=speed, dt=dt, n_arms=n_arms)
        traj_n = len(positions)
        traj_duration_actual = float(times[-1])

        # PCRB
        P_pcrb = pcrb_trajectory(
            positions,
            field,
            dx=dx_val,
            sigma_meas=sigma_meas,
            P_init=P_init,
            Q=Q,
        )
        pcrb_metrics = navigation_improvement(P_pcrb)
        pcrb_metrics = {k: v for k, v in pcrb_metrics.items() if not isinstance(v, np.ndarray)}

        # Particle Filter (Monte Carlo)
        pf_mc_result = run_pf_mc(
            positions,
            times,
            field,
            dx=dx_val,
            sigma_meas=sigma_meas,
            P_init=P_init,
            Q=Q,
            n_mc=n_mc,
            n_particles=n_particles,
            base_seed=seed + idx * 1000,
        )

        pf_metrics = navigation_improvement(pf_mc_result["P_trajectory_mean"])
        pf_metrics = {k: v for k, v in pf_metrics.items() if not isinstance(v, np.ndarray)}

        rec = _pf_result_record(
            speed=speed,
            duration=duration,
            D=D,
            beta=beta,
            corr_len=corr_len,
            map_size=map_size,
            dx=dx_val,
            extent_km=extent_km,
            n_arms=n_arms,
            dt=dt,
            sigma_meas=sigma_meas,
            sigma_v=sigma_v,
            sigma_a=sigma_a,
            sigma_init=sigma_init,
            ins_drift_rate=ins_drift_rate,
            seed=seed,
            map_metrics=map_metrics,
            traj_n=traj_n,
            traj_duration_actual=traj_duration_actual,
            pcrb_metrics=pcrb_metrics,
            pf_mc_result=pf_mc_result,
            pf_metrics=pf_metrics,
            status="ok",
        )
        return idx, rec, None

    except Exception as exc:  # noqa: BLE001
        tb_str = traceback.format_exc()
        rec = _pf_result_record(
            speed=speed,
            duration=duration,
            D=D,
            beta=beta,
            corr_len=corr_len,
            map_size=map_size,
            dx=dx_val,
            extent_km=extent_km,
            n_arms=n_arms,
            dt=dt,
            sigma_meas=sigma_meas,
            sigma_v=sigma_v,
            sigma_a=sigma_a,
            sigma_init=sigma_init,
            ins_drift_rate=ins_drift_rate,
            seed=seed,
            map_metrics={},
            traj_n=0,
            traj_duration_actual=0.0,
            pcrb_metrics={},
            pf_mc_result={},
            pf_metrics={},
            status=f"error: {exc}",
        )
        return idx, rec, tb_str


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def run_pf_experiment(
    speeds: list[float] | None = None,
    durations: list[float] | None = None,
    betas: list[float] | None = None,
    corr_lens_m: list[float] | None = None,
    dx_values: list[float] | None = None,
    initial_uncertainties: list[float] | None = None,
    ins_drift_rates: list[float] | None = None,
    sigma_meas_values: list[float] | None = None,
    n_particles: int = _DEFAULT_N_PARTICLES,
    n_mc: int = _DEFAULT_N_MC,
    extent_km: float = 10.0,
    dx: float = _DEFAULT_DX,
    n_arms: int = 8,
    dt: float = 1.0,
    sigma_v: float = 0.0,
    calibration_eval_time: float = 3600.0,
    seed: int = 42,
    outdir: str = "data/experiments",
    save: bool = True,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Sweep trajectory and map parameters, computing both PCRB and Particle Filter
    navigation improvement metrics for direct comparison.

    Parameters
    ----------
    speeds : list of float
        Vehicle speeds [m/s].
    durations : list of float
        Total trajectory durations [s].
    betas : list of float
        PSD exponents (map spectral slope).
    corr_lens_m : list of float
        Correlation length roll-off scale [m].
    dx_values : list of float or None
        Pixel sizes to sweep [m/px]. If None, uses the single ``dx`` value.
    initial_uncertainties : list of float or None
        Initial position uncertainty 1-sigma [m].
    ins_drift_rates : list of float or None
        INS drift rates [m/s] to sweep. One calibration is pre-computed per
        unique value. Default: [1000/3600] (≈ 1 km/hr).
    sigma_meas_values : list of float or None
        Measurement noise std values [field_units] to sweep. Default: [0.1].
    n_particles : int
        Particles per filter run.
    n_mc : int
        Independent PF runs per configuration (for MC statistics).
    extent_km : float
        Physical map side length [km].
    dx : float
        Fallback pixel size [m/px] when dx_values is None.
    n_arms : int
        Star arm count.
    dt : float
        Time step [s].
    sigma_v : float
        White velocity noise [m/s].
    calibration_eval_time : float
        Time horizon used for INS calibration binary search [s].
    seed : int
        Base random seed.
    outdir : str
        Output directory for CSV results.
    save : bool
        Save results to CSV.
    n_workers : int or None
        Number of parallel worker processes. None uses ``os.cpu_count()``.

    Returns
    -------
    list of dict
        One result record per swept combination, containing both PCRB and PF
        metrics.
    """
    speeds = speeds or _DEFAULT_SPEEDS
    durations = durations or _DEFAULT_DURATIONS
    betas = betas or _DEFAULT_BETAS
    corr_lens_m = corr_lens_m or _DEFAULT_CORR_LENS_M
    dx_values = dx_values or [dx]
    initial_uncertainties = initial_uncertainties or _DEFAULT_SIGMA_INITS
    ins_drift_rates = ins_drift_rates or _DEFAULT_INS_DRIFT_RATES
    sigma_meas_values = sigma_meas_values or _DEFAULT_SIGMA_MEAS
    workers = n_workers or os.cpu_count() or 1

    max_D = extent_km * 1000 / 2.0

    logger.info("Map extent: %g km × %g km", extent_km, extent_km)
    logger.info("dx: %s m/px  →  grids: %s", dx_values, [round(extent_km * 1000 / d) for d in dx_values])
    logger.info("Particles per run: %d,  MC runs per config: %d", n_particles, n_mc)

    # Pre-compute INS calibration for each unique drift rate.
    # This avoids redundant binary searches (one per unique rate).
    logger.info("Pre-computing INS calibration for %d drift rate(s)...", len(ins_drift_rates))
    calibration_cache: dict[float, tuple[float, np.ndarray]] = {}
    for drift_rate in ins_drift_rates:
        sigma_a, Q = calibrate_sigma_a(
            drift_rate,
            eval_time=calibration_eval_time,
            dt=dt,
            n_mc=200,
            seed=seed,
        )
        calibration_cache[drift_rate] = (sigma_a, Q)
        logger.info(
            "  drift_rate=%.5f m/s (%.2f m/hr)  sigma_a=%.6f  Q_diag=%.3e",
            drift_rate, drift_rate * 3600, sigma_a, Q[0, 0],
        )

    configs = list(product(
        speeds, durations, betas, corr_lens_m, dx_values,
        initial_uncertainties, ins_drift_rates, sigma_meas_values,
    ))
    n_total = len(configs)
    logger.info(
        "Running %d configurations (%d speeds × %d durations × %d betas × %d corr_lens "
        "× %d dx × %d sigma_inits × %d drift_rates × %d sigma_meas) on %d workers...",
        n_total, len(speeds), len(durations), len(betas), len(corr_lens_m),
        len(dx_values), len(initial_uncertainties), len(ins_drift_rates),
        len(sigma_meas_values), workers,
    )

    work_args = []
    for idx, (speed, duration, beta, corr_len_m_val, dx_val, sigma_init,
              drift_rate, sigma_meas) in enumerate(configs):
        sigma_a, Q = calibration_cache[drift_rate]
        work_args.append((
            idx, n_total, speed, duration, beta, corr_len_m_val, dx_val, sigma_init,
            extent_km, max_D, n_arms, dt, sigma_meas, sigma_v,
            sigma_a, Q, drift_rate, n_particles, n_mc, seed,
        ))

    results_by_idx: dict[int, dict] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_pf_config, a): a[0] for a in work_args}
        for future in as_completed(futures):
            idx, rec, tb = future.result()
            results_by_idx[idx] = rec
            completed += 1

            status = rec["status"]
            if status == "boundary_violation":
                logger.info(
                    "[%4d/%d] SKIP  v=%4.0f T=%4.0fs β=%s cl=%5.0fm σ₀=%5.0fm  D=%.0fm >= max %.0fm",
                    completed, n_total,
                    rec["speed_m_s"], rec["duration_s"], rec["beta"],
                    rec["corr_len_m"], rec["sigma_init_m"],
                    rec["leg_length_m"], max_D,
                )
            elif status.startswith("error"):
                logger.error(
                    "[%4d/%d] ERROR v=%4.0f T=%4.0fs β=%s cl=%5.0fm σ₀=%5.0fm  %s",
                    completed, n_total,
                    rec["speed_m_s"], rec["duration_s"], rec["beta"],
                    rec["corr_len_m"], rec["sigma_init_m"], status,
                )
                if tb:
                    logger.debug("Config %d traceback:\n%s", idx, tb)
            else:
                logger.info(
                    "[%4d/%d] v=%4.0f T=%4.0fs D=%6.0fm  β=%s cl=%5.0fm σ₀=%5.0fm  "
                    "pcrb=%5.2fx  pf=%5.2fx  pf/pcrb=%.2f",
                    completed, n_total,
                    rec["speed_m_s"], rec["duration_s"], rec["leg_length_m"],
                    rec["beta"], rec["corr_len_m"], rec["sigma_init_m"],
                    rec["pcrb_improvement_ratio"], rec["pf_improvement_ratio"],
                    rec["pf_vs_pcrb_ratio"],
                )

    results = [results_by_idx[i] for i in range(n_total)]

    # --- Summary table ---
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        header = (
            f"{'speed':>6}  {'dur':>6}  {'beta':>5}  {'cl_m':>6}  "
            f"{'σ₀(m)':>7}  {'D(m)':>6}  "
            f"{'pcrb_ratio':>10}  {'pf_ratio':>8}  {'pf/pcrb':>7}  "
            f"{'pf_neff_min':>11}"
        )
        logger.info("--- Summary (PCRB vs PF improvement ratio) ---")
        logger.info(header)
        logger.info("-" * 95)
        for r in ok:
            logger.info(
                "%6.1f  %6.0f  %5.1f  %6.0f  %7.0f  %6.0f  %10.2fx  %8.2fx  %7.2f  %11.0f",
                r["speed_m_s"], r["duration_s"], r["beta"], r["corr_len_m"],
                r["sigma_init_m"], r["leg_length_m"],
                r["pcrb_improvement_ratio"], r["pf_improvement_ratio"],
                r["pf_vs_pcrb_ratio"], r["pf_n_eff_min"],
            )

    # --- Save ---
    if save:
        out_path = Path(outdir) / "pf_experiment_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Results saved to %s  (%d records)", out_path, len(results))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_pf_config(config_path: str, quick: bool) -> dict:
    """Load pf_verification section from a YAML config file."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pf_sec = cfg.get("pf_verification", {})
    sweep = {k: [float(v) for v in vals] for k, vals in pf_sec.get("sweep", {}).items()}
    fixed = pf_sec.get("fixed", {})
    if quick:
        sweep = {k: v[:2] for k, v in sweep.items()}
    return {"sweep": sweep, "fixed": fixed}


def main() -> None:
    """CLI entry point for the star trajectory PF experiment sweep."""
    parser = argparse.ArgumentParser(
        description="Star trajectory Particle Filter vs PCRB experiment sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML config file containing a 'pf_verification' section. "
            "When provided, sweep parameters are read from that section and CLI "
            "dimension flags are ignored."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Restrict each sweep dimension to its first 2 values (config mode only).",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=None,
        metavar="V",
        help="Vehicle speeds [m/s].",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=None,
        metavar="T",
        help="Total trajectory durations [s].",
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=None,
        metavar="B",
        help="PSD exponents (map spectral slope).",
    )
    parser.add_argument(
        "--corr-lens-m",
        nargs="+",
        type=float,
        default=None,
        dest="corr_lens_m",
        metavar="CL",
        help="Correlation length roll-off scale [m].",
    )
    parser.add_argument(
        "--sigma-inits",
        nargs="+",
        type=float,
        default=None,
        dest="sigma_inits",
        metavar="S",
        help="Initial position uncertainty 1-sigma [m].",
    )
    parser.add_argument(
        "--ins-drift-rates",
        nargs="+",
        type=float,
        default=None,
        dest="ins_drift_rates",
        metavar="R",
        help="INS drift rates [m/s] to sweep. Example: --ins-drift-rates 0.02778 0.2778",
    )
    parser.add_argument(
        "--sigma-meas-values",
        nargs="+",
        type=float,
        default=None,
        dest="sigma_meas_values",
        metavar="S",
        help="Measurement noise std values [field_units] to sweep.",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=_DEFAULT_N_PARTICLES,
        metavar="N",
        help="Particles per filter run.",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=_DEFAULT_N_MC,
        metavar="N",
        help="Independent PF runs per configuration (Monte Carlo statistics).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=_DEFAULT_DX,
        metavar="M",
        help="Pixel size [m/px] used when --dx-values is not set.",
    )
    parser.add_argument(
        "--dx-values",
        nargs="+",
        type=float,
        default=None,
        dest="dx_values",
        metavar="M",
        help="Pixel sizes to sweep [m/px]. Overrides --dx when set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/experiments",
        help="Output directory for CSV results.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes. Default: os.cpu_count().",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        dest="log_file",
        metavar="PATH",
        help=(
            "Path for the log file. Default: <outdir>/pf_experiment_<timestamp>.log. "
            "Pass 'none' to disable file logging."
        ),
    )

    args = parser.parse_args()

    # Configure logging (console + optional file)
    log_file = args.log_file
    if log_file is None and not args.no_save:
        log_dir = Path(args.outdir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir / f"pf_experiment_{timestamp}.log")
    elif log_file and log_file.lower() == "none":
        log_file = None

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    if log_file:
        logger.info("Log file: %s", log_file)

    if args.config is not None:
        logger.info("Config: %s%s", args.config, " (quick)" if args.quick else "")
        pf_cfg = _load_pf_config(args.config, args.quick)
        sweep = pf_cfg["sweep"]
        fixed = pf_cfg["fixed"]
        speeds = sweep.get("speed")
        durations = sweep.get("duration")
        betas = sweep.get("beta")
        corr_lens_m = sweep.get("correlation_length")
        dx_values = sweep.get("dx")
        sigma_inits = sweep.get("sigma_init")
        ins_drift_rates = sweep.get("ins_drift_rate")
        sigma_meas_values = sweep.get("sigma_meas")
        n_particles = fixed.get("n_particles", args.n_particles)
        n_mc = fixed.get("n_mc", args.n_mc)
        n_workers = fixed.get("n_workers", args.workers)
    else:
        speeds = args.speeds
        durations = args.durations
        betas = args.betas
        corr_lens_m = args.corr_lens_m
        dx_values = args.dx_values
        sigma_inits = args.sigma_inits
        ins_drift_rates = args.ins_drift_rates
        sigma_meas_values = args.sigma_meas_values
        n_particles = args.n_particles
        n_mc = args.n_mc
        n_workers = args.workers

    run_pf_experiment(
        speeds=speeds,
        durations=durations,
        betas=betas,
        corr_lens_m=corr_lens_m,
        initial_uncertainties=sigma_inits,
        ins_drift_rates=ins_drift_rates,
        sigma_meas_values=sigma_meas_values,
        n_particles=n_particles,
        n_mc=n_mc,
        dx=args.dx,
        dx_values=dx_values,
        seed=args.seed,
        outdir=args.outdir,
        save=not args.no_save,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
