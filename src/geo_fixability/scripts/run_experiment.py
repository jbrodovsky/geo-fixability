"""
Eight-pointed star trajectory PCRB experiment sweep.

Sweeps over trajectory parameters (speed, duration), map spectral parameters
(beta, correlation_length_m, dx), and initial position uncertainty, computing
the PCRB navigation improvement for each combination. Results are saved as CSV.

Usage
-----
    uv run run-star-experiment                                    # full default sweep
    uv run run-star-experiment --no-mc                            # PCRB only, skip MC INS
    uv run run-star-experiment --speeds 5 10 --durations 600 2400
    uv run run-star-experiment --betas 1.3 2.0 --corr-lens-m 300 2500
    uv run run-star-experiment --dx-values 10 25 50 100 250 500  # resolution sweep
    uv run run-star-experiment --initial-uncertainties 100 500 1000 2000
    uv run run-star-experiment --seed 0 --outdir data/experiments
    uv run run-star-experiment --workers 8                        # explicit worker count

Resolution sweep (quad-tree divisors of 10,000 m):
    --dx-values 5 10 25 50 100 200 250 500 1000 2500
    NOTE: dx < 10 m produces grids > 1 M pixels and may be slow.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

from geo_fixability.mapping import generate_field_spectral
from geo_fixability.metrics import map_information_content
from geo_fixability.navigation import navigation_improvement, pcrb_trajectory, run_mc_ins
from geo_fixability.trajectory import calibrate_sigma_a, star_trajectory


# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

_DEFAULT_SPEEDS          = [2.0, 5.0, 10.0, 20.0]           # m/s
_DEFAULT_DURATIONS       = [300.0, 600.0, 1200.0, 2400.0]    # s
_DEFAULT_BETAS           = [0.7, 1.3, 2.0, 3.5]
_DEFAULT_CORR_LENS_M     = [300.0, 1000.0, 2500.0, 4500.0]   # m
_DEFAULT_DX_VALUES       = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0]  # m/px
_DEFAULT_SIGMA_INITS     = [100.0, 500.0, 1000.0, 2000.0]    # m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_record(
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
    seed: int,
    map_metrics: dict,
    traj_n: int,
    traj_duration_actual: float,
    pcrb_metrics: dict,
    mc_rmse: float | None,
    mc_n_runs: int,
    status: str,
) -> dict:
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
        "pcrb_cep_final_m": pcrb_metrics.get("cep_final_m", float("nan")),
        "pcrb_cep_improvement_ratio": pcrb_metrics.get("cep_improvement_ratio", float("nan")),
        # --- MC INS results ---
        "mc_final_rmse_m": mc_rmse if mc_rmse is not None else float("nan"),
        "mc_n_runs": mc_n_runs,
        # --- Status ---
        "status": status,
    }
    return {k: v.item() if hasattr(v, "item") else v for k, v in rec.items()}


# ---------------------------------------------------------------------------
# Per-configuration worker (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _run_single_config(args: tuple) -> tuple[int, dict]:
    """
    Run one experiment configuration and return (idx, result_record).

    All inputs are passed as a single tuple so the function can be used with
    ``ProcessPoolExecutor.submit`` without closure state.
    """
    (
        idx, n_total,
        speed, duration, beta, corr_len_m_val, dx_val, sigma_init,
        extent_km, max_D, n_arms, dt,
        sigma_meas, sigma_v, sigma_a, Q,
        run_mc, n_mc,
        seed,
    ) = args

    D = speed * duration / (2.0 * n_arms)
    map_size = round(extent_km * 1000 / dx_val)
    corr_len = corr_len_m_val / dx_val

    # Boundary check
    if D >= max_D:
        rec = _result_record(
            speed=speed, duration=duration, D=D, beta=beta, corr_len=corr_len,
            map_size=map_size, dx=dx_val, extent_km=extent_km,
            n_arms=n_arms, dt=dt,
            sigma_meas=sigma_meas, sigma_v=sigma_v, sigma_a=sigma_a,
            sigma_init=sigma_init, seed=seed,
            map_metrics={}, traj_n=0, traj_duration_actual=0.0,
            pcrb_metrics={}, mc_rmse=None, mc_n_runs=0,
            status="boundary_violation",
        )
        return idx, rec

    P_init = np.eye(2) * (sigma_init ** 2)

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

        P_traj = pcrb_trajectory(
            positions, field, dx=dx_val,
            sigma_meas=sigma_meas,
            P_init=P_init,
            Q=Q,
        )
        pcrb_metrics = navigation_improvement(P_traj)
        pcrb_metrics = {k: v for k, v in pcrb_metrics.items()
                        if not isinstance(v, np.ndarray)}

        mc_rmse = None
        mc_n_runs_actual = 0
        if run_mc:
            mc_res = run_mc_ins(
                positions, times,
                sigma_v=sigma_v,
                sigma_a=sigma_a,
                n_runs=n_mc,
                seed=seed,
            )
            mc_rmse = float(mc_res["rmse_total"][-1])
            mc_n_runs_actual = n_mc

        rec = _result_record(
            speed=speed, duration=duration, D=D, beta=beta, corr_len=corr_len,
            map_size=map_size, dx=dx_val, extent_km=extent_km,
            n_arms=n_arms, dt=dt,
            sigma_meas=sigma_meas, sigma_v=sigma_v, sigma_a=sigma_a,
            sigma_init=sigma_init, seed=seed,
            map_metrics=map_metrics, traj_n=traj_n,
            traj_duration_actual=traj_duration_actual,
            pcrb_metrics=pcrb_metrics,
            mc_rmse=mc_rmse, mc_n_runs=mc_n_runs_actual,
            status="ok",
        )

    except Exception as exc:  # noqa: BLE001
        rec = _result_record(
            speed=speed, duration=duration, D=D, beta=beta, corr_len=corr_len,
            map_size=map_size, dx=dx_val, extent_km=extent_km,
            n_arms=n_arms, dt=dt,
            sigma_meas=sigma_meas, sigma_v=sigma_v, sigma_a=sigma_a,
            sigma_init=sigma_init, seed=seed,
            map_metrics={}, traj_n=0, traj_duration_actual=0.0,
            pcrb_metrics={}, mc_rmse=None, mc_n_runs=0,
            status=f"error: {exc}",
        )

    return idx, rec


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    speeds: list[float] | None = None,
    durations: list[float] | None = None,
    betas: list[float] | None = None,
    corr_lens_m: list[float] | None = None,
    dx_values: list[float] | None = None,
    initial_uncertainties: list[float] | None = None,
    extent_km: float = 10.0,
    dx: float = 30.0,
    n_arms: int = 8,
    dt: float = 1.0,
    sigma_meas: float = 0.1,
    sigma_v: float = 0.0,
    target_drift_rate: float = 1000.0 / 3600.0,
    calibration_eval_time: float = 3600.0,
    n_mc: int = 50,
    run_mc: bool = True,
    seed: int = 42,
    outdir: str = "data/experiments",
    save: bool = True,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Sweep trajectory, map, and initial uncertainty parameters, computing PCRB
    improvement metrics.

    Parameters
    ----------
    speeds : list of float
        Vehicle speeds [m/s]. Default [2, 5, 10, 20].
    durations : list of float
        Total trajectory durations [s]. Default [300, 600, 1200, 2400].
    betas : list of float
        PSD exponents (map spectral slope). Default [0.7, 1.3, 2.0, 3.5].
    corr_lens_m : list of float
        Correlation length roll-off scale [m]. Default [300, 1000, 2500, 4500].
    dx_values : list of float or None
        Pixel sizes to sweep [m/px]. If None, uses the single ``dx`` value.
    initial_uncertainties : list of float or None
        Initial position uncertainty 1-sigma values [m].
        Default [100, 500, 1000, 2000].
        P_init = diag(sigma^2, sigma^2) for each value.
    extent_km : float
        Physical map side length [km]. Default 10.0.
    dx : float
        Fallback pixel size [m/px] used when dx_values is None. Default 30.0.
    n_arms : int
        Star arm count. Default 8.
    dt : float
        Time step [s]. Default 1.0.
    sigma_meas : float
        Measurement noise std [field_units]. Default 0.1.
    sigma_v : float
        White velocity noise [m/s]. Default 0.0.
    target_drift_rate : float
        IMU calibration drift rate [m/s]. Default 1000/3600 (1 km/hour).
    calibration_eval_time : float
        Calibration time horizon [s]. Default 3600.
    n_mc : int
        MC runs per configuration. Default 50.
    run_mc : bool
        Run open-loop MC INS simulation. Default True.
    seed : int
        Base random seed. Default 42.
    outdir : str
        Output directory for CSV results. Default "data/experiments".
    save : bool
        Save results to CSV. Default True.
    n_workers : int or None
        Number of parallel worker processes. None uses ``os.cpu_count()``.

    Returns
    -------
    list of dict
        One result record per swept combination.
    """
    speeds = speeds or _DEFAULT_SPEEDS
    durations = durations or _DEFAULT_DURATIONS
    betas = betas or _DEFAULT_BETAS
    corr_lens_m = corr_lens_m or _DEFAULT_CORR_LENS_M
    dx_values = dx_values or _DEFAULT_DX_VALUES
    initial_uncertainties = initial_uncertainties or _DEFAULT_SIGMA_INITS
    workers = n_workers or os.cpu_count() or 1

    max_D = extent_km * 1000 / 2.0

    print(f"Map extent: {extent_km} km × {extent_km} km")
    print(f"dx sweep: {dx_values} m/px  "
          f"→  grids: {[round(extent_km * 1000 / d) for d in dx_values]}")
    if any(d < 10.0 for d in dx_values):
        print("WARNING: dx < 10 m produces grids > 1 M pixels and may be slow.")
    print(f"Initial uncertainties: {initial_uncertainties} m")
    print(f"Calibrating IMU noise (target drift: {target_drift_rate * 3.6:.3f} km/h "
          f"over {calibration_eval_time:.0f} s)...")
    sigma_a, Q = calibrate_sigma_a(
        target_drift_rate,
        eval_time=calibration_eval_time,
        dt=dt,
        n_mc=200,
        seed=seed,
    )
    print(f"  sigma_a = {sigma_a:.6f} m/s^(3/2)   Q_diag = {Q[0, 0]:.3e} m^2")

    configs = list(product(
        speeds, durations, betas, corr_lens_m, dx_values, initial_uncertainties
    ))
    n_total = len(configs)
    print(f"\nRunning {n_total} configurations "
          f"({len(speeds)} speeds × {len(durations)} durations × "
          f"{len(betas)} betas × {len(corr_lens_m)} corr_lens × "
          f"{len(dx_values)} dx_values × "
          f"{len(initial_uncertainties)} init_uncertainties) "
          f"on {workers} workers...\n")

    # Build argument tuples for workers
    work_args = [
        (
            idx, n_total,
            speed, duration, beta, corr_len_m_val, dx_val, sigma_init,
            extent_km, max_D, n_arms, dt,
            sigma_meas, sigma_v, sigma_a, Q,
            run_mc, n_mc,
            seed,
        )
        for idx, (speed, duration, beta, corr_len_m_val, dx_val, sigma_init)
        in enumerate(configs)
    ]

    # Collect results indexed by original config order
    results_by_idx: dict[int, dict] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_config, a): a[0] for a in work_args}
        for future in as_completed(futures):
            idx, rec = future.result()
            results_by_idx[idx] = rec
            completed += 1

            status = rec["status"]
            if status == "boundary_violation":
                print(f"[{completed:4d}/{n_total}] SKIP  "
                      f"v={rec['speed_m_s']:4.0f} T={rec['duration_s']:4.0f}s "
                      f"β={rec['beta']} cl={rec['corr_len_m']:5.0f}m "
                      f"dx={rec['dx_m']:6.1f} σ₀={rec['sigma_init_m']:5.0f}m  "
                      f"D={rec['leg_length_m']:.0f}m >= max {max_D:.0f}m")
            elif status.startswith("error"):
                print(f"[{completed:4d}/{n_total}] ERROR "
                      f"v={rec['speed_m_s']:4.0f} T={rec['duration_s']:4.0f}s "
                      f"β={rec['beta']} cl={rec['corr_len_m']:5.0f}m "
                      f"dx={rec['dx_m']:6.1f} σ₀={rec['sigma_init_m']:5.0f}m  "
                      f"{status}")
            else:
                ratio = rec["pcrb_improvement_ratio"]
                rmse_str = (f"  MC_rmse={rec['mc_final_rmse_m']:.1f}m"
                            if rec["mc_final_rmse_m"] == rec["mc_final_rmse_m"]  # not nan
                            else "")
                print(f"[{completed:4d}/{n_total}] "
                      f"v={rec['speed_m_s']:4.0f} T={rec['duration_s']:4.0f}s "
                      f"D={rec['leg_length_m']:6.0f}m  "
                      f"β={rec['beta']} cl={rec['corr_len_m']:5.0f}m "
                      f"dx={rec['dx_m']:6.1f} σ₀={rec['sigma_init_m']:5.0f}m  "
                      f"ratio={ratio:5.2f}x{rmse_str}")

    # Restore insertion order
    results = [results_by_idx[i] for i in range(n_total)]

    # --- Summary table ---
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        print("\n--- Summary (PCRB improvement ratio) ---")
        print(f"{'speed':>6}  {'dur':>6}  {'beta':>5}  {'cl_m':>6}  "
              f"{'dx_m':>6}  {'grid':>5}  {'σ₀(m)':>7}  {'D(m)':>6}  "
              f"{'ratio':>7}  {'pcrb_final(m)':>14}  {'grad_rms':>9}")
        print("-" * 100)
        for r in ok:
            print(f"{r['speed_m_s']:6.1f}  {r['duration_s']:6.0f}  "
                  f"{r['beta']:5.1f}  {r['corr_len_m']:6.0f}  "
                  f"{r['dx_m']:6.1f}  {r['grid_size']:5d}  "
                  f"{r['sigma_init_m']:7.0f}  "
                  f"{r['leg_length_m']:6.0f}  "
                  f"{r['pcrb_improvement_ratio']:7.2f}x  "
                  f"{r['pcrb_final_uncertainty_m']:14.2f}  "
                  f"{r['gradient_rms']:9.4f}")

    # --- Save ---
    if save:
        out_path = Path(outdir) / "star_trajectory_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path}  ({len(results)} records)")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the star trajectory experiment sweep."""
    parser = argparse.ArgumentParser(
        description="Eight-pointed star trajectory navigation experiment sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--speeds", nargs="+", type=float, default=None,
        metavar="V",
        help="Vehicle speeds [m/s]. Default: 2 5 10 20",
    )
    parser.add_argument(
        "--durations", nargs="+", type=float, default=None,
        metavar="T",
        help="Total trajectory durations [s]. Default: 300 600 1200 2400. "
             "Arm length D = speed*T/16; must satisfy D < map_size*dx/2.",
    )
    parser.add_argument(
        "--betas", nargs="+", type=float, default=None,
        metavar="B",
        help="PSD exponents (map spectral slope). Default: 0.7 1.3 2.0 3.5",
    )
    parser.add_argument(
        "--corr-lens-m", nargs="+", type=float, default=None,
        dest="corr_lens_m",
        metavar="CL",
        help="Correlation length roll-off scale [m]. Default: 300 1000 2500 4500",
    )
    parser.add_argument(
        "--initial-uncertainties", nargs="+", type=float, default=None,
        dest="initial_uncertainties",
        metavar="S",
        help="Initial position uncertainty 1-sigma [m]. Default: 100 500 1000 2000",
    )
    parser.add_argument("--extent-km", type=float, default=10.0, metavar="KM",
                        dest="extent_km",
                        help="Physical map side length [km]. Default 10.0.")
    parser.add_argument("--dx", type=float, default=30.0, metavar="M",
                        help="Pixel size [m/px] used when --dx-values is not set.")
    parser.add_argument(
        "--dx-values", nargs="+", type=float, default=None,
        dest="dx_values", metavar="M",
        help="Pixel sizes to sweep [m/px]. Overrides --dx when set. "
             "WARNING: values < 10 m produce grids > 1 M pixels and may be slow.",
    )
    parser.add_argument("--sigma-meas", type=float, default=0.1, metavar="S",
                        help="Measurement noise std [field_units].")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip open-loop Monte Carlo INS simulation.")
    parser.add_argument("--n-mc", type=int, default=50, metavar="N",
                        help="MC runs per configuration.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--outdir", type=str, default="data/experiments",
                        help="Output directory for CSV results.")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save results to disk.")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Number of parallel worker processes. "
                             "Default: os.cpu_count().")

    args = parser.parse_args()

    run_experiment(
        speeds=args.speeds,
        durations=args.durations,
        betas=args.betas,
        corr_lens_m=args.corr_lens_m,
        dx_values=args.dx_values,
        initial_uncertainties=args.initial_uncertainties,
        extent_km=args.extent_km,
        dx=args.dx,
        sigma_meas=args.sigma_meas,
        run_mc=not args.no_mc,
        n_mc=args.n_mc,
        seed=args.seed,
        outdir=args.outdir,
        save=not args.no_save,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
