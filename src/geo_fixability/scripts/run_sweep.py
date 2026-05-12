"""
Unified parameter sweep runner for geo-fixability pipeline.

Loads a sweep configuration (YAML or JSON), pre-computes INS calibration for
each unique drift-rate/dt pair, then dispatches all scenario combinations to a
``ProcessPoolExecutor`` and writes results to CSV.

Usage
-----
    uv run run-sweep                                  # uses sweep_config.yaml
    uv run run-sweep --config sweep_config.yaml
    uv run run-sweep --quick --workers 4              # 2 values per dim, quick test
    uv run run-sweep --run-pf --n-particles 200       # include particle filter
    uv run run-sweep --outdir data/my_sweep           # custom output directory
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from geo_fixability.pipeline import ScenarioParams, load_sweep_config, run_scenario
from geo_fixability.trajectory import calibrate_sigma_a


# ---------------------------------------------------------------------------
# Top-level worker (must be picklable → module-level function)
# ---------------------------------------------------------------------------


def _worker(
    idx: int,
    params_dict: dict,
    calibration_cache: dict[tuple, tuple[float, np.ndarray]],
) -> tuple[int, dict]:
    """Run one scenario and return (original_index, result_dict)."""
    key = (params_dict["ins_drift_rate"], params_dict["dt"])
    sigma_a, Q = calibration_cache[key]
    result = run_scenario(params_dict, sigma_a=sigma_a, Q=Q)
    return idx, result


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------


def _make_summary_plots(csv_path: Path, outdir: Path) -> None:
    """Generate three quick diagnostic plots from a completed sweep CSV."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available — skipping plots.", file=sys.stderr)
        return

    df = pd.read_csv(csv_path)
    ok = df[df["status"].isin(["ok", "warn_oob"])].copy()
    if ok.empty:
        print("No successful rows — skipping plots.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Scatter: improvement ratio vs information index
    ax = axes[0]
    sc = ax.scatter(
        ok["information_index"],
        ok["cep_improvement_ratio"],
        c=ok["speed"],
        cmap="viridis",
        s=6,
        alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label="speed [m/s]")
    ax.set_xlabel("information_index")
    ax.set_ylabel("cep_improvement_ratio")
    ax.set_title("Improvement vs Map Info")

    # 2) Heatmap: sigma_init × speed → mean cep_improvement_ratio
    ax = axes[1]
    if "sigma_init" in ok.columns and "speed" in ok.columns:
        pivot = ok.pivot_table(
            index="sigma_init",
            columns="speed",
            values="cep_improvement_ratio",
            aggfunc="mean",
        )
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="plasma",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0f}" for v in pivot.index], fontsize=7)
        ax.set_xlabel("speed [m/s]")
        ax.set_ylabel("sigma_init [m]")
        ax.set_title("Mean CEP Improvement")
        plt.colorbar(im, ax=ax)

    # 3) Histogram of final_cep_m
    ax = axes[2]
    valid_cep = ok["final_cep_m"].dropna()
    valid_cep = valid_cep[np.isfinite(valid_cep) & (valid_cep > 0)]
    if not valid_cep.empty:
        ax.hist(np.log10(valid_cep), bins=40, edgecolor="none", alpha=0.8)
        ax.set_xlabel("log10(final_cep_m)")
        ax.set_ylabel("count")
        ax.set_title("Distribution of Final CEP")

    fig.tight_layout()
    plot_path = outdir / "sweep_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Summary plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geo-fixability parameter sweep runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="sweep_config.yaml",
        help="Path to YAML or JSON sweep config file.",
    )
    parser.add_argument(
        "--outdir",
        default="data/sweep",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Output CSV filename (default: from config or 'sweep_results.csv').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only first 2 values per sweep dimension (rapid validation).",
    )
    parser.add_argument(
        "--run-pf",
        action="store_true",
        help="Enable particle filter for each scenario (much slower).",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Particles per PF run (overrides config fixed.n_particles).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (overrides config fixed.seed).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip summary plot generation.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    print(f"Loading config from {config_path} …")
    params_list, output_cfg = load_sweep_config(config_path)

    # Apply --quick: keep first 2 values per sweep dimension
    if args.quick:
        import json as _json

        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                yaml = None  # type: ignore[assignment]
        else:
            yaml = None

        if suffix in {".yaml", ".yml"} and yaml is not None:
            with config_path.open() as fh:
                raw_cfg = yaml.safe_load(fh)
        else:
            with config_path.open() as fh:
                raw_cfg = _json.load(fh)

        sweep_section = raw_cfg.get("sweep", {})
        fixed_section = raw_cfg.get("fixed", {})
        quick_sweep = {k: v[:2] for k, v in sweep_section.items()}
        from geo_fixability.pipeline import params_to_combinations

        params_list = params_to_combinations(quick_sweep, fixed_section)

    # Apply CLI overrides to every scenario
    overrides: dict = {}
    if args.run_pf:
        overrides["run_pf"] = True
    if args.n_particles is not None:
        overrides["n_particles"] = args.n_particles
    if args.seed is not None:
        overrides["seed"] = args.seed

    if overrides:
        new_list = []
        for p in params_list:
            d = asdict(p)
            d.update(overrides)
            new_list.append(ScenarioParams(**d))
        params_list = new_list

    n_total = len(params_list)
    print(f"Total scenarios: {n_total}")

    # ------------------------------------------------------------------
    # Output directory and file
    # ------------------------------------------------------------------
    outdir = Path(args.outdir or output_cfg.get("outdir", "data/sweep"))
    outdir.mkdir(parents=True, exist_ok=True)

    csv_filename = (
        args.filename
        or output_cfg.get("filename", "sweep_results.csv")
    )
    csv_path = outdir / csv_filename
    print(f"Results will be saved to {csv_path}")

    # ------------------------------------------------------------------
    # Pre-compute INS calibration cache (serial — avoids redundant work)
    # ------------------------------------------------------------------
    unique_keys = {(p.ins_drift_rate, p.dt) for p in params_list}
    print(
        f"Pre-computing INS calibration for {len(unique_keys)} unique "
        f"(drift_rate, dt) combination(s) …"
    )
    calibration_cache: dict[tuple, tuple[float, np.ndarray]] = {}
    for rate, dt_val in sorted(unique_keys):
        sigma_a, Q = calibrate_sigma_a(
            target_drift_rate=rate,
            eval_time=max(max(p.duration for p in params_list if p.ins_drift_rate == rate), 3600.0),
            dt=dt_val,
            n_mc=200,
            seed=42,
        )
        calibration_cache[(rate, dt_val)] = (sigma_a, Q)
        print(f"  drift_rate={rate:.5f} m/s, dt={dt_val}s → sigma_a={sigma_a:.4e}")

    # ------------------------------------------------------------------
    # Dispatch parallel sweep
    # ------------------------------------------------------------------
    print(f"\nRunning sweep with {args.workers} worker(s) …\n")
    t0 = time.perf_counter()

    results_by_idx: dict[int, dict] = {}
    n_ok = 0
    n_err = 0

    # Serialize params as plain dicts for cross-process compatibility
    params_dicts = [asdict(p) for p in params_list]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, idx, pd, calibration_cache): idx
            for idx, pd in enumerate(params_dicts)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results_by_idx[idx] = result

            status = result.get("status", "?")
            if status in ("ok", "warn_oob"):
                n_ok += 1
            else:
                n_err += 1

            elapsed = time.perf_counter() - t0
            done = len(results_by_idx)
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_total - done) / rate if rate > 0 else float("inf")

            improvement = result.get("cep_improvement_ratio", float("nan"))
            print(
                f"[{done:5d}/{n_total}] "
                f"status={status:<12s}  "
                f"improvement={improvement:6.2f}×  "
                f"elapsed={elapsed:6.1f}s  "
                f"eta={eta:6.1f}s"
            )

    elapsed_total = time.perf_counter() - t0
    print(
        f"\nDone: {n_ok} ok, {n_err} errors in {elapsed_total:.1f}s "
        f"({n_total / elapsed_total:.1f} scenarios/s)"
    )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    ordered_results = [results_by_idx[i] for i in range(n_total)]
    all_keys: list[str] = []
    seen_keys: set[str] = set()
    for r in ordered_results:
        for k in r:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in ordered_results:
            row = {k: r.get(k, float("nan")) for k in all_keys}
            writer.writerow(row)

    print(f"CSV saved → {csv_path}  ({n_total} rows, {len(all_keys)} columns)")

    # ------------------------------------------------------------------
    # Summary plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        _make_summary_plots(csv_path, outdir)


if __name__ == "__main__":
    main()
