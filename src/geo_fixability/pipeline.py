"""
Unified pipeline for geo-fixability parameter sweep experiments.

Provides a single ``run_scenario`` function that takes a ``ScenarioParams``
dataclass and returns a flat dict of results suitable for one CSV row.  Also
provides ``load_sweep_config`` and ``params_to_combinations`` helpers for
building parameter grids from a YAML/JSON config file.

Typical use
-----------
Single scenario::

    from geo_fixability.pipeline import run_scenario, ScenarioParams
    result = run_scenario(ScenarioParams(
        sigma_init=1000, speed=10, duration=600,
        ins_drift_rate=0.27778, sigma_meas=0.1,
        beta=2.0, correlation_length=1000, dx=50,
    ))

Parameter sweep::

    from geo_fixability.pipeline import load_sweep_config
    params_list, output_cfg = load_sweep_config("sweep_config.yaml")
    results = [run_scenario(p) for p in params_list]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from geo_fixability.features import extract_all_features
from geo_fixability.filtering import run_pf_mc
from geo_fixability.mapping import generate_field_spectral
from geo_fixability.metrics import map_information_content
from geo_fixability.navigation import navigation_improvement, pcrb_trajectory
from geo_fixability.trajectory import (
    calibrate_sigma_a,
    figure8_trajectory,
    lawnmower_trajectory,
    spiral_trajectory,
    star_trajectory,
    straight_trajectory,
)

__all__ = [
    "ScenarioParams",
    "run_scenario",
    "load_sweep_config",
    "params_to_combinations",
]

_NaN = float("nan")


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioParams:
    """
    Parameters for a single pipeline scenario.

    All fields are plain Python scalars so instances are picklable for use
    with ``concurrent.futures.ProcessPoolExecutor``.

    Primary sweep dimensions
    ------------------------
    sigma_init : float
        Initial 1-sigma position uncertainty [m].
    speed : float
        Vehicle speed [m/s].
    duration : float
        Trajectory duration [s].
    ins_drift_rate : float
        INS drift rate [m/s].  Convert from km/hr: ``km_per_hr * 1000 / 3600``.
    sigma_meas : float
        Measurement noise standard deviation [field units].
    beta : float
        Spectral PSD exponent for ``generate_field_spectral``.
    correlation_length : float
        Spatial feature scale [m].
    dx : float
        Map resolution [m/pixel].
    """

    # Required — primary sweep dimensions
    sigma_init: float
    speed: float
    duration: float
    ins_drift_rate: float
    sigma_meas: float
    beta: float
    correlation_length: float
    dx: float

    # Optional — fixed defaults
    dt: float = 1.0
    seed: int = 42
    trajectory_type: str = "auto"   # "auto" | "star" | "lawnmower" | "spiral" | "straight" | "figure8"
    run_pf: bool = False
    n_particles: int = 500
    n_pf_mc: int = 5
    max_map_npix: int = 2048        # hard cap on field size (pixels per side)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round_up_to_multiple(n: int, base: int = 64) -> int:
    """Round *n* up to the nearest multiple of *base*."""
    return ((n + base - 1) // base) * base


def _compute_map_geometry(params: ScenarioParams) -> tuple[int, float, float]:
    """
    Compute field size (pixels), map extent [m], and effective correlation
    length in pixels for a given scenario.

    Returns
    -------
    map_npix : int
        Square field side length [pixels].
    map_extent_m : float
        Physical extent of one field side [m].
    corr_len_px : float
        Correlation length [pixels] (may be clamped to map_npix / 4).
    """
    total_distance = params.speed * params.duration
    map_extent_m = max(2.0 * total_distance, 10.0 * params.correlation_length)

    raw_npix = math.ceil(map_extent_m / params.dx)
    raw_npix = _round_up_to_multiple(raw_npix, 64)
    map_npix = min(raw_npix, params.max_map_npix)

    # Physical extent from the (possibly capped) pixel count
    map_extent_m = map_npix * params.dx

    corr_len_px = params.correlation_length / params.dx
    corr_len_px = min(corr_len_px, map_npix / 4.0)

    return map_npix, map_extent_m, corr_len_px


def _generate_trajectory(
    params: ScenarioParams,
    map_extent_m: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Generate a trajectory and return (positions, times, type_used).

    ``positions`` is (N, 2) in metres, origin-centred.
    ``type_used`` is the concrete trajectory type that was generated.
    """
    ttype = params.trajectory_type
    total_distance = params.speed * params.duration

    # Auto-selection
    if ttype == "auto":
        if total_distance > 2.0 * map_extent_m:
            ttype = "lawnmower"
        else:
            ttype = "star"

    if ttype == "star":
        D = 0.4 * map_extent_m
        positions, times = star_trajectory(D, params.speed, params.dt)
    elif ttype == "lawnmower":
        side = 0.8 * map_extent_m
        lane_spacing = max(params.correlation_length, params.dx)
        positions, times = lawnmower_trajectory(
            width=side,
            height=side,
            lane_spacing=lane_spacing,
            speed=params.speed,
            dt=params.dt,
        )
        # Trim to requested duration
        max_steps = int(params.duration / params.dt) + 1
        positions = positions[:max_steps]
        times = times[:max_steps]
    elif ttype == "spiral":
        R_max = 0.4 * map_extent_m
        n_turns = max(1.0, total_distance / (2.0 * math.pi * R_max))
        positions, times = spiral_trajectory(R_max, n_turns, params.speed, params.dt)
    elif ttype == "straight":
        positions, times = straight_trajectory(
            total_distance, params.speed, heading_deg=45.0, dt=params.dt
        )
    elif ttype == "figure8":
        A = 0.3 * map_extent_m
        positions, times = figure8_trajectory(A, params.speed, params.dt)
    else:
        raise ValueError(f"Unknown trajectory_type: {params.trajectory_type!r}")

    return positions, times, ttype


def _nan_result(params: ScenarioParams, status: str) -> dict:
    """Return a result dict filled with NaN for all outputs."""
    return {
        "sigma_init": params.sigma_init,
        "speed": params.speed,
        "duration": params.duration,
        "ins_drift_rate": params.ins_drift_rate,
        "sigma_meas": params.sigma_meas,
        "beta": params.beta,
        "correlation_length": params.correlation_length,
        "dx": params.dx,
        "dt": params.dt,
        "seed": params.seed,
        "trajectory_type_requested": params.trajectory_type,
        "map_npix": _NaN,
        "map_extent_m": _NaN,
        "trajectory_type": _NaN,
        "sigma_a": _NaN,
        "map_size_capped": False,
        "correlation_clamped": False,
        "gradient_rms": _NaN,
        "information_index": _NaN,
        "spatial_diversity_index": _NaN,
        "localization_bound_m": _NaN,
        "correlation_length_m": _NaN,
        "localization_index": _NaN,
        "initial_cep_m": _NaN,
        "final_cep_m": _NaN,
        "cep_improvement_ratio": _NaN,
        "initial_sigma_m": _NaN,
        "final_sigma_m": _NaN,
        "mean_speed": _NaN,
        "total_distance": _NaN,
        "effective_dof": _NaN,
        "cumulative_fim_trace": _NaN,
        "pf_final_cep_m": _NaN,
        "pf_improvement_ratio": _NaN,
        "pf_n_eff_mean": _NaN,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_scenario(
    params: ScenarioParams | dict,
    *,
    sigma_a: float | None = None,
    Q: np.ndarray | None = None,
) -> dict:
    """
    Run the complete pipeline for a single scenario.

    Parameters
    ----------
    params : ScenarioParams or dict
        Scenario parameters.  A dict is converted to ``ScenarioParams``.
    sigma_a : float, optional
        Pre-computed IMU acceleration noise [m/s^(3/2)].  If supplied, the
        calibration step is skipped (useful in sweep runners that pre-compute
        the cache to avoid repeated binary searches).
    Q : ndarray (2, 2), optional
        Pre-computed per-step process noise covariance [m^2].  Must be
        supplied together with *sigma_a*.

    Returns
    -------
    dict
        Flat dict of input parameters, map metrics, PCRB results, trajectory
        features, and (if ``params.run_pf``) particle-filter results.
        On error, all output fields are ``nan`` and ``status`` describes
        the exception.
    """
    if isinstance(params, dict):
        params = ScenarioParams(**params)

    try:
        return _run_scenario_inner(params, sigma_a=sigma_a, Q=Q)
    except Exception as exc:  # noqa: BLE001
        return _nan_result(params, f"error: {exc}")


def _run_scenario_inner(
    params: ScenarioParams,
    sigma_a: float | None,
    Q: np.ndarray | None,
) -> dict:
    # ------------------------------------------------------------------
    # Step 1 — INS calibration (skip if pre-supplied)
    # ------------------------------------------------------------------
    if sigma_a is None or Q is None:
        sigma_a, Q = calibrate_sigma_a(
            target_drift_rate=params.ins_drift_rate,
            eval_time=max(params.duration, 3600.0),
            dt=params.dt,
            n_mc=100,
            seed=params.seed,
        )

    # ------------------------------------------------------------------
    # Step 2 — Map geometry
    # ------------------------------------------------------------------
    map_npix, map_extent_m, corr_len_px = _compute_map_geometry(params)

    raw_npix_uncapped = _round_up_to_multiple(
        math.ceil(max(
            2.0 * params.speed * params.duration,
            10.0 * params.correlation_length,
        ) / params.dx),
        64,
    )
    map_size_capped = raw_npix_uncapped > params.max_map_npix
    correlation_clamped = (params.correlation_length / params.dx) > (map_npix / 4.0)

    # ------------------------------------------------------------------
    # Step 3 — Map generation
    # ------------------------------------------------------------------
    rng_state = np.random.get_state()
    np.random.seed(params.seed)
    field = generate_field_spectral(
        shape=(map_npix, map_npix),
        beta=params.beta,
        correlation_length=corr_len_px,
    )
    np.random.set_state(rng_state)

    # ------------------------------------------------------------------
    # Step 4 — Map metrics
    # ------------------------------------------------------------------
    map_metrics = map_information_content(field, dx=params.dx, sigma_meas=params.sigma_meas)

    # ------------------------------------------------------------------
    # Step 5 — Trajectory generation
    # ------------------------------------------------------------------
    positions, times, ttype_used = _generate_trajectory(params, map_extent_m)

    # Warn if most of trajectory is outside the field
    half = map_npix * params.dx / 2.0
    oob = np.any(np.abs(positions) > half, axis=1)
    fraction_oob = float(oob.mean())
    status = "warn_oob" if fraction_oob > 0.05 else "ok"

    # ------------------------------------------------------------------
    # Step 6 — PCRB
    # ------------------------------------------------------------------
    P_init = params.sigma_init ** 2 * np.eye(2)
    P_traj = pcrb_trajectory(
        positions=positions,
        field=field,
        dx=params.dx,
        sigma_meas=params.sigma_meas,
        P_init=P_init,
        Q=Q,
    )
    pcrb_nav = navigation_improvement(P_traj)

    # ------------------------------------------------------------------
    # Step 7 — Feature extraction
    # ------------------------------------------------------------------
    features = extract_all_features(positions, times, field, params.dx, params.sigma_meas)

    # ------------------------------------------------------------------
    # Step 8 — Optional particle filter
    # ------------------------------------------------------------------
    pf_final_cep = _NaN
    pf_improvement_ratio = _NaN
    pf_n_eff_mean = _NaN

    if params.run_pf:
        pf_mc = run_pf_mc(
            positions_true=positions,
            times=times,
            field=field,
            dx=params.dx,
            sigma_meas=params.sigma_meas,
            P_init=P_init,
            Q=Q,
            n_mc=params.n_pf_mc,
            n_particles=params.n_particles,
            base_seed=params.seed,
        )
        pf_nav = navigation_improvement(pf_mc["P_trajectory_mean"])
        pf_final_cep = float(pf_nav["cep_final_m"])
        pf_improvement_ratio = float(pf_nav["cep_improvement_ratio"])
        pf_n_eff_mean = float(pf_mc["n_eff_mean"])

    # ------------------------------------------------------------------
    # Step 9 — Assemble result
    # ------------------------------------------------------------------
    result = {
        # Input parameters
        "sigma_init": params.sigma_init,
        "speed": params.speed,
        "duration": params.duration,
        "ins_drift_rate": params.ins_drift_rate,
        "sigma_meas": params.sigma_meas,
        "beta": params.beta,
        "correlation_length": params.correlation_length,
        "dx": params.dx,
        "dt": params.dt,
        "seed": params.seed,
        "trajectory_type_requested": params.trajectory_type,
        # Derived geometry / flags
        "map_npix": map_npix,
        "map_extent_m": map_extent_m,
        "trajectory_type": ttype_used,
        "sigma_a": float(sigma_a),
        "map_size_capped": map_size_capped,
        "correlation_clamped": correlation_clamped,
        # Map metrics
        "gradient_rms": float(map_metrics["gradient_rms"]),
        "information_index": float(map_metrics["information_index"]),
        "spatial_diversity_index": float(map_metrics["spatial_diversity_index"]),
        "localization_bound_m": float(map_metrics["localization_bound_m"]),
        "correlation_length_m": float(map_metrics["correlation_length_m"]),
        "localization_index": float(map_metrics["localization_index"]),
        # PCRB results
        "initial_cep_m": float(pcrb_nav["cep_initial_m"]),
        "final_cep_m": float(pcrb_nav["cep_final_m"]),
        "cep_improvement_ratio": float(pcrb_nav["cep_improvement_ratio"]),
        "initial_sigma_m": float(pcrb_nav["initial_uncertainty_m"]),
        "final_sigma_m": float(pcrb_nav["final_uncertainty_m"]),
        # Trajectory features
        "mean_speed": float(features.get("mean_speed_m_s", _NaN)),
        "total_distance": float(features.get("total_distance_m", _NaN)),
        "effective_dof": float(features.get("effective_dof", _NaN)),
        "cumulative_fim_trace": float(features.get("cumulative_fim_trace", _NaN)),
        # PF results (NaN if not run)
        "pf_final_cep_m": pf_final_cep,
        "pf_improvement_ratio": pf_improvement_ratio,
        "pf_n_eff_mean": pf_n_eff_mean,
        # Status
        "status": status,
    }

    return result


# ---------------------------------------------------------------------------
# Sweep config helpers
# ---------------------------------------------------------------------------


def params_to_combinations(
    sweep: dict[str, list],
    fixed: dict,
) -> list[ScenarioParams]:
    """
    Expand a sweep dict and fixed-params dict into a list of ``ScenarioParams``.

    Parameters
    ----------
    sweep : dict
        Maps parameter names to lists of values to sweep over.  The
        Cartesian product of all lists is taken.
    fixed : dict
        Maps parameter names to single scalar values applied to every scenario.

    Returns
    -------
    list[ScenarioParams]
        One entry per combination.  Seeds are offset by scenario index so
        each scenario gets a different map realisation.
    """
    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]
    base_seed = int(fixed.get("seed", 42))

    scenarios: list[ScenarioParams] = []
    for idx, combo in enumerate(product(*value_lists)):
        kw = dict(zip(keys, combo))
        kw.update(fixed)
        kw["seed"] = base_seed + idx
        scenarios.append(ScenarioParams(**kw))

    return scenarios


def load_sweep_config(
    path: str | Path | dict,
) -> tuple[list[ScenarioParams], dict]:
    """
    Load a sweep configuration from a YAML or JSON file (or a plain dict).

    The config must have a ``sweep`` section (lists of values per parameter)
    and may have ``fixed`` and ``output`` sections.

    Parameters
    ----------
    path : str, Path, or dict
        Path to a ``.yaml`` / ``.yml`` / ``.json`` file, or an already-parsed
        dict conforming to the same structure.

    Returns
    -------
    params_list : list[ScenarioParams]
        All scenario combinations.
    output_cfg : dict
        The ``output`` section of the config (empty dict if absent).
    """
    if isinstance(path, dict):
        cfg = path
    else:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "pyyaml is required to load YAML config files. "
                    "Install it with: uv add pyyaml"
                ) from exc
            with path.open() as fh:
                cfg = yaml.safe_load(fh)
        elif suffix == ".json":
            with path.open() as fh:
                cfg = json.load(fh)
        else:
            raise ValueError(
                f"Unsupported config file type: {suffix!r}. Use .yaml, .yml, or .json."
            )

    sweep_section = cfg.get("sweep", {})
    fixed_section = cfg.get("fixed", {})
    output_cfg = cfg.get("output", {})

    if not sweep_section:
        raise ValueError("Config must contain a non-empty 'sweep' section.")

    params_list = params_to_combinations(sweep_section, fixed_section)
    return params_list, output_cfg
