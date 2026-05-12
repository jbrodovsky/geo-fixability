"""
Bootstrap Particle Filter (SIR) for geophysical map-aided navigation.

Provides:
- particle_filter: Sequential Importance Resampling over a 2D position state
- run_pf_mc: Monte Carlo aggregation of multiple particle filter runs
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interp_batch(
    field: np.ndarray,
    px_arr: np.ndarray,
    py_arr: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation of a scalar field at multiple positions.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Scalar field (zero-mean, unit-variance). Physical origin at array centre.
    px_arr : ndarray, shape (M,)
        East positions [m]. Positive east.
    py_arr : ndarray, shape (M,)
        North positions [m]. Positive north.
    dx : float
        Pixel size [m/pixel].

    Returns
    -------
    vals : ndarray, shape (M,)
        Interpolated field values. 0.0 for out-of-bounds positions.
    oob_mask : ndarray of bool, shape (M,)
        True where the position lies outside the field boundary.

    Notes
    -----
    Coordinate mapping matches ``navigation.interpolate_field``:
    ``col_f = nx/2 + px/dx``, ``row_f = ny/2 - py/dx``.
    Rows increase downward; y (north) increases upward, hence the negation.
    """
    ny, nx = field.shape
    col_f = nx / 2.0 + px_arr / dx      # (M,)
    row_f = ny / 2.0 - py_arr / dx      # (M,)

    oob_mask = (col_f < 0) | (col_f > nx - 1) | (row_f < 0) | (row_f > ny - 1)

    # Clamp into valid range so integer indices are safe (OOB values zeroed later)
    col_c = np.clip(col_f, 0.0, nx - 1.0 - 1e-9)
    row_c = np.clip(row_f, 0.0, ny - 1.0 - 1e-9)

    c0 = np.floor(col_c).astype(np.intp)   # (M,)
    r0 = np.floor(row_c).astype(np.intp)   # (M,)
    dc = col_c - c0                          # fractional column offset
    dr = row_c - r0                          # fractional row offset

    w00 = (1.0 - dr) * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w10 = dr * (1.0 - dc)
    w11 = dr * dc

    vals = (
        w00 * field[r0,     c0    ]
      + w01 * field[r0,     c0 + 1]
      + w10 * field[r0 + 1, c0    ]
      + w11 * field[r0 + 1, c0 + 1]
    )
    vals[oob_mask] = 0.0
    return vals, oob_mask


def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Systematic resampling.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalised particle weights (must sum to 1).
    rng : numpy.random.Generator

    Returns
    -------
    indices : ndarray of int, shape (N,)
        Particle indices to keep (with replacement).
    """
    n = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0                          # guard against floating-point rounding
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    return np.searchsorted(cumsum, positions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def particle_filter(
    positions_true: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
    P_init: np.ndarray,
    Q: np.ndarray,
    n_particles: int = 1000,
    seed: int | None = None,
    resample_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Bootstrap Particle Filter (SIR) for 2D geophysical map-aided navigation.

    State: 2D position x = [p_x, p_y] [m].
    Process: x_k = x_{k-1} + w_k,  w_k ~ N(0, Q).  (random walk, matches PCRB)
    Measurement: z_k = h(x_k) + v_k,  v_k ~ N(0, sigma_meas²).

    Parameters
    ----------
    positions_true : ndarray, shape (N, 2)
        Ground-truth trajectory [m], columns [east, north].
    times : ndarray, shape (N,)
        Time stamps [s].
    field : ndarray, shape (ny, nx)
        Geophysical scalar field (zero-mean, unit-variance). Origin at centre.
    dx : float
        Field pixel size [m/pixel].
    sigma_meas : float
        Measurement noise standard deviation [field units].
    P_init : ndarray, shape (2, 2)
        Initial position covariance [m²].
    Q : ndarray, shape (2, 2)
        Per-step process noise covariance [m²]. Assumed diagonal.
    n_particles : int
        Number of particles. Default 1000.
    seed : int or None
        Random seed for reproducibility. Default None.
    resample_threshold : float
        Resample when N_eff < resample_threshold * n_particles. Default 0.5.

    Returns
    -------
    dict with keys:

    ``P_trajectory`` : ndarray, shape (N, 2, 2)
        Weighted sample covariance at each step [m²].
    ``mean_trajectory`` : ndarray, shape (N, 2)
        Weighted particle mean at each step [m].
    ``pos_uncertainty`` : ndarray, shape (N,)
        Positional uncertainty sqrt(trace(P_k)) at each step [m].
    ``n_eff`` : ndarray, shape (N,)
        Effective particle count 1/sum(w²) at each step.

    Notes
    -----
    Measurements are simulated as z_k = h(x_true[k]) + N(0, sigma_meas²),
    so each call to this function produces an independent noise realisation.

    The filter cannot outperform the PCRB; pf_cep_final / pcrb_cep_final >= 1
    always holds in expectation.
    """
    rng = np.random.default_rng(seed)
    n_steps = len(positions_true)
    m = n_particles
    r_var = sigma_meas ** 2
    noise_std = np.sqrt(np.diag(Q))          # (2,) per-axis process noise std

    # --- Initialise particles from prior N(x_true[0], P_init) ---
    L = np.linalg.cholesky(P_init)
    particles = (
        positions_true[0] + rng.standard_normal((m, 2)) @ L.T
    )                                        # (M, 2)
    log_w = np.full(m, -np.log(float(m)))   # uniform log-weights

    # --- Allocate output arrays ---
    P_traj = np.empty((n_steps, 2, 2))
    mean_traj = np.empty((n_steps, 2))
    pos_unc = np.empty(n_steps)
    n_eff_arr = np.empty(n_steps)

    # Step 0: record prior (no measurement update yet)
    P_traj[0] = P_init.copy()
    mean_traj[0] = positions_true[0].copy()
    pos_unc[0] = np.sqrt(float(np.trace(P_init)))
    n_eff_arr[0] = float(m)

    # --- Main filter loop ---
    for k in range(1, n_steps):
        # (a) Predict: independent random walk per axis
        particles += rng.standard_normal((m, 2)) * noise_std

        # (b) Simulate scalar measurement at the true position
        h_true, _ = _interp_batch(
            field,
            np.array([positions_true[k, 0]]),
            np.array([positions_true[k, 1]]),
            dx,
        )
        z_k = float(h_true[0]) + rng.standard_normal() * sigma_meas

        # (c) Log-likelihood update for all particles
        h_p, oob = _interp_batch(field, particles[:, 0], particles[:, 1], dx)
        log_w += -0.5 * (z_k - h_p) ** 2 / r_var
        log_w[oob] = -np.inf

        # (d) Normalise using log-sum-exp for numerical stability
        finite = np.isfinite(log_w)
        if not finite.any():
            warnings.warn(
                f"All particle weights are zero at step {k}; resetting to uniform.",
                RuntimeWarning,
                stacklevel=2,
            )
            log_w[:] = -np.log(float(m))
            finite = np.ones(m, dtype=bool)

        log_w[finite] -= log_w[finite].max()
        log_w[~finite] = -np.inf
        w = np.where(finite, np.exp(log_w), 0.0)
        w /= w.sum()

        # (e) Weighted mean and covariance
        mu = (w[:, None] * particles).sum(axis=0)         # (2,)
        diff = particles - mu                              # (M, 2)
        P_k = (
            w[:, None, None] * diff[:, :, None] * diff[:, None, :]
        ).sum(axis=0)                                      # (2, 2)
        P_k = (P_k + P_k.T) * 0.5                        # enforce symmetry

        n_eff_k = 1.0 / float(np.sum(w ** 2))

        P_traj[k] = P_k
        mean_traj[k] = mu
        pos_unc[k] = np.sqrt(max(float(np.trace(P_k)), 0.0))
        n_eff_arr[k] = n_eff_k

        # (f) Systematic resampling when effective count drops too low
        if n_eff_k < resample_threshold * m:
            indices = _systematic_resample(w, rng)
            particles = particles[indices]
            log_w[:] = -np.log(float(m))

    return {
        "P_trajectory": P_traj,
        "mean_trajectory": mean_traj,
        "pos_uncertainty": pos_unc,
        "n_eff": n_eff_arr,
    }


def run_pf_mc(
    positions_true: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
    P_init: np.ndarray,
    Q: np.ndarray,
    n_mc: int = 10,
    n_particles: int = 1000,
    base_seed: int = 0,
    resample_threshold: float = 0.5,
) -> dict[str, np.ndarray | float | int]:
    """
    Run the Bootstrap Particle Filter multiple times and aggregate statistics.

    Each run uses seed ``base_seed + i`` so results are reproducible but
    statistically independent across runs.

    Parameters
    ----------
    positions_true : ndarray, shape (N, 2)
    times : ndarray, shape (N,)
    field : ndarray, shape (ny, nx)
    dx : float
    sigma_meas : float
    P_init : ndarray, shape (2, 2)
    Q : ndarray, shape (2, 2)
    n_mc : int
        Number of independent filter runs. Default 10.
    n_particles : int
        Particles per run. Default 1000.
    base_seed : int
        Seeds run from base_seed to base_seed + n_mc - 1. Default 0.
    resample_threshold : float

    Returns
    -------
    dict with keys:

    ``pos_uncertainty_mean`` : ndarray, shape (N,)
        Mean positional uncertainty across MC runs [m].
    ``pos_uncertainty_std`` : ndarray, shape (N,)
        Std of positional uncertainty across MC runs [m].
    ``P_trajectory_mean`` : ndarray, shape (N, 2, 2)
        Mean covariance matrix across MC runs [m²].
    ``n_eff_mean`` : float
        Mean effective particle count over all steps and runs.
    ``n_eff_min`` : float
        Minimum effective particle count (worst-case degeneracy indicator).
    ``n_mc`` : int
    ``n_particles`` : int
    """
    runs = [
        particle_filter(
            positions_true, times, field, dx,
            sigma_meas, P_init, Q,
            n_particles=n_particles,
            seed=base_seed + i,
            resample_threshold=resample_threshold,
        )
        for i in range(n_mc)
    ]

    all_pos_unc = np.stack([r["pos_uncertainty"] for r in runs])   # (n_mc, N)
    all_n_eff   = np.stack([r["n_eff"]           for r in runs])   # (n_mc, N)
    all_P_traj  = np.stack([r["P_trajectory"]    for r in runs])   # (n_mc, N, 2, 2)

    return {
        "pos_uncertainty_mean": all_pos_unc.mean(axis=0),
        "pos_uncertainty_std":  all_pos_unc.std(axis=0),
        "P_trajectory_mean":    all_P_traj.mean(axis=0),
        "n_eff_mean":           float(all_n_eff.mean()),
        "n_eff_min":            float(all_n_eff.min()),
        "n_mc":                 n_mc,
        "n_particles":          n_particles,
    }
