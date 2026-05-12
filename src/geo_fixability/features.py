"""
Feature extraction for geophysical navigation fixability prediction.

Provides:
- trajectory_features: kinematic features from a position/time sequence
- trajectory_map_features: trajectory-map interaction features
- extract_all_features: convenience wrapper combining all feature groups
"""

import math

import numpy as np

from geo_fixability.metrics import map_information_content
from geo_fixability.navigation import compute_field_gradients, interpolate_field


def trajectory_features(
    positions: np.ndarray,
    times: np.ndarray,
    correlation_length_m: float | None = None,
) -> dict[str, float]:
    """
    Compute kinematic features from a trajectory.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time stamps [s].
    correlation_length_m : float or None
        Spatial correlation length [m] of the map; used to compute
        ``effective_dof``. If None, ``effective_dof`` is nan.

    Returns
    -------
    dict with keys:

    ``mean_speed_m_s`` : float
        Mean step speed [m/s].
    ``total_distance_m`` : float
        Cumulative path length [m].
    ``trajectory_duration_s`` : float
        Total elapsed time [s].
    ``heading_variance_rad2`` : float
        Circular variance of per-step headings [rad²].
    ``heading_change_count`` : int
        Number of steps where |Δheading| > π/8.
    ``measurement_density_per_km`` : float
        Samples per km of path length.
    ``effective_dof`` : float
        total_distance_m / correlation_length_m; nan if correlation_length_m
        is None.
    """
    dpos = np.diff(positions, axis=0)            # (N-1, 2)
    dt = np.diff(times)                          # (N-1,)

    step_dist = np.linalg.norm(dpos, axis=1)     # (N-1,)
    total_distance = float(step_dist.sum())
    duration = float(times[-1] - times[0])
    mean_dt = float(dt.mean()) if len(dt) > 0 else 1.0
    mean_speed = total_distance / duration if duration > 0 else 0.0

    # Heading at each step: arctan2(Δeast, Δnorth)
    headings = np.arctan2(dpos[:, 0], dpos[:, 1])   # (N-1,)

    # Circular variance: 1 - |mean of unit vectors|
    circ_var = float(1.0 - abs(np.mean(np.exp(1j * headings))))

    # Heading changes > π/8
    dhead = np.diff(headings)
    dhead = (dhead + math.pi) % (2 * math.pi) - math.pi  # wrap to (-π, π]
    heading_change_count = int(np.sum(np.abs(dhead) > math.pi / 8))

    meas_density = len(positions) / (total_distance / 1000.0) if total_distance > 0 else 0.0

    effective_dof = (
        total_distance / correlation_length_m
        if correlation_length_m is not None and correlation_length_m > 0
        else float("nan")
    )

    return {
        "mean_speed_m_s": mean_speed,
        "total_distance_m": total_distance,
        "trajectory_duration_s": duration,
        "heading_variance_rad2": circ_var,
        "heading_change_count": float(heading_change_count),
        "measurement_density_per_km": meas_density,
        "effective_dof": effective_dof,
    }


def trajectory_map_features(
    positions: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
) -> dict[str, float]:
    """
    Compute trajectory-map interaction features.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time stamps [s].
    field : ndarray, shape (ny, nx)
        Geophysical scalar field (zero-mean, unit-variance).
    dx : float
        Pixel size [m/pixel].
    sigma_meas : float
        Measurement noise standard deviation [field units].

    Returns
    -------
    dict with keys:

    ``gradient_rms_along_path`` : float
        RMS gradient magnitude at each step [field_units/m].
    ``gradient_parallel_mean`` : float
        Mean |along-track gradient component| [field_units/m].
    ``gradient_perpendicular_mean`` : float
        Mean |cross-track gradient component| [field_units/m].
    ``gradient_alignment_variance`` : float
        Circular variance of angle between heading and gradient [rad²].
    ``cumulative_fim_trace`` : float
        Total Fisher information Σ ||∇h||² / σ² along the trajectory.
    ``localization_bound_along_path_m`` : float
        sigma_meas / gradient_rms_along_path [m]; inf if flat.
    ``fraction_oob`` : float
        Fraction of steps whose position falls outside the field boundary.
    """
    gx, gy = compute_field_gradients(field, dx)

    N = len(positions)
    grad_mags = np.empty(N)
    grad_xs = np.empty(N)
    grad_ys = np.empty(N)
    oob_count = 0

    for k in range(N):
        px, py = float(positions[k, 0]), float(positions[k, 1])
        try:
            _, h_gx, h_gy = interpolate_field(field, gx, gy, px, py, dx)
            grad_xs[k] = h_gx
            grad_ys[k] = h_gy
            grad_mags[k] = math.sqrt(h_gx ** 2 + h_gy ** 2)
        except ValueError:
            grad_xs[k] = 0.0
            grad_ys[k] = 0.0
            grad_mags[k] = 0.0
            oob_count += 1

    fraction_oob = oob_count / N

    # Per-step headings from consecutive positions (use step k for step k)
    dpos = np.diff(positions, axis=0)   # (N-1, 2)
    step_dist = np.linalg.norm(dpos, axis=1)
    nonzero = step_dist > 0
    heading_east = np.where(nonzero, dpos[:, 0] / np.where(nonzero, step_dist, 1.0), 0.0)
    heading_north = np.where(nonzero, dpos[:, 1] / np.where(nonzero, step_dist, 1.0), 0.0)

    # Align gradient array with step k (use gradient at start of each step)
    gx_step = grad_xs[:-1]
    gy_step = grad_ys[:-1]

    # Along-track: dot product of gradient with heading unit vector
    parallel = np.abs(gx_step * heading_east + gy_step * heading_north)
    # Cross-track: magnitude of cross product (scalar in 2D: gx*hn - gy*he)
    perp = np.abs(gx_step * heading_north - gy_step * heading_east)

    # Gradient direction angle
    grad_angle = np.arctan2(grad_ys[:-1], grad_xs[:-1])
    heading_angle = np.arctan2(dpos[:, 0], dpos[:, 1])
    angle_diff = grad_angle - heading_angle
    # Circular variance of the angle difference
    align_var = float(1.0 - abs(np.mean(np.exp(1j * angle_diff))))

    grad_rms = float(np.sqrt(np.mean(grad_mags ** 2)))
    fim_total = float(np.sum(grad_mags ** 2) / sigma_meas ** 2)
    loc_bound = sigma_meas / grad_rms if grad_rms > 0 else float("inf")

    return {
        "gradient_rms_along_path": grad_rms,
        "gradient_parallel_mean": float(np.mean(parallel)),
        "gradient_perpendicular_mean": float(np.mean(perp)),
        "gradient_alignment_variance": align_var,
        "cumulative_fim_trace": fim_total,
        "localization_bound_along_path_m": loc_bound,
        "fraction_oob": fraction_oob,
    }


def patch_heterogeneity(field: np.ndarray, n_subdivisions: int = 4) -> float:
    """
    Measure spatial heterogeneity of a terrain patch.

    Partitions the patch into ``n_subdivisions × n_subdivisions`` non-overlapping
    sub-windows and returns the coefficient of variation (std/mean) of the
    per-sub-window gradient RMS.  High values indicate patchy terrain where
    some sub-regions are feature-rich and others flat — a known failure mode
    for the global-feature regression model.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Normalised terrain patch (zero-mean, unit-variance).
    n_subdivisions : int
        Grid dimension for sub-windows. Default 4 (→ 16 sub-windows).

    Returns
    -------
    float
        Coefficient of variation of sub-window gradient RMS.  Returns 0.0
        if all sub-windows have zero gradient (flat field).
    """
    ny, nx = field.shape
    h = ny // n_subdivisions
    w = nx // n_subdivisions
    if h < 4 or w < 4:
        return 0.0

    sub_grms = np.empty(n_subdivisions * n_subdivisions)
    k = 0
    for i in range(n_subdivisions):
        for j in range(n_subdivisions):
            sub = field[i * h:(i + 1) * h, j * w:(j + 1) * w]
            gy_sub, gx_sub = np.gradient(sub)
            sub_grms[k] = float(np.sqrt(np.mean(gx_sub ** 2 + gy_sub ** 2)))
            k += 1

    mean_grms = sub_grms.mean()
    if mean_grms < 1e-12:
        return 0.0
    return float(sub_grms.std() / mean_grms)


def extract_all_features(
    positions: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
) -> dict[str, float]:
    """
    Extract the full feature vector for one (trajectory, map) scenario.

    Combines map information content, trajectory kinematics, and
    trajectory-map interaction features into a single flat dict.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time stamps [s].
    field : ndarray, shape (ny, nx)
        Geophysical scalar field (zero-mean, unit-variance).
    dx : float
        Pixel size [m/pixel].
    sigma_meas : float
        Measurement noise standard deviation [field units].

    Returns
    -------
    dict
        Merged feature dict (~22 keys). Interaction features override
        map/trajectory features on key conflicts.  Includes ``dx_m = dx``
        as an explicit predictor.
    """
    map_info = map_information_content(field, dx=dx, sigma_meas=sigma_meas)
    traj_feats = trajectory_features(
        positions, times,
        correlation_length_m=map_info.get("correlation_length_m"),
    )
    interaction = trajectory_map_features(positions, times, field, dx, sigma_meas)

    result: dict[str, float] = {}
    result.update(map_info)
    result.update(traj_feats)
    result.update(interaction)
    result["dx_m"] = float(dx)
    return result
