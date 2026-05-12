"""
Navigation performance bounds for geophysical map-aided navigation.

Provides:
- compute_field_gradients: central-difference gradient with north/east sign convention
- interpolate_field: bilinear interpolation at continuous positions in metres
- pcrb_trajectory: Posterior Cramér-Rao Bound recursion along a trajectory
- navigation_improvement: improvement metrics from a PCRB trajectory
- run_mc_ins: open-loop Monte Carlo INS drift simulation
"""

import math

import numpy as np

from geo_fixability.trajectory import imu_dead_reckoning


# ---------------------------------------------------------------------------
# Field gradient and interpolation
# ---------------------------------------------------------------------------


def compute_field_gradients(
    field: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial gradients of a scalar field using central differences.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Scalar field values (zero-mean, unit-variance).
    dx : float
        Pixel size [m/pixel]. Equal spacing assumed in both dimensions.

    Returns
    -------
    gx : ndarray, shape (ny, nx)
        East (x) gradient [field_units / m].
    gy : ndarray, shape (ny, nx)
        North (y) gradient [field_units / m].

    Notes
    -----
    ``np.gradient`` returns ``[d/d_row, d/d_col]`` in units of
    field-value per pixel. Rows increase **downward** in array indexing
    while y (north) increases **upward** in the physical frame, so the
    row-derivative is negated to obtain a north-positive gradient:

    .. code-block:: text

        gx =  d_col / dx   (east-positive, no sign change needed)
        gy = -d_row / dx   (north-positive, negated to flip row direction)
    """
    d_row, d_col = np.gradient(field)  # units: field-value / pixel
    gx = d_col / dx
    gy = -d_row / dx
    return gx, gy


def interpolate_field(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    px: float,
    py: float,
    dx: float,
) -> tuple[float, float, float]:
    """
    Bilinearly interpolate field value and gradient at a continuous position.

    Maps a physical position in metres (origin at field centre) to pixel
    coordinates and bilinearly interpolates the pre-computed field and
    gradient arrays.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Scalar field (zero-mean, unit-variance).
    gx : ndarray, shape (ny, nx)
        East gradient from ``compute_field_gradients`` [field_units/m].
    gy : ndarray, shape (ny, nx)
        North gradient from ``compute_field_gradients`` [field_units/m].
    px : float
        Vehicle x-position [m], origin at field centre, east-positive.
    py : float
        Vehicle y-position [m], origin at field centre, north-positive.
    dx : float
        Pixel size [m/pixel].

    Returns
    -------
    h_val : float
        Interpolated field value [field_units].
    h_gx : float
        Interpolated east gradient [field_units/m].
    h_gy : float
        Interpolated north gradient [field_units/m].

    Raises
    ------
    ValueError
        If (px, py) lies outside the field extent.
    """
    ny, nx = field.shape

    # Physical → pixel (fractional)
    col_f = nx / 2.0 + px / dx
    row_f = ny / 2.0 - py / dx

    if not (0.0 <= col_f <= nx - 1 and 0.0 <= row_f <= ny - 1):
        raise ValueError(
            f"Position ({px:.1f}, {py:.1f}) m is outside the field extent "
            f"({nx * dx / 2:.1f} m half-width). "
            f"Pixel coords: row={row_f:.2f}, col={col_f:.2f}."
        )

    r0 = min(int(row_f), ny - 2)
    c0 = min(int(col_f), nx - 2)
    dr = row_f - r0
    dc = col_f - c0

    # Bilinear weights
    w00 = (1.0 - dr) * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w10 = dr * (1.0 - dc)
    w11 = dr * dc

    def _interp(arr: np.ndarray) -> float:
        return (
            w00 * arr[r0, c0]
            + w01 * arr[r0, c0 + 1]
            + w10 * arr[r0 + 1, c0]
            + w11 * arr[r0 + 1, c0 + 1]
        )

    return _interp(field), _interp(gx), _interp(gy)


# ---------------------------------------------------------------------------
# PCRB recursion
# ---------------------------------------------------------------------------


def pcrb_trajectory(
    positions: np.ndarray,
    field: np.ndarray,
    dx: float,
    sigma_meas: float,
    P_init: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Compute the Posterior Cramér-Rao Bound along a trajectory.

    Implements the Kalman information-filter recursion for a 2D position
    state aiding with scalar geophysical measurements. This equals the
    PCRB under the linear-Gaussian approximation (gradient evaluated at
    the true position).

    **State**: x = [px, py] (2D position, metres)

    **Dynamics** (position-only state, driven by IMU):

    .. math::

        x_k = x_{k-1} + w_k, \\quad w_k \\sim \\mathcal{N}(0, Q)

    **Measurement**:

    .. math::

        z_k = h(x_k) + v_k, \\quad v_k \\sim \\mathcal{N}(0, \\sigma_{\\text{meas}}^2)

    where :math:`h` is the geophysical field value, linearised as:

    .. math::

        H_k = [g_x(p_k),\\; g_y(p_k)] \\in \\mathbb{R}^{1 \\times 2}

    **Recursion** (Joseph-form update for numerical stability):

    .. code-block:: text

        P_pred = P[k-1] + Q
        S = H_k P_pred H_k^T + sigma_meas^2       (scalar)
        K = P_pred H_k^T / S                       (2×1)
        IKH = I - K H_k
        P[k] = IKH P_pred IKH^T + K sigma_meas^2 K^T

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        True vehicle positions [m]. These are linearisation points; no
        feedback or estimation occurs.
    field : ndarray, shape (ny, nx)
        Scalar geophysical field (zero-mean, unit-variance).
    dx : float
        Pixel size [m/pixel].
    sigma_meas : float
        Measurement noise standard deviation [field_units]. Must be > 0.
    P_init : ndarray, shape (2, 2)
        Initial position covariance [m^2].
    Q : ndarray, shape (2, 2)
        Process noise covariance [m^2] per time step.

    Returns
    -------
    P_trajectory : ndarray, shape (N, 2, 2)
        Posterior covariance lower bound [m^2] at each step.
        ``P_trajectory[0] = P_init``.

    Notes
    -----
    When the gradient magnitude at a position is below 1e-9 field_units/m,
    the update is skipped (``P[k] = P_pred``). This correctly encodes zero
    Fisher information at flat-terrain locations.

    The Joseph form guarantees positive semi-definiteness even after
    hundreds of sequential updates, unlike the standard ``(I - KH) P`` form.
    """
    gx, gy = compute_field_gradients(field, dx)

    N = len(positions)
    P_traj = np.empty((N, 2, 2))
    P_traj[0] = P_init.copy()

    I2 = np.eye(2)
    R = sigma_meas ** 2
    _GRAD_THRESH = 1e-9

    P = P_init.copy()

    for k in range(1, N):
        # --- Predict ---
        P_pred = P + Q

        # --- Measurement Jacobian at true position ---
        px, py = positions[k]
        try:
            _, h_gx, h_gy = interpolate_field(field, gx, gy, px, py, dx)
        except ValueError:
            # Out-of-bounds: skip update (conservative)
            P = P_pred
            P_traj[k] = P
            continue

        grad_mag = math.sqrt(h_gx ** 2 + h_gy ** 2)
        if grad_mag < _GRAD_THRESH:
            # No information at this location
            P = P_pred
            P_traj[k] = P
            continue

        # H_k: shape (1, 2)
        H = np.array([[h_gx, h_gy]])

        # --- Update (Joseph form) ---
        S = float((H @ P_pred @ H.T)[0, 0]) + R  # scalar
        K = (P_pred @ H.T) / S                   # (2, 1)
        IKH = I2 - K @ H                          # (2, 2)
        P = IKH @ P_pred @ IKH.T + K * R * K.T   # Joseph form

        # Enforce symmetry
        P = (P + P.T) * 0.5
        P_traj[k] = P

    return P_traj


# ---------------------------------------------------------------------------
# Improvement metrics
# ---------------------------------------------------------------------------


def navigation_improvement(
    P_trajectory: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Compute navigation performance improvement metrics from a PCRB trajectory.

    Parameters
    ----------
    P_trajectory : ndarray, shape (N, 2, 2)
        PCRB covariance matrices from ``pcrb_trajectory``.

    Returns
    -------
    dict with the following keys:

    ``pos_uncertainty`` : ndarray, shape (N,)
        RMS position uncertainty = ``sqrt(trace(P[k]))`` at each step [m].
        Represents the combined 1-sigma error in x and y.

    ``initial_uncertainty_m`` : float
        ``pos_uncertainty[0]`` [m].

    ``final_uncertainty_m`` : float
        ``pos_uncertainty[-1]`` [m].

    ``improvement_ratio`` : float
        ``initial_uncertainty / final_uncertainty``. Values > 1 indicate
        improvement.

    ``reduction_fraction`` : float
        ``1 - final / initial``. In [0, 1]; 0 = no improvement.

    ``cep_initial_m`` : float
        Circular Error Probable at start [m], approximated as
        ``sigma_per_axis * sqrt(2 * ln 2)`` where
        ``sigma_per_axis = sqrt(trace(P_init) / 2)``.

    ``cep_final_m`` : float
        CEP at end of trajectory [m].

    ``cep_improvement_ratio`` : float
        ``cep_initial / cep_final``.

    Notes
    -----
    CEP for a 2D isotropic Gaussian with per-axis variance ``sigma^2``:

    .. math::

        \\text{CEP} = \\sigma \\sqrt{2 \\ln 2} \\approx 1.1774\\,\\sigma

    For anisotropic ``P`` this is an approximation using the mean axis
    variance ``sqrt(trace(P) / 2)``.
    """
    _CEP_FACTOR = math.sqrt(2.0 * math.log(2.0))  # ≈ 1.1774

    # Per-step RMS uncertainty
    traces = np.trace(P_trajectory, axis1=1, axis2=2)   # (N,)
    pos_unc = np.sqrt(np.maximum(traces, 0.0))

    u0 = float(pos_unc[0])
    uf = float(pos_unc[-1])
    ratio = u0 / uf if uf > 0 else math.inf
    reduction = 1.0 - uf / u0 if u0 > 0 else 0.0

    sigma_axis_init = math.sqrt(max(float(np.trace(P_trajectory[0])) / 2.0, 0.0))
    sigma_axis_final = math.sqrt(max(float(np.trace(P_trajectory[-1])) / 2.0, 0.0))
    cep_init = sigma_axis_init * _CEP_FACTOR
    cep_final = sigma_axis_final * _CEP_FACTOR
    cep_ratio = cep_init / cep_final if cep_final > 0 else math.inf

    return {
        "pos_uncertainty": pos_unc,
        "initial_uncertainty_m": u0,
        "final_uncertainty_m": uf,
        "improvement_ratio": ratio,
        "reduction_fraction": reduction,
        "cep_initial_m": cep_init,
        "cep_final_m": cep_final,
        "cep_improvement_ratio": cep_ratio,
    }


# ---------------------------------------------------------------------------
# Open-loop Monte Carlo INS simulation
# ---------------------------------------------------------------------------


def run_mc_ins(
    positions_true: np.ndarray,
    times: np.ndarray,
    sigma_v: float,
    sigma_a: float,
    n_runs: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Open-loop Monte Carlo INS drift simulation.

    Runs ``n_runs`` dead-reckoning simulations along the true trajectory
    with no map corrections. Used to empirically validate the calibrated
    ``sigma_a`` and to compare open-loop drift against the PCRB.

    Parameters
    ----------
    positions_true : ndarray, shape (N, 2)
        True trajectory [m].
    times : ndarray, shape (N,)
        Time array [s].
    sigma_v : float
        White velocity noise [m/s].
    sigma_a : float
        Calibrated acceleration noise [m/s^(3/2)].
    n_runs : int
        Number of Monte Carlo runs. Default 50.
    seed : int
        Base random seed; run i uses ``seed + i``. Default 0.

    Returns
    -------
    dict with keys:

    ``rmse_x`` : ndarray, shape (N,)
        RMS position error in x across runs [m].
    ``rmse_y`` : ndarray, shape (N,)
        RMS position error in y across runs [m].
    ``rmse_total`` : ndarray, shape (N,)
        ``sqrt(rmse_x^2 + rmse_y^2)`` [m].
    ``errors_all`` : ndarray, shape (n_runs, N, 2)
        All individual run position errors [m].
    """
    N = len(positions_true)
    errors_all = np.empty((n_runs, N, 2))

    for i in range(n_runs):
        _, err = imu_dead_reckoning(
            positions_true, times,
            sigma_v=sigma_v,
            sigma_a=sigma_a,
            seed=seed + i,
        )
        errors_all[i] = err

    rmse_x = np.sqrt(np.mean(errors_all[:, :, 0] ** 2, axis=0))
    rmse_y = np.sqrt(np.mean(errors_all[:, :, 1] ** 2, axis=0))
    rmse_total = np.sqrt(rmse_x ** 2 + rmse_y ** 2)

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_total": rmse_total,
        "errors_all": errors_all,
    }
