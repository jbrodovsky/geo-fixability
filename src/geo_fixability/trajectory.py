"""
Trajectory generation and INS noise simulation for geophysical navigation.

Provides:
- star_trajectory: eight-pointed star pattern centered at origin
- straight_trajectory: straight-line flight at a given heading
- lawnmower_trajectory: grid/boustrophedon coverage pattern
- spiral_trajectory: outward Archimedean spiral
- random_walk_trajectory: constant-speed random-heading walk
- figure8_trajectory: two tangent circles (lemniscate approximation)
- imu_dead_reckoning: correlated velocity noise → position drift simulation
- calibrate_sigma_a: Monte Carlo calibration of IMU noise to a target drift rate
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def star_trajectory(
    D: float,
    speed: float,
    dt: float = 1.0,
    n_arms: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an n-armed star trajectory centered at the origin.

    The vehicle travels from the center to each arm tip and back, visiting
    all n_arms directions in order of increasing angle (0°, 45°, ...).
    Every arm traversal passes through the center at exact coordinates to
    avoid floating-point accumulation across legs.

    Parameters
    ----------
    D : float
        Arm length — distance from center to arm tip [m]. Must be > 0.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    dt : float
        Time step [s]. Default 1.0.
    n_arms : int
        Number of star arms. Default 8 (eight-pointed star, 45° separation).

    Returns
    -------
    positions : ndarray, shape (N, 2)
        True position [m] at each time step. Column 0 = x (east),
        column 1 = y (north). Starts and ends at (0, 0).
    times : ndarray, shape (N,)
        Time [s] at each sample, starting at 0.

    Notes
    -----
    Total trajectory distance = 2 * n_arms * D [m].
    Total duration           = 2 * n_arms * D / speed [s].
    Number of samples N      ≈ total_duration / dt + 1.

    Each leg is sampled at intervals of ``speed * dt`` metres. The final
    sample of each leg is placed exactly at the endpoint (center or tip)
    regardless of whether the leg length is a multiple of the step size.
    This guarantees exact center crossings and exact tip positions.

    Examples
    --------
    >>> pos, t = star_trajectory(D=1000.0, speed=5.0, dt=1.0)
    >>> pos[0]
    array([0., 0.])
    >>> pos[-1]
    array([0., 0.])
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Arm tip directions
    angles = [2 * math.pi * i / n_arms for i in range(n_arms)]
    tips = np.array([[D * math.cos(a), D * math.sin(a)] for a in angles])
    center = np.zeros(2)

    # Build the leg sequence: center→tip, tip→center, repeated for each arm
    legs: list[tuple[np.ndarray, np.ndarray]] = []
    for tip in tips:
        legs.append((center.copy(), tip.copy()))
        legs.append((tip.copy(), center.copy()))

    # Sample each leg
    step_dist = speed * dt
    segments: list[np.ndarray] = []

    for start, end in legs:
        leg_len = np.linalg.norm(end - start)
        n_steps = max(1, int(leg_len / step_dist))
        # Evenly spaced fractions from 0 to 1, excluding the endpoint
        # (it will be added as the start of the next leg or at the very end)
        fracs = np.linspace(0.0, 1.0, n_steps + 1)[:-1]
        pts = start + np.outer(fracs, end - start)
        segments.append(pts)

    # Append the final center position (trajectory ends at origin)
    all_pts = np.vstack(segments + [center.reshape(1, 2)])

    N = len(all_pts)
    times = np.arange(N) * dt

    return all_pts, times


def straight_trajectory(
    D: float,
    speed: float,
    heading_deg: float = 0.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a straight-line trajectory from the origin.

    Parameters
    ----------
    D : float
        Total distance to travel [m]. Must be > 0.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    heading_deg : float
        Compass heading: 0 = north, 90 = east, 180 = south, 270 = west [degrees].
        Default 0.0 (north).
    dt : float
        Time step [s]. Default 1.0.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time [s] starting at 0.

    Notes
    -----
    Total samples N = floor(D / (speed * dt)) + 1.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    heading_rad = math.radians(heading_deg)
    step = speed * dt
    N = int(D / step) + 1
    d_vals = np.arange(N) * step
    # Compass: east = sin(heading), north = cos(heading)
    east = d_vals * math.sin(heading_rad)
    north = d_vals * math.cos(heading_rad)
    positions = np.column_stack([east, north])
    times = np.arange(N) * dt
    return positions, times


def lawnmower_trajectory(
    width: float,
    height: float,
    lane_spacing: float,
    speed: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a boustrophedon (lawnmower/grid) coverage trajectory.

    The vehicle starts at the south-west corner (-width/2, -height/2), sweeps
    east, steps north by lane_spacing, then sweeps west, and so on until the
    full height is covered.

    Parameters
    ----------
    width : float
        East-west extent of the coverage area [m]. Must be > 0.
    height : float
        North-south extent [m]. Must be > 0.
    lane_spacing : float
        Distance between adjacent east-west passes [m]. Must be > 0.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    dt : float
        Time step [s]. Default 1.0.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time [s] starting at 0.
    """
    if width <= 0 or height <= 0 or lane_spacing <= 0:
        raise ValueError("width, height, and lane_spacing must all be positive")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    n_passes = max(1, round(height / lane_spacing))
    step = speed * dt
    waypoints: list[np.ndarray] = []

    for i in range(n_passes):
        y = -height / 2.0 + i * (height / max(n_passes - 1, 1)) if n_passes > 1 else 0.0
        if i == 0 and n_passes == 1:
            y = -height / 2.0
        x_start = -width / 2.0 if i % 2 == 0 else width / 2.0
        x_end = width / 2.0 if i % 2 == 0 else -width / 2.0
        n_steps = max(1, int(abs(x_end - x_start) / step))
        xs = np.linspace(x_start, x_end, n_steps + 1)
        ys = np.full(n_steps + 1, y)
        waypoints.append(np.column_stack([xs, ys]))

    positions = np.vstack(waypoints)
    times = np.arange(len(positions)) * dt
    return positions, times


def spiral_trajectory(
    R_max: float,
    n_turns: float,
    speed: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an outward Archimedean spiral trajectory.

    The radius grows linearly with arc-length so that the vehicle travels at
    constant speed from the origin outward.

    Parameters
    ----------
    R_max : float
        Maximum radius at the end of the spiral [m]. Must be > 0.
    n_turns : float
        Total number of full revolutions. Must be >= 1.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    dt : float
        Time step [s]. Default 1.0.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time [s] starting at 0.

    Notes
    -----
    Archimedean spiral: ``r(θ) = R_max * θ / (2π * n_turns)``.
    Arc-length parametrisation is computed numerically via cumulative sum on a
    fine θ grid, then resampled at uniform speed * dt arc-length increments.
    """
    if R_max <= 0:
        raise ValueError(f"R_max must be positive, got {R_max}")
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    theta_max = 2.0 * math.pi * n_turns
    # Fine grid for arc-length integration
    n_fine = max(10000, int(theta_max * 200))
    theta_fine = np.linspace(0.0, theta_max, n_fine)

    a = R_max / theta_max  # spiral constant r = a * θ
    r_fine = a * theta_fine

    # Arc-length element ds = sqrt(r² + (dr/dθ)²) dθ = sqrt(r² + a²) dθ
    ds = np.sqrt(r_fine ** 2 + a ** 2) * (theta_fine[1] - theta_fine[0])
    s_fine = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    total_arc = float(s_fine[-1])
    N = int(total_arc / (speed * dt)) + 1
    s_uniform = np.arange(N) * speed * dt
    s_uniform = np.minimum(s_uniform, total_arc)

    theta_sampled = np.interp(s_uniform, s_fine, theta_fine)
    r_sampled = a * theta_sampled
    east = r_sampled * np.cos(theta_sampled)
    north = r_sampled * np.sin(theta_sampled)

    positions = np.column_stack([east, north])
    times = np.arange(N) * dt
    return positions, times


def random_walk_trajectory(
    D_total: float,
    step_scale: float,
    speed: float,
    dt: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a constant-speed random-heading walk.

    The heading evolves as a random walk: at each time step the heading
    changes by a zero-mean Gaussian increment whose standard deviation scales
    with ``speed * dt / step_scale``. Larger ``step_scale`` means smoother
    (less turning) paths.

    Parameters
    ----------
    D_total : float
        Total desired path length [m]. Must be > 0.
    step_scale : float
        Turning radius scale [m]. Larger values produce straighter paths.
        Typical range: 0.5 * (speed * dt) to 10 * D_total. Must be > 0.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    dt : float
        Time step [s]. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time [s] starting at 0.
    """
    if D_total <= 0:
        raise ValueError(f"D_total must be positive, got {D_total}")
    if step_scale <= 0:
        raise ValueError(f"step_scale must be positive, got {step_scale}")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    rng = np.random.default_rng(seed)
    N = int(D_total / (speed * dt)) + 1
    heading = 0.0  # start heading north
    sigma_turn = speed * dt / step_scale

    positions = np.empty((N, 2))
    positions[0] = 0.0
    times = np.arange(N) * dt

    for k in range(1, N):
        heading += float(rng.normal(0.0, sigma_turn))
        # Wrap to (-π, π]
        heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
        step = speed * dt
        positions[k, 0] = positions[k - 1, 0] + step * math.sin(heading)
        positions[k, 1] = positions[k - 1, 1] + step * math.cos(heading)

    return positions, times


def figure8_trajectory(
    A: float,
    speed: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a figure-eight trajectory from two tangent circles.

    The vehicle traverses circle 1 centered at (-A, 0) counterclockwise,
    then circle 2 centered at (+A, 0) clockwise. Both circles have radius A
    and are connected at the origin.

    Parameters
    ----------
    A : float
        Circle radius (and centre offset) [m]. Must be > 0.
    speed : float
        Constant vehicle speed [m/s]. Must be > 0.
    dt : float
        Time step [s]. Default 1.0.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        Positions [m]. Column 0 = east, column 1 = north.
    times : ndarray, shape (N,)
        Time [s] starting at 0.

    Notes
    -----
    Total path length = 2 * 2π * A.  The trajectory starts and ends at
    approximately (0, 0).
    """
    if A <= 0:
        raise ValueError(f"A must be positive, got {A}")
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    omega = speed / A             # angular rate [rad/s]
    N_circle = max(2, round(2.0 * math.pi * A / (speed * dt)))
    # Use one more sample to include the endpoint before concatenation
    theta1 = np.linspace(0.0, 2.0 * math.pi, N_circle + 1)[:-1]
    # Circle 1: centre (-A, 0), counterclockwise; starts at (0, 0) → θ=0 → (A+(-A), 0)
    # Parametrise: point = centre + A * [cos(θ + π), sin(θ + π)] with θ from 0→2π
    # At θ=0: (-A + A*cos(π), A*sin(π)) = (0, 0) ✓
    east1 = -A + A * np.cos(theta1 + math.pi)
    north1 = A * np.sin(theta1 + math.pi)

    # Circle 2: centre (+A, 0), clockwise; starts at (0, 0) → θ=0 → (-A+A, 0)
    # Parametrise: point = centre + A * [cos(π - θ), sin(π - θ)]
    # At θ=0: (A + A*cos(π), A*sin(π)) = (0, 0) ✓
    theta2 = np.linspace(0.0, 2.0 * math.pi, N_circle + 1)[:-1]
    east2 = A + A * np.cos(math.pi - theta2)
    north2 = A * np.sin(math.pi - theta2)

    east = np.concatenate([east1, east2])
    north = np.concatenate([north1, north2])
    positions = np.column_stack([east, north])
    times = np.arange(len(positions)) * dt
    return positions, times


# ---------------------------------------------------------------------------
# IMU noise model
# ---------------------------------------------------------------------------


def imu_dead_reckoning(
    positions_true: np.ndarray,
    times: np.ndarray,
    sigma_v: float,
    sigma_a: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate IMU dead-reckoning with correlated velocity noise.

    Models two noise sources that are integrated to produce position drift:

    1. **White velocity noise** (``sigma_v``): independent at each time step,
       contributes a random walk in position.
    2. **Velocity random walk** from acceleration noise (``sigma_a``): the
       velocity bias drifts over time, producing position error that grows
       approximately as t^(3/2) — characteristic of real IMU behaviour.

    The noise dynamics per step (x and y axes independently):

    .. code-block:: text

        eta_a[k] ~ N(0, sigma_a^2 / dt)        # acceleration noise
        drift[k] = drift[k-1] + eta_a[k] * dt  # velocity bias accumulates
        eta_v[k] ~ N(0, sigma_v^2)             # white velocity noise
        v_noisy[k] = v_true[k] + eta_v[k] + drift[k]
        p_noisy[k] = p_noisy[k-1] + v_noisy[k] * dt

    Parameters
    ----------
    positions_true : ndarray, shape (N, 2)
        Ground-truth positions [m]. Column 0 = x (east), column 1 = y (north).
    times : ndarray, shape (N,)
        Time [s] at each sample. Must be monotonically increasing.
    sigma_v : float
        White velocity noise standard deviation [m/s]. Set to 0.0 to disable.
    sigma_a : float
        Acceleration noise spectral density [m/s^(3/2)].
        Drives a velocity random walk. Set to 0.0 for purely white noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    positions_noisy : ndarray, shape (N, 2)
        Dead-reckoning position estimate [m].
    position_error : ndarray, shape (N, 2)
        Error = positions_noisy - positions_true [m].

    Notes
    -----
    Position error standard deviation from drift alone:

    .. math::

        \\sigma_p(t) \\approx \\sigma_a \\sqrt{t^3 / 3}

    giving a drift rate at time T of :math:`\\sigma_a \\sqrt{T / 3}`.

    The first position is anchored to the true starting position; only
    subsequent steps accumulate error.
    """
    rng = np.random.default_rng(seed)
    N = len(positions_true)

    # True velocity from finite differences (forward difference at step 0)
    v_true = np.diff(positions_true, axis=0, prepend=positions_true[:1])
    dt_arr = np.diff(times, prepend=times[0])
    # Avoid division by zero for the prepended duplicate
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 1.0
    v_true = v_true / dt_arr[:, np.newaxis]

    positions_noisy = np.empty_like(positions_true)
    positions_noisy[0] = positions_true[0]

    drift = np.zeros(2)

    for k in range(1, N):
        dt_k = times[k] - times[k - 1]
        if dt_k <= 0:
            positions_noisy[k] = positions_noisy[k - 1]
            continue

        # Acceleration noise increment (per sqrt(Hz) convention)
        if sigma_a > 0:
            eta_a = rng.normal(0.0, sigma_a / math.sqrt(dt_k), size=2)
            drift += eta_a * dt_k

        eta_v = rng.normal(0.0, sigma_v, size=2) if sigma_v > 0 else np.zeros(2)

        v_noisy = v_true[k] + eta_v + drift
        positions_noisy[k] = positions_noisy[k - 1] + v_noisy * dt_k

    position_error = positions_noisy - positions_true
    return positions_noisy, position_error


# ---------------------------------------------------------------------------
# IMU calibration
# ---------------------------------------------------------------------------


def calibrate_sigma_a(
    target_drift_rate: float,
    eval_time: float = 3600.0,
    dt: float = 1.0,
    n_mc: int = 200,
    tol: float = 0.01,
    max_iter: int = 50,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Calibrate ``sigma_a`` so IMU position drift matches a target drift rate.

    Runs a binary search on ``sigma_a``. For each candidate value, ``n_mc``
    Monte Carlo dead-reckoning simulations are performed on a stationary
    truth trajectory (vehicle stays at the origin). The RMS position error
    at ``eval_time`` is compared to ``target_drift_rate * eval_time``; the
    bracket is tightened until the relative error is within ``tol``.

    Parameters
    ----------
    target_drift_rate : float
        Target RMS position error rate [m/s]. For example, 1 km/hour =
        1000 / 3600 ≈ 0.2778 m/s.
    eval_time : float
        Simulation duration over which drift is evaluated [s]. Default 3600 s
        (1 hour), matching the common "km/hour" specification.
    dt : float
        Time step [s]. Default 1.0.
    n_mc : int
        Monte Carlo runs per candidate ``sigma_a``. Default 200.
    tol : float
        Convergence tolerance on relative error |achieved - target| / target.
        Default 0.01 (1%).
    max_iter : int
        Maximum binary search iterations. Default 50.
    seed : int
        Base random seed. Run i uses seed + i for reproducibility.

    Returns
    -------
    sigma_a : float
        Calibrated acceleration noise [m/s^(3/2)].
    Q : ndarray, shape (2, 2)
        Process noise covariance matrix for use in the PCRB recursion::

            Q = sigma_a^2 * dt * I_2

        Derivation: the velocity drift adds variance ``sigma_a^2 / dt`` per
        second to velocity; over one time step dt the position increment
        variance is ``(sigma_a^2 / dt) * dt^2 = sigma_a^2 * dt`` per axis.

    Notes
    -----
    The analytic estimate used to initialise the bracket:

    .. math::

        \\sigma_a^{\\text{init}} = r \\sqrt{3 / T}

    is derived from :math:`\\sigma_p(T) = \\sigma_a \\sqrt{T^3/3} = r \\cdot T`,
    solving for :math:`\\sigma_a`. Binary search corrects for discrete-time
    bias relative to the continuous-time formula.
    """
    target_error = target_drift_rate * eval_time

    # Analytic initial estimate
    sigma_a_init = target_drift_rate * math.sqrt(3.0 / eval_time)

    # Stationary truth trajectory
    N = int(eval_time / dt) + 1
    positions_truth = np.zeros((N, 2))
    times = np.arange(N) * dt

    def _run_mc(sigma_a_candidate: float) -> float:
        """Return RMS position error at eval_time over n_mc runs."""
        errors_sq = []
        for i in range(n_mc):
            _, err = imu_dead_reckoning(
                positions_truth, times,
                sigma_v=0.0,
                sigma_a=sigma_a_candidate,
                seed=seed + i,
            )
            # Scalar position error at final step
            errors_sq.append(err[-1, 0] ** 2 + err[-1, 1] ** 2)
        return math.sqrt(float(np.mean(errors_sq)))

    lo = sigma_a_init / 10.0
    hi = sigma_a_init * 10.0

    sigma_a = sigma_a_init
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        achieved = _run_mc(mid)
        rel_err = abs(achieved - target_error) / target_error
        if rel_err < tol:
            sigma_a = mid
            break
        if achieved > target_error:
            hi = mid
        else:
            lo = mid
        sigma_a = mid

    Q = np.eye(2) * (sigma_a ** 2 * dt)
    return sigma_a, Q
