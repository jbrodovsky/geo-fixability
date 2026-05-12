"""
Core unit tests for geo-fixability modules.

All tests use small arrays (64×64 fields, 50-step trajectories) and complete
in under 30 seconds total.
"""

import math

import numpy as np
import pytest

from geo_fixability.features import (
    trajectory_features,
    trajectory_map_features,
    extract_all_features,
)
from geo_fixability.mapping import generate_field_spectral
from geo_fixability.trajectory import (
    star_trajectory,
    straight_trajectory,
    lawnmower_trajectory,
    spiral_trajectory,
    random_walk_trajectory,
    figure8_trajectory,
    imu_dead_reckoning,
)
from geo_fixability.navigation import (
    pcrb_trajectory,
    navigation_improvement,
    compute_field_gradients,
)
from geo_fixability.filtering import particle_filter
from geo_fixability.metrics import map_information_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_field(seed=42):
    """64×64 synthetic spectral field."""
    np.random.seed(seed)
    return generate_field_spectral((64, 64), beta=2.0, correlation_length=5.0)


def _tiny_star(D=500.0, speed=5.0, dt=10.0):
    """Short star trajectory that stays well within a 64×64 field at dx=30."""
    return star_trajectory(D=D, speed=speed, dt=dt, n_arms=8)


def _pcrb_inputs(field, dx=30.0):
    """Default PCRB inputs for a 64×64 field at 30 m/pixel."""
    P_init = np.eye(2) * 500.0 ** 2
    Q = np.eye(2) * 1.0
    return P_init, Q, dx


# ---------------------------------------------------------------------------
# mapping.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 3.0])
def test_spectral_field_normalization(beta):
    np.random.seed(42)
    field = generate_field_spectral((64, 64), beta=beta, correlation_length=5.0)
    assert abs(field.mean()) < 0.05
    assert abs(field.std() - 1.0) < 0.05


def test_spectral_field_shape():
    np.random.seed(42)
    field = generate_field_spectral((32, 48), beta=2.0, correlation_length=4.0)
    assert field.shape == (32, 48)


# ---------------------------------------------------------------------------
# trajectory.py
# ---------------------------------------------------------------------------

def test_star_trajectory_origin():
    pos, _ = star_trajectory(D=500.0, speed=5.0, dt=1.0, n_arms=8)
    assert np.allclose(pos[0], [0.0, 0.0], atol=1e-9)
    assert np.allclose(pos[-1], [0.0, 0.0], atol=1e-6)


def test_star_trajectory_distance():
    D, n_arms = 500.0, 8
    pos, _ = star_trajectory(D=D, speed=5.0, dt=1.0, n_arms=n_arms)
    diffs = np.diff(pos, axis=0)
    total_dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
    expected = 2 * n_arms * D
    assert abs(total_dist - expected) / expected < 0.01


def test_star_trajectory_times_monotonic():
    _, times = star_trajectory(D=500.0, speed=5.0, dt=1.0)
    assert np.all(np.diff(times) > 0)


def test_imu_dead_reckoning_shape():
    pos, times = _tiny_star()
    sigma_v, sigma_a = 0.01, 0.001
    noisy, err = imu_dead_reckoning(pos, times, sigma_v, sigma_a, seed=0)
    assert noisy.shape == pos.shape
    assert err.shape == pos.shape


def test_imu_dead_reckoning_anchored():
    pos, times = _tiny_star()
    noisy, err = imu_dead_reckoning(pos, times, 0.01, 0.001, seed=0)
    # First position is anchored to truth
    assert np.allclose(err[0], [0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# navigation.py — PCRB
# ---------------------------------------------------------------------------

def test_pcrb_shape():
    field = _small_field()
    pos, times = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    P_traj = pcrb_trajectory(pos, field, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    assert P_traj.shape == (len(pos), 2, 2)


def test_pcrb_first_equals_P_init():
    field = _small_field()
    pos, times = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    P_traj = pcrb_trajectory(pos, field, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    assert np.allclose(P_traj[0], P_init)


def test_pcrb_improving():
    """Uncertainty should decrease along a star trajectory over a non-flat field."""
    field = _small_field()
    pos, times = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    P_traj = pcrb_trajectory(pos, field, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    trace_init = float(np.trace(P_traj[0]))
    trace_final = float(np.trace(P_traj[-1]))
    assert trace_final < trace_init


def test_pcrb_flat_field_no_improvement():
    """On a zero field (no gradient) PCRB should not improve below P_init."""
    flat = np.zeros((64, 64))
    pos, times = _tiny_star()
    P_init = np.eye(2) * 500.0 ** 2
    Q = np.eye(2) * 1.0
    dx = 30.0
    P_traj = pcrb_trajectory(pos, flat, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    # Final trace ≥ initial trace (no information to reduce uncertainty)
    assert np.trace(P_traj[-1]) >= np.trace(P_traj[0]) - 1e-6


# ---------------------------------------------------------------------------
# navigation.py — navigation_improvement
# ---------------------------------------------------------------------------

_EXPECTED_IMPROVEMENT_KEYS = {
    "pos_uncertainty",
    "initial_uncertainty_m",
    "final_uncertainty_m",
    "improvement_ratio",
    "reduction_fraction",
    "cep_initial_m",
    "cep_final_m",
    "cep_improvement_ratio",
}


def test_navigation_improvement_keys():
    field = _small_field()
    pos, _ = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    P_traj = pcrb_trajectory(pos, field, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    result = navigation_improvement(P_traj)
    assert _EXPECTED_IMPROVEMENT_KEYS == set(result.keys())


def test_navigation_improvement_ratio_ge_1():
    field = _small_field()
    pos, _ = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    P_traj = pcrb_trajectory(pos, field, dx, sigma_meas=1.0, P_init=P_init, Q=Q)
    result = navigation_improvement(P_traj)
    assert result["improvement_ratio"] >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# filtering.py — particle_filter
# ---------------------------------------------------------------------------

_EXPECTED_PF_KEYS = {"P_trajectory", "mean_trajectory", "pos_uncertainty", "n_eff"}


def test_particle_filter_output_keys():
    np.random.seed(0)
    field = _small_field()
    pos, times = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    result = particle_filter(
        pos, times, field, dx,
        sigma_meas=1.0, P_init=P_init, Q=Q,
        n_particles=200, seed=0,
    )
    assert _EXPECTED_PF_KEYS == set(result.keys())


def test_particle_filter_output_shapes():
    np.random.seed(0)
    field = _small_field()
    pos, times = _tiny_star()
    N = len(pos)
    P_init, Q, dx = _pcrb_inputs(field)
    result = particle_filter(
        pos, times, field, dx,
        sigma_meas=1.0, P_init=P_init, Q=Q,
        n_particles=200, seed=0,
    )
    assert result["P_trajectory"].shape == (N, 2, 2)
    assert result["mean_trajectory"].shape == (N, 2)
    assert result["pos_uncertainty"].shape == (N,)
    assert result["n_eff"].shape == (N,)


def test_particle_filter_uncertainty_bounded():
    """PF position uncertainty should decrease over a non-flat field."""
    np.random.seed(1)
    field = _small_field()
    pos, times = _tiny_star()
    P_init, Q, dx = _pcrb_inputs(field)
    result = particle_filter(
        pos, times, field, dx,
        sigma_meas=1.0, P_init=P_init, Q=Q,
        n_particles=500, seed=1,
    )
    assert result["pos_uncertainty"][-1] < result["pos_uncertainty"][0]


# ---------------------------------------------------------------------------
# metrics.py — map_information_content
# ---------------------------------------------------------------------------

_EXPECTED_METRIC_KEYS = {
    "information_index",
    "gradient_rms",
    "correlation_length_px",
    "correlation_length_m",
    "localization_bound_px",
    "localization_bound_m",
    "spatial_diversity_index",
    "localization_index",
}


def test_map_info_content_keys():
    field = _small_field()
    result = map_information_content(field, dx=30.0, sigma_meas=1.0)
    assert _EXPECTED_METRIC_KEYS == set(result.keys())


def test_map_info_content_values_finite():
    field = _small_field()
    result = map_information_content(field, dx=30.0, sigma_meas=1.0)
    for key, val in result.items():
        if key not in ("localization_bound_px", "localization_bound_m"):
            assert math.isfinite(val), f"{key} = {val} is not finite"


def test_map_info_content_flat_field():
    flat = np.zeros((64, 64))
    result = map_information_content(flat, dx=30.0, sigma_meas=1.0)
    assert result["information_index"] == pytest.approx(0.0, abs=1e-9)
    assert result["localization_index"] == pytest.approx(0.0, abs=1e-9)
    assert result["gradient_rms"] == pytest.approx(0.0, abs=1e-9)


def test_map_info_content_nonzero_for_spectral_field():
    field = _small_field()
    result = map_information_content(field, dx=30.0, sigma_meas=1.0)
    assert result["information_index"] > 0.0
    assert result["gradient_rms"] > 0.0
    assert result["localization_index"] > 0.0


# ---------------------------------------------------------------------------
# trajectory.py — new patterns
# ---------------------------------------------------------------------------

def test_straight_trajectory_heading_north():
    pos, times = straight_trajectory(D=1000.0, speed=5.0, heading_deg=0.0, dt=1.0)
    # Heading north: east ≈ 0, north > 0
    assert np.allclose(pos[:, 0], 0.0, atol=1e-9)
    assert pos[-1, 1] == pytest.approx(1000.0, rel=0.01)


def test_straight_trajectory_heading_east():
    pos, times = straight_trajectory(D=500.0, speed=10.0, heading_deg=90.0, dt=1.0)
    # Heading east: north ≈ 0, east > 0
    assert np.allclose(pos[:, 1], 0.0, atol=1e-9)
    assert pos[-1, 0] == pytest.approx(500.0, rel=0.01)


def test_straight_trajectory_total_distance():
    D = 800.0
    pos, _ = straight_trajectory(D=D, speed=4.0, heading_deg=45.0, dt=1.0)
    dist = float(np.linalg.norm(pos[-1]))
    assert abs(dist - D) / D < 0.01


def test_lawnmower_bounding_box():
    W, H = 1000.0, 800.0
    pos, _ = lawnmower_trajectory(width=W, height=H, lane_spacing=100.0, speed=5.0)
    assert pos[:, 0].min() >= -W / 2 - 1.0
    assert pos[:, 0].max() <= W / 2 + 1.0
    assert pos[:, 1].min() >= -H / 2 - 1.0
    assert pos[:, 1].max() <= H / 2 + 1.0


def test_lawnmower_shape():
    pos, times = lawnmower_trajectory(1000.0, 500.0, 100.0, 5.0, dt=1.0)
    assert pos.shape[1] == 2
    assert len(times) == len(pos)


def test_spiral_max_radius():
    R_max = 800.0
    pos, _ = spiral_trajectory(R_max=R_max, n_turns=3, speed=5.0, dt=1.0)
    assert float(np.max(np.linalg.norm(pos, axis=1))) <= R_max * 1.05


def test_spiral_starts_at_origin():
    pos, _ = spiral_trajectory(800.0, 2, 5.0, dt=1.0)
    assert np.allclose(pos[0], [0.0, 0.0], atol=1e-6)


def test_random_walk_total_distance():
    D_total = 500.0
    pos, _ = random_walk_trajectory(D_total=D_total, step_scale=50.0, speed=5.0, dt=1.0, seed=7)
    diffs = np.diff(pos, axis=0)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    assert abs(path_len - D_total) / D_total < 0.02


def test_figure8_returns_to_origin():
    A, speed, dt = 300.0, 5.0, 1.0
    pos, _ = figure8_trajectory(A=A, speed=speed, dt=dt)
    # Last point should be close to (0, 0)
    assert float(np.linalg.norm(pos[-1])) < speed * dt * 3


def test_figure8_shape():
    pos, times = figure8_trajectory(A=500.0, speed=5.0, dt=1.0)
    assert pos.shape[1] == 2
    assert len(times) == len(pos)


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

_EXPECTED_TRAJ_FEAT_KEYS = {
    "mean_speed_m_s", "total_distance_m", "trajectory_duration_s",
    "heading_variance_rad2", "heading_change_count",
    "measurement_density_per_km", "effective_dof",
}

_EXPECTED_INTERACTION_KEYS = {
    "gradient_rms_along_path", "gradient_parallel_mean",
    "gradient_perpendicular_mean", "gradient_alignment_variance",
    "cumulative_fim_trace", "localization_bound_along_path_m", "fraction_oob",
}


def test_trajectory_features_star():
    pos, times = _tiny_star()
    result = trajectory_features(pos, times)
    assert _EXPECTED_TRAJ_FEAT_KEYS == set(result.keys())
    assert result["total_distance_m"] > 0.0
    assert result["mean_speed_m_s"] > 0.0


def test_trajectory_features_effective_dof():
    pos, times = _tiny_star()
    result = trajectory_features(pos, times, correlation_length_m=100.0)
    assert math.isfinite(result["effective_dof"])
    assert result["effective_dof"] > 0.0


def test_trajectory_map_features_keys():
    field = _small_field()
    pos, times = _tiny_star()
    result = trajectory_map_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    assert _EXPECTED_INTERACTION_KEYS == set(result.keys())


def test_trajectory_map_features_finite():
    field = _small_field()
    pos, times = _tiny_star()
    result = trajectory_map_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    for key, val in result.items():
        if key != "localization_bound_along_path_m":
            assert math.isfinite(val), f"{key} = {val}"


def test_cumulative_fim_positive():
    field = _small_field()
    pos, times = _tiny_star()
    result = trajectory_map_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    assert result["cumulative_fim_trace"] > 0.0


def test_fraction_oob_zero_for_star_inside_field():
    """Star trajectory with D=500m at dx=30 stays inside a 64×64 field
    (half-width = 64*30/2 = 960 m > 500 m)."""
    field = _small_field()
    pos, times = _tiny_star(D=500.0)
    result = trajectory_map_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    assert result["fraction_oob"] == pytest.approx(0.0, abs=1e-9)


def test_extract_all_features_count_and_dx():
    field = _small_field()
    pos, times = _tiny_star()
    result = extract_all_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    assert len(result) >= 20
    assert result["dx_m"] == pytest.approx(30.0)


def test_extract_all_features_all_finite():
    field = _small_field()
    pos, times = _tiny_star()
    result = extract_all_features(pos, times, field, dx=30.0, sigma_meas=1.0)
    non_finite_allowed = {"localization_bound_along_path_m", "localization_bound_px",
                          "localization_bound_m", "effective_dof"}
    for key, val in result.items():
        if key not in non_finite_allowed:
            assert math.isfinite(val), f"{key} = {val}"


def test_all_trajectories_compatible_with_pcrb():
    """All 6 patterns can be passed to pcrb_trajectory on a 128×128 field."""
    np.random.seed(0)
    field = generate_field_spectral((128, 128), beta=2.0, correlation_length=8.0)
    dx = 30.0
    P_init = np.eye(2) * 500.0 ** 2
    Q = np.eye(2) * 1.0
    sigma_meas = 1.0

    patterns = [
        star_trajectory(D=400.0, speed=5.0, dt=10.0),
        straight_trajectory(D=800.0, speed=5.0, heading_deg=30.0, dt=10.0),
        lawnmower_trajectory(800.0, 600.0, 150.0, 5.0, dt=10.0),
        spiral_trajectory(500.0, 2, 5.0, dt=10.0),
        random_walk_trajectory(800.0, 80.0, 5.0, dt=10.0, seed=0),
        figure8_trajectory(300.0, 5.0, dt=10.0),
    ]
    for pos, times in patterns:
        P_traj = pcrb_trajectory(pos, field, dx, sigma_meas, P_init, Q)
        assert P_traj.shape == (len(pos), 2, 2)
