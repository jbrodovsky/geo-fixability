# Geophysical Anomaly Navigation Fixability Study

## Project Overview

This project develops a predictive model that relates vehicle trajectory characteristics and geophysical map properties to achievable navigation accuracy. The goal is to answer: **"If I drive over this area at this speed and heading, how much will my navigation solution improve?"**

### Objectives

1. Generate synthetic scalar fields (terrain, magnetic, gravity) with controllable information content and define how we classify that information content
2. Simulate INS trajectories with various speed/heading profiles
3. Implement RBPF filter with geophysical anomaly aiding
4. Compute theoretical performance bounds (PCRB, Fisher Information)
5. Extract trajectory and map features predictive of performance
6. Build ML regression model: (trajectory, map) → uncertainty reduction
7. Validate on real datasets (SRTM15+, WDMAM, KITTI)

### Scope

- **Generalized problem**: Zero-mean Gaussian measurement noise on spatially-correlated scalar fields
- **Platform**: 3-DOF kinematic vehicle on Cartesian plane
- **Filters**: Rao-Blackwellized Particle Filter
- **Output**: Journal paper + dissertation chapter

## Theoretical Background

### Problem Formulation

**State**: x = [px, py, v, heading]ᵀ

**Dynamics**: 

- Dead reckoning with constant velocity
- Error growth: σ_INS(t) = σ_0 + drift_rate × t

**Measurement Model**:

- z_k = h(x_k) + v_k
- h(x_k) = map_value(px, py) 
- v_k ~ N(0, σ_meas²)

**Map**: Spatially-correlated scalar field M(x,y)

### Performance Metrics

1. **Fisher Information Matrix (FIM)**:

   - FIM = E[∇h(x)ᵀ R⁻¹ ∇h(x)]
   - Measures instantaneous information from measurement
   - Gradient-based: strong gradients = more information
1. **Posterior Cramér-Rao Bound (PCRB)**:

   - Theoretical lower bound on estimation covariance
   - Accounts for sequential measurements and dynamics
   - Recursion: J_k = [D_{k-1} - F_{k-1}ᵀ(J_{k-1} + G_{k-1})⁻¹F_{k-1}]⁻¹ + H_kᵀR_k⁻¹H_k
   - PCRB⁻¹ = achievable covariance
1. **Empirical RMSE**:

   - Monte Carlo filter runs
   - Final position error vs PCRB (filter efficiency)
   - "Fixability" metric should be something along the lines of a position measurement sensor configuration: given my trajectory parameters, I know that I am here with +/- x precision.

## Repository Structure

```
geophysical-fixability/
├── README.md (this file)
├── requirements.txt
├── src/
│   ├── map_generation/
│   │   ├── spectral_synthesis.py      # Generate synthetic fields
│   │   ├── grf_generator.py           # Gaussian Random Fields
│   │   ├── fbm_generator.py           # Fractional Brownian Motion
│   │   └── real_data_loader.py        # Load SRTM15+, WDMAM, etc.
│   ├── trajectory/
│   │   ├── kinematics.py              # 3-DOF dynamics
│   │   ├── trajectory_generator.py    # Pattern generation
│   │   └── ins_simulator.py           # Dead reckoning with drift
│   ├── filtering/
│   │   ├── rbpf.py                    # Rao-Blackwellized PF
│   │   └── measurement_model.py       # Map interpolation
│   ├── bounds/
│   │   ├── fisher_information.py      # FIM computation
│   │   ├── pcrb.py                    # PCRB recursion
│   │   └── observability.py           # Gramian analysis
│   ├── features/
│   │   ├── map_features.py            # Spectral, gradient, entropy
│   │   ├── trajectory_features.py     # Speed, heading, sampling
│   │   └── combined_features.py       # Trajectory-map interaction
│   ├── ml/
│   │   ├── dataset_generation.py      # Monte Carlo sim runner
│   │   ├── train.py                   # XGBoost/LightGBM training
│   │   ├── evaluate.py                # Cross-validation, metrics
│   │   └── feature_importance.py      # SHAP analysis
│   └── visualization/
│       ├── plot_maps.py
│       ├── plot_trajectories.py
│       └── plot_results.py
├── experiments/
│   ├── synthetic_sweep.py             # Parameter sweep on synthetic data
│   ├── real_data_validation.py        # SRTM15+, WDMAM, KITTI
│   └── ablation_studies.py            # Feature/filter comparisons
├── data/
│   ├── synthetic/                     # Generated maps
│   ├── real/                          # Downloaded datasets
│   └── results/                       # Simulation outputs
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── paper_figures.ipynb
└── tests/
    └── unit tests for each module
```

## Data Generation

### Synthetic Map Parameters

Generate scalar fields with controllable information content using spectral synthesis:

**Primary parameters:**

- `beta`: Power spectral density exponent [0, 3]
    - 0 = white noise (uncorrelated)
    - 1 = pink noise (natural)
    - 2 = Brownian (terrain-like)
    - 3 = very smooth
- `correlation_length`: Spatial correlation [5, 100] pixels
- `anisotropy_ratio`: Elongation factor [1, 10]
- `anisotropy_angle`: Feature orientation [0, 2π]
- `dynamic_range`: Amplitude relative to noise [0.1, 10] × σ_meas

**Secondary parameters:**

- `hurst_exponent`: For fractional Brownian motion [0.1, 0.9]
- `sparsity`: Fraction of active vs flat regions
- `feature_size_distribution`: Multi-scale content

**Implementation**:

```python
field = generate_field_spectral(
    shape=(1024, 1024),
    beta=2.0,
    correlation_length=20.0,
    anisotropy_ratio=3.0,
    angle=np.pi/4,
    dynamic_range=5.0
)
```

### Trajectory Generation

**Patterns to test:**

- Straight line (various headings)
- Grid pattern (lawn mower)
- Spiral
- Random walk
- Figure-8
- Obstacle avoidance path

**Parameters:**

- Initial position: [x0, y0]
- Initial velocity: [vx0, vy0]
- Acceleration commands: [ax(t), ay(t)]
- Duration: T seconds
- Update rate: Δt (e.g., 10 Hz)

**Speed profiles:**

- Constant: [0.5, 1, 5, 10, 20, 50] m/s
- Variable: acceleration/deceleration phases

**Implementation**:

```python
trajectory = generate_trajectory(
    pattern='grid',
    speed=10.0,  # m/s
    duration=300.0,  # seconds
    grid_spacing=50.0,  # meters
    update_rate=10.0  # Hz
)
```

## Filter Implementation

### RBPF Configuration

**Particle filter** for position (px, py)

**Kalman filter** for velocity conditioned on position

**Particles**: N = [100, 500, 1000]

**Resampling**: Systematic, when N_eff < N/2

## Feature Extraction

### Map Features (along trajectory)

**Spatial statistics:**

```python
- power_spectral_density(map_along_trajectory)
- correlation_length_estimate(map)
- fractal_dimension(map)
- semivariogram(map)
```

**Gradient features:**

```python
- gradient_magnitude_mean
- gradient_magnitude_std
- gradient_perpendicular_to_velocity  # Cross-track
- gradient_parallel_to_velocity       # Along-track
- observability_ratio = cross_track / along_track
```

**Information content:**

```python
- local_entropy(map, window_size=20)
- local_variance(map, window_size=20)
- feature_density (zero crossings, peaks)
- ridge_crossing_count
- dominant_spatial_frequency
```

**Uniqueness:**

```python
- normalized_cross_correlation(local_region, full_map)
- local_autocorrelation
- distance_to_nearest_similar_value
```

### Trajectory Features

**Kinematics:**

```python
- mean_speed, std_speed
- total_distance
- trajectory_duration
- heading_variance
- heading_change_count (> threshold)
- curvature_statistics
```

**Measurement characteristics:**

```python
- total_measurement_count
- measurement_density (measurements/km)
- effective_dof = trajectory_length / correlation_length
- nyquist_ratio = (speed/update_rate) / (2 * spatial_frequency)
```

**Observability proxies:**

```python
- cumulative_fisher_information
- heading_diversity_index
- cross_track_gradient_integral
- ridge_crossings_at_angle (non-parallel)
```

### Combined Features

**Trajectory-map interaction:**

```python
- SNR_effective = dynamic_range / (σ_meas * sqrt(speed))
- information_accumulation_rate
- gradient_alignment_statistics
- spatial_sampling_adequacy
```

## Analytical Bounds

### Fisher Information Computation

```python
def compute_fisher_information(position, map_data, sigma_meas):
    """
    FIM at single measurement
    """
    gradient = compute_gradient(map_data, position)  # [∂h/∂x, ∂h/∂y]
    H = gradient.reshape(1, -1)  # Measurement Jacobian
    R_inv = 1 / sigma_meas**2
    FIM = H.T @ R_inv @ H
    return FIM

def accumulated_fisher_information(trajectory, map_data, sigma_meas):
    """
    Cumulative FIM along trajectory
    """
    FIM_total = np.zeros((2, 2))
    for position in trajectory:
        FIM_total += compute_fisher_information(position, map_data, sigma_meas)
    return FIM_total
```

### PCRB Recursion

```python
def compute_pcrb(trajectory, map_data, dynamics_params, meas_params):
    """
    Posterior Cramér-Rao Bound for entire trajectory
    
    Returns: covariance_bound (array of shape [N, state_dim, state_dim])
    """
    N = len(trajectory)
    state_dim = 5  # [px, py, vx, vy, heading]
    
    J = np.zeros((N, state_dim, state_dim))
    J[0] = np.linalg.inv(initial_covariance)  # Prior information
    
    for k in range(1, N):
        # Prediction step
        F_k = compute_dynamics_jacobian(trajectory[k-1], dynamics_params)
        Q_k = process_noise_covariance(dynamics_params)
        D_k = Q_k  # Simplification
        
        J_pred = np.linalg.inv(
            F_k @ np.linalg.inv(J[k-1]) @ F_k.T + Q_k
        )
        
        # Update step
        H_k = compute_measurement_jacobian(trajectory[k], map_data)
        R_k = meas_params['sigma_meas']**2
        
        J[k] = J_pred + H_k.T @ (1/R_k) @ H_k
    
    # Convert information to covariance
    covariance_bounds = np.array([np.linalg.inv(J[k]) for k in range(N)])
    
    return covariance_bounds
```

## ML Pipeline

### Dataset Generation

Run Monte Carlo simulations:

```python
# Pseudocode
n_maps = 1000
n_trajectories_per_map = 50

dataset = []
for map_params in parameter_sweep:
    map_field = generate_field_spectral(**map_params)
    
    for traj_params in trajectory_variations:
        trajectory = generate_trajectory(**traj_params)
        
        # Run filter
        rbpf_result = run_rbpf(trajectory, map_field, filter_params)

        # Compute bounds
        pcrb = compute_pcrb(trajectory, map_field, ...)
        fim_total = accumulated_fisher_information(...)

        # Extract features
        map_features = extract_map_features(map_field, trajectory)
        traj_features = extract_trajectory_features(trajectory)

        # Record
        dataset.append({
            'features': {**map_features, **traj_features},
            'final_rmse': rbpf_result['final_position_rmse'],
            'final_covariance': rbpf_result['final_covariance'],
            'pcrb_bound': pcrb[-1],
            'fim_total': fim_total,
            'map_params': map_params,
            'traj_params': traj_params
        })
```

### Training

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Prepare data
X = np.array([d['features'] for d in dataset])
y_rmse = np.array([d['final_rmse'] for d in dataset])
y_cov = np.array([d['final_covariance'] for d in dataset])  # Matrix output

# Train regressor
model_rmse = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8
)
model_rmse.fit(X, y_rmse)

# Feature importance
import shap
explainer = shap.TreeExplainer(model_rmse)
shap_values = explainer.shap_values(X)
```

### Evaluation Metrics

1. **R² score**: Model fit quality
2. **RMSE prediction error**: |predicted - actual|
3. **PCRB comparison**: How close is empirical RMSE to theoretical bound?
4. **Feature importance**: Which features matter most?
5. **Generalization**: Performance on real datasets (SRTM15+, WDMAM, KITTI)

## Experimental Plan

### Phase 1: Synthetic Data

**Parameter sweep:**

- β: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- correlation_length: [10, 20, 50, 100] pixels
- anisotropy_ratio: [1, 2, 5]
- speed: [1, 5, 10, 20, 50] m/s
- trajectory_pattern: [straight, grid, spiral, random]

**Total combinations**: ~1000 scenarios × 50 MC runs = 50k simulations

### Phase 2: Analytical Validation

- Compute PCRB for all scenarios
- Compare filter RMSE to PCRB (efficiency gap)
- Identify cases where filter is suboptimal

### Phase 3: ML Model

- Train on synthetic data
- Ablation: which features are necessary?
- Interpretability: SHAP analysis

### Phase 4: Real Data

- Download SRTM15+ (terrain)
- Download WDMAM (magnetic)
- KITTI trajectories (if applicable)
- Extract regions, run filters, validate predictions

## Usage Examples

### Generate synthetic map

```python
from src.map_generation import generate_field_spectral

map_field = generate_field_spectral(
    shape=(512, 512),
    beta=2.0,
    correlation_length=30.0,
    anisotropy_ratio=2.0,
    angle=0.5
)
```

### Generate trajectory

```python
from src.trajectory import generate_trajectory

traj = generate_trajectory(
    pattern='grid',
    speed=10.0,
    duration=200.0,
    bounds=(0, 512, 0, 512)
)
```

### Compute PCRB

```python
from src.bounds import compute_pcrb

pcrb_cov = compute_pcrb(
    trajectory=traj,
    map_field=map_field,
    dynamics_params={'sigma_accel': 0.01},
    meas_params={'sigma_meas': 1.0}
)

theoretical_rmse = np.sqrt(np.trace(pcrb_cov[-1, :2, :2]))
print(f"PCRB lower bound: {theoretical_rmse:.2f} m")
```

### Extract features

```python
from src.features import extract_all_features

features = extract_all_features(
    trajectory=traj,
    map_field=map_field,
    sigma_meas=1.0
)
# Returns: dict with ~30-50 features
```

### Train ML model

```python
from src.ml import generate_dataset, train_model

dataset = generate_dataset(n_scenarios=1000, n_mc_runs=50)
model = train_model(dataset, target='final_rmse')
model.save('fixability_predictor.json')
```

## Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
xgboost>=2.0
scikit-learn>=1.3
pandas>=2.0
shap>=0.42
tqdm  # Progress bars
```

## Key Design Decisions

1. **Cartesian plane**: Simplifies to 3-DOF, avoids Earth curvature complexity
2. **Zero-mean Gaussian measurement**: Generalizes across anomaly types
3. **Spectral synthesis**: Fast, controllable, physically realistic
4. **PCRB baseline**: Establishes theoretical best-case performance
5. **XGBoost**: Handles nonlinearity, feature importance, fast training
6. **Monte Carlo validation**: Empirical verification of predictions

## Open Questions for Agents

1. Should we model correlated process noise (INS drift)?
2. Multi-output regression (predict full covariance matrix) or separate models (RMSE, trace, determinant)?
3. How to handle particle filter degeneracy in PCRB computation?
4. Should anisotropy angle be relative to trajectory or absolute?
5. Best way to characterize "ridge crossings" - zero crossings of raw field or gradient?

## Success Criteria

- **Analytical**: PCRB computed correctly, matches simplified cases
- **Empirical**: Filter RMSE within 20% of PCRB on average
- **ML**: R² > 0.85 on test set, generalizes to real data
- **Paper**: Clear relationship between (β, λ, speed, heading) and achievable accuracy

## References

- Tichavsky et al. (2005): "Posterior Cramér-Rao Bounds for Discrete-Time Nonlinear Filtering"
- Van Trees (2001): "Detection, Estimation, and Modulation Theory"
- Mandic & Chambers (2001): "Recurrent Neural Networks for Prediction"
- Your previous work on particle filter navigation

---

**Last updated**: [Date]

**Contact**: James ([email])