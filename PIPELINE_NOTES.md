# Pipeline Walkthrough

## The Scientific Question

The project asks: **given that a vehicle knows its starting position only to within some uncertainty, how much can a geophysical map reduce that uncertainty as the vehicle drives over it?**

The answer depends on six interacting factors:
- How uncertain the starting position is (`sigma_init`)
- How far and fast the vehicle travels (`speed`, `duration`)
- How badly the INS drifts between map measurements (`ins_drift_rate`)
- How noisy the sensor reading the map is (`sigma_meas`)
- How information-rich the map is (`beta`, `correlation_length`)
- How finely the map is resolved (`dx`)

The sweep runs all combinations of these and records the theoretical improvement ratio.

---

## The Pipeline, Stage by Stage

### Stage 1 — INS Calibration

```
calibrate_sigma_a(ins_drift_rate) → (sigma_a, Q)
```

The INS (Inertial Navigation System) drifts: it accumulates position error over time because accelerometer noise integrates twice into position. `calibrate_sigma_a` runs a binary search — it tries candidate noise levels, simulates 100 Monte Carlo dead-reckoning runs, and finds the `sigma_a` (acceleration noise spectral density, m/s^(3/2)) that produces the target drift rate (e.g. 1 km/hr error after 1 hour).

The result `Q` is the **per-step process noise covariance** (2×2, m²) used in the PCRB filter recursion. This gets pre-computed once per unique `(drift_rate, dt)` pair before the parallel workers launch — it's the slow part if done naïvely.

### Stage 2 — Map Generation

```
generate_field_spectral(shape, beta, correlation_length_px) → field (ny, nx)
```

A synthetic scalar map is created using spectral synthesis:
- A random Fourier spectrum is drawn, then shaped by a power-law filter `|f|^(-beta/2)` — higher `beta` → smoother, longer-wavelength features
- `correlation_length` controls the characteristic feature size; features smaller than this are suppressed
- The result is normalized to zero mean, unit variance

The map must be large enough to contain the trajectory. The auto-sizing logic is:

```
map_extent_m = max(2 × total_distance, 10 × correlation_length)
map_npix     = min(2048, ceil(map_extent_m / dx))   rounded to multiple of 64
```

The 2048-pixel cap prevents RAM explosions for extreme combinations (e.g. speed=20 m/s × duration=2400 s × dx=10 m/pixel would need a 9600-pixel field).

### Stage 3 — Map Metrics

```
map_information_content(field, dx, sigma_meas) → dict
```

Before running the PCRB, the map is characterized:
- **`gradient_rms`**: RMS of ∇h — how strongly the field changes per metre. Higher → more information per step.
- **`information_index`** [0,1]: normalized Shannon entropy of the power spectrum. 0 = single-frequency (useless), 1 = white noise (maximum spatial variety).
- **`localization_bound_m`**: `sigma_meas / gradient_rms` — the Cramér-Rao lower bound on position given a single measurement at a random location.
- **`localization_index`** [0,1]: log-scale summary of how sub-pixel the localization bound is.

These describe the map independently of the trajectory.

### Stage 4 — Trajectory Generation

```
star_trajectory(D, speed, dt) → (positions [N,2], times [N])
```

The default trajectory is an **8-armed star**: the vehicle drives from the center to each of 8 arm tips (at 45° intervals) and back. Arm length `D = 0.4 × map_extent_m`, so the trajectory fills ~80% of the map width.

The auto-selector switches to **lawnmower** when `total_distance > 2 × map_extent_m` — i.e. when a star trajectory would require driving so far that the map can't contain it meaningfully.

Why star by default? It gives uniform angular coverage and passes through many map features, which produces a well-conditioned Fisher Information Matrix.

### Stage 5 — PCRB (Posterior Cramér-Rao Bound)

```
pcrb_trajectory(positions, field, dx, sigma_meas, P_init, Q) → P_traj [N, 2, 2]
```

This is the theoretical heart of the pipeline. At each timestep:

1. **Predict**: `P_pred = P + Q` — uncertainty grows due to INS drift
2. **Measure**: read the map gradient at the true position → `H = [∂h/∂x, ∂h/∂y]`
3. **Update** (Joseph form for numerical stability):
   - Innovation variance: `S = H P_pred H^T + sigma_meas²`
   - Kalman gain: `K = P_pred H^T / S`
   - Posterior: `P = (I - KH) P_pred (I - KH)^T + K sigma_meas² K^T`

The result `P_traj[k]` is the **theoretical minimum covariance** achievable by any unbiased estimator at step `k`. It's a lower bound — a real filter (like the particle filter) will be equal to or worse than this.

### Stage 6 — Navigation Improvement Metrics

```
navigation_improvement(P_traj) → dict
```

From the covariance trajectory:
- **`initial_cep_m`**: Circular Error Probable at step 0 — the radius of the circle containing 50% probability, derived from `sigma_init`
- **`final_cep_m`**: CEP at the end of the trajectory
- **`cep_improvement_ratio`**: `initial_cep / final_cep` — the headline number. A ratio of 100 means the map reduced position uncertainty 100×.

CEP is computed as `sqrt(trace(P)/2) × sqrt(2 ln 2) ≈ 1.1774 × sigma_per_axis`.

### Stage 7 — Feature Extraction

```
extract_all_features(positions, times, field, dx, sigma_meas) → dict (~22 keys)
```

This builds the ML feature vector for each scenario — things like gradient RMS along the path, heading variance, alignment between heading and gradient direction, cumulative Fisher information trace, effective degrees of freedom (path length / correlation length). These are used later to train a regression model that can predict `cep_improvement_ratio` for new scenarios without running the full PCRB.

---

## The Parameter Sweep

### What's Being Swept

| Parameter | Values (full) | Meaning |
|---|---|---|
| `sigma_init` | 100, 500, 1000, 5000 m | Starting uncertainty — anything from GPS-denied dead reckoning (100 m) to catastrophic loss of fix (5 km) |
| `speed` | 2, 5, 10, 20 m/s | Vehicle speed — slow UGV to fast aircraft |
| `duration` | 300, 600, 1200, 2400 s | Mission length — 5 min to 40 min |
| `ins_drift_rate` | 0.1, 0.5, 1.0 km/hr | INS quality — tactical-grade to MEMS |
| `sigma_meas` | 0.01, 0.05, 0.1, 0.5 | Sensor noise relative to map std — precise to noisy |
| `beta` | 0.7, 1.3, 2.0, 3.5 | Map roughness — β=0.7 is nearly white noise, β=3.5 is very smooth |
| `correlation_length` | 300, 1000, 2500, 4500 m | Feature scale — fine to coarse |
| `dx` | 10, 25, 50, 100, 250, 500 m/px | Map resolution — high-res survey to coarse atlas |

Full grid: 4×4×4×3×4×4×4×6 = **18,432 scenarios**. The `--quick` flag takes the first 2 values per dimension → 256 scenarios.

### How the Workers Are Organized

```
main process:
  load config → build 18,432 ScenarioParams
  pre-compute calibration cache (3 unique drift rates × 1 dt = 3 binary searches)
  dispatch all scenarios to ProcessPoolExecutor

each worker process:
  receive (idx, params_dict, calibration_cache)
  look up (sigma_a, Q) from cache — no recalibration
  call run_scenario()
  return (idx, result_dict)

main process:
  collect results in original order
  write CSV
  generate summary plots
```

The calibration cache is the key efficiency trick: `calibrate_sigma_a` takes ~5 seconds each, and there are only 3 unique drift rates in the full sweep. Without caching, 18,432 scenarios × 5 s = 25 hours just for calibration. With caching, it's 3 × 5 s = 15 seconds.

### Output CSV

Each row is one scenario. The 35 columns include:
- All 8 input parameters
- Derived geometry (map size, trajectory type used, sigma_a)
- Map characterization (gradient_rms, information_index, etc.)
- PCRB results (`initial_cep_m`, `final_cep_m`, `cep_improvement_ratio`)
- Trajectory features for ML (`effective_dof`, `cumulative_fim_trace`, etc.)
- Status (`ok`, `warn_oob` if >5% of trajectory steps fell outside the map, `error: ...` on exception)

---

## Running It

```bash
# Quick test: 256 scenarios, ~20 seconds, 4 workers
uv run run-sweep --quick --workers 4

# Full sweep: 18,432 scenarios
uv run run-sweep --workers 8 --outdir data/sweep

# With particle filter (much slower — adds empirical PF comparison):
uv run run-sweep --quick --run-pf --n-particles 200

# Custom config or output directory:
uv run run-sweep --config sweep_config.yaml --outdir data/my_experiment

# Single scenario in Python:
from geo_fixability.pipeline import run_scenario, ScenarioParams
r = run_scenario(ScenarioParams(
    sigma_init=1000, speed=10, duration=600,
    ins_drift_rate=0.27778, sigma_meas=0.1,
    beta=2.0, correlation_length=1000, dx=50,
))
print(r['cep_improvement_ratio'])   # ~1677×
```

After the sweep CSV exists, the existing `analyze-results` CLI trains an XGBoost model on it and generates SHAP importance plots — that's the next step in the pipeline (currently in `run_pipeline.sh` step 5).
