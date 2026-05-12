# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project develops a predictive model that relates vehicle trajectory characteristics and geophysical map properties to achievable navigation accuracy. The core question: "If I drive over this area at this speed and heading, how much will my navigation solution improve?"

The project uses:

- Synthetic scalar fields (terrain, magnetic, gravity) with controllable information content
- INS trajectory simulation with various speed/heading profiles
- RBPF (Rao-Blackwellized Particle Filter) with geophysical anomaly aiding
- Theoretical performance bounds (PCRB, Fisher Information)
- ML regression to predict: (trajectory, map) → uncertainty reduction

## Build and Development Commands

This project uses `uv` for package management with Python 3.13+.

```bash
# Install dependencies
uv sync

# Run Python with project dependencies
uv run python

# Run a script
uv run python path/to/script.py

# Add a dependency
uv add <package>
```

## Code Architecture

### Source Structure

All source code lives in `src/geo_fixability/`:

- `mapping.py` - Synthetic geophysical map generation with three methods:
    - `generate_field_spectral()` - Spectral synthesis with power-law spectrum (most control)
    - `generate_field_grf()` - Gaussian Random Fields with Matérn covariance
    - `generate_fbm()` - Fractional Brownian Motion

### Planned Modules (per README)

- `trajectory/` - 3-DOF kinematics, trajectory generation, INS simulation
- `filtering/` - RBPF implementation
- `bounds/` - Fisher Information Matrix and PCRB computation
- `features/` - Map and trajectory feature extraction
- `ml/` - Dataset generation, training, evaluation

## Code Conventions

- Python 3.13+ features
- Type hints for function signatures
- NumPy-style docstrings with units and ranges for physical parameters
- All generated fields normalized to zero mean and unit variance
- Use vectorized NumPy operations over loops
- Use FFT for spectral methods

## Key Domain Concepts

**State vector**: x = [px, py, vx, vy, heading]ᵀ

**Key map parameters**:

- `beta`: PSD exponent (0=white noise, 2=Brownian/terrain-like, 3=very smooth)
- `correlation_length`: Spatial feature scale in pixels
- `hurst`: Smoothness parameter (0.5=Brownian, <0.5=rough, >0.5=smooth)
- `anisotropy_ratio` and `angle`: Directional properties

**Performance metrics**:

- Fisher Information Matrix (FIM) - gradient-based instantaneous information
- Posterior Cramér-Rao Bound (PCRB) - theoretical lower bound on estimation covariance
- Empirical RMSE from Monte Carlo filter runs



## General git tools

You have acces to the basic `git` tool and commands as well as the `gh` tool for interacting with GitHub.