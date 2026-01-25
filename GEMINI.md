# GEMINI.md

This file provides a comprehensive guide for an AI assistant to understand and contribute to the "Geophysical Anomaly Navigation Fixability Study" project.

## Project Overview

This is a Python-based research project that aims to build a predictive model relating vehicle trajectories and geophysical map properties to achievable navigation accuracy. The central question is: "Given a specific path over a given terrain, how much can we improve our position estimate?"

The project involves:

1. **Synthetic Data Generation:** Creating artificial geophysical maps (e.g., terrain, magnetic fields) with controllable characteristics.
2. **Trajectory Simulation:** Modeling vehicle movement with different patterns and speeds.
3. **Filtering:** Applying advanced filtering techniques (UKF, RBPF) to simulate navigation using geophysical data.
4. **Performance Analysis:** Calculating theoretical performance bounds (PCRB, FIM) and empirical results.
5. **Machine Learning:** Training a model to predict navigation uncertainty reduction based on trajectory and map features.

The project is still in its early stages, with the foundational `mapping.py` module created. The `README.md` file contains a very detailed and ambitious plan for the entire project.

## Building and Running

The project uses Python. The main dependencies are listed in `pyproject.toml` and the `README.md`.

### Dependencies

To install the dependencies, you can use `pip`:

```bash
pip install -e .
```

The `README.md` also lists other libraries that will be needed as the project grows: `xgboost`, `scikit-learn`, `pandas`, `shap`, and `filterpy`.

### Running Simulations

The `README.md` provides examples of how to run the different components of the project. For example, to generate a synthetic map:

```python
from src.geo_fixability.mapping import generate_field_spectral

map_field = generate_field_spectral(
    shape=(512, 512),
    beta=2.0,
    correlation_length=30.0,
    anisotropy_ratio=2.0,
    angle=0.5
)
```

As the project develops, the main entry points for running simulations will likely be in the `experiments/` directory, as outlined in the `README.md`.

### Testing

The `README.md` mentions a `tests/` directory for unit tests, but it has not been created yet. A testing framework like `pytest` would be a good choice for this project.

## Development Conventions

* **Code Style:** The existing code in `src/geo_fixability/mapping.py` follows standard Python conventions (PEP 8).
* **Structure:** The project is structured as a Python package under the `src` directory. The `README.md` outlines a detailed modular structure that should be followed as new components are added.
* **Documentation:** The code should be well-documented with docstrings, as seen in `src/geo_fixability/mapping.py`.
* **Modularity:** The project is intended to be highly modular, with separate components for map generation, trajectory simulation, filtering, feature extraction, and machine learning. This modularity should be maintained as the project grows.
* **Git:** This is a git repository. Commits should be clear and descriptive.

## Key Files

* `README.md`: The primary source of information about the project's goals, methods, and structure. It is very detailed and should be consulted often.
* `pyproject.toml`: Defines the project's metadata and dependencies.
* `src/geo_fixability/__init__.py`: Makes the `geo_fixability` directory a Python package.
* `src/geo_fixability/mapping.py`: The first module, containing functions for generating synthetic maps.

## General git tools

You have acces to the basic git tool and commands as well as the gh tool for interacting with GitHub.