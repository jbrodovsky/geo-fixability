# GitHub Copilot Instructions for geo-fixability

## Project Overview

This repository focuses on studying the theoretical ability of geophysical map matching to provide position fixes based on map information and trajectory specifications. It generates synthetic geophysical contour maps and trajectories to develop relationships between maps, trajectories, and position fix accuracy/reliability.

## Code Style and Conventions

### Python Standards

- Use Python 3.13+ features
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include docstrings for all public functions and classes
- Place all source code in `src/geo_fixability/`

### Documentation

- Use NumPy-style docstrings
- Document mathematical parameters clearly (e.g., "beta: PSD exponent (0=white, 2=brownian)")
- Include units and ranges for physical parameters
- Reference theoretical models when applicable (e.g., Matérn, Fractional Brownian Motion)

### Naming Conventions

- Use snake_case for functions and variables
- Use descriptive parameter names that reflect their physical meaning
- Prefix internal/helper functions with underscore if not part of public API

## Key Dependencies

- **NumPy**: Array operations, FFT, random number generation
- **SciPy**: Signal processing (ndimage), spatial operations (cdist), special functions
- **Matplotlib**: Visualization of contour maps and trajectories

## Domain-Specific Context

### Geophysical Maps

- Maps are represented as 2D scalar fields (numpy arrays)
- Common generation methods:
    - Spectral synthesis (most control over spatial statistics)
    - Gaussian Random Fields (simpler, uses Matérn covariance)
    - Fractional Brownian Motion (for rough/smooth terrain)
- Key parameters:
    - `correlation_length`: Spatial scale of features (in pixels)
    - `beta`: Power spectral density exponent
    - `hurst`: Smoothness parameter (0.5 = Brownian)
    - `anisotropy_ratio` and `angle`: Directional properties

### Position Fixing

- Focus on map-matching algorithms for position estimation
- Consider observability and uniqueness of position solutions
- Analyze how map characteristics and trajectory properties affect fix accuracy

### Future Development Areas

- Trajectory generation (planned)
- Map-matching algorithms (planned)
- Position fix quality metrics (planned)
- Uncertainty quantification (planned)

## Mathematical Notation

- Use standard statistical notation (μ, σ, ν)
- Use KaTeX syntax in markdown: inline `$equation$`, blocks `$$equation$$`
- Common symbols:
    - k: spatial frequency
    - β (beta): spectral exponent
    - ν (nu): Matérn smoothness parameter
    - H: Hurst exponent

## Code Organization

When suggesting new features:

- Add field generation methods to [`mapping.py`](src/geo_fixability/mapping.py)
- Create new modules for distinct functionality (e.g., `trajectories.py`, `matching.py`)
- Use `__init__.py` to expose public API
- Keep the `py.typed` marker for type checking support

## Testing and Validation

- Validate generated fields have correct statistical properties
- Verify numerical stability (handle division by zero, matrix conditioning)
- Check edge cases (small correlation lengths, extreme parameter values)
- Visualize results to ensure physical plausibility

## Performance Considerations

- Use vectorized NumPy operations over loops
- Consider memory efficiency for large maps (see `n_samples` limitation in `generate_field_grf`)
- Leverage FFT for spectral methods (O(n log n) complexity)
- Use sparse operations when applicable

## Common Patterns

### Field Normalization

Always normalize generated fields to zero mean and unit variance:

```python
field = (field - field.mean()) / field.std()
```

### Frequency Domain Operations

Use `np.fft.fftfreq` for proper frequency grid generation in spectral methods.

### Random Number Generation

Use `np.random` for reproducibility. Consider adding `seed` parameters for deterministic testing.

## Questions to Ask

When the user requests new functionality, consider:

1. What statistical or physical properties should the generated data have?
2. What parameters control these properties?
3. How does this relate to real-world geophysical data?
4. What validation or visualization would help verify correctness?
5. Are there numerical stability concerns?