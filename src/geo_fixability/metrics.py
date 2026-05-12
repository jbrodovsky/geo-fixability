"""
Information-content metrics for geophysical navigation maps.
"""

import numpy as np


def map_information_content(
    field: np.ndarray,
    dx: float = 1.0,
    sigma_meas: float = 1.0,
) -> dict[str, float]:
    """
    Quantify the navigation information content of a geophysical map.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Map values. Should be zero-mean and unit-variance.
    dx : float
        Pixel size [m/pixel]. Default 1.0.
    sigma_meas : float
        Measurement noise standard deviation [field units]. Default 1.0.

    Returns
    -------
    dict with keys:

    ``information_index`` : float in [0, 1]
        Normalized Shannon entropy of the power spectral density.
        0 = flat map (all power at DC; no position information).
        1 = white noise (power uniform across all spatial frequencies).
        For a 10 km map with spectral synthesis: achievable range ≈ [0.10, 0.97].
        Controlled primarily by the PSD slope (beta parameter): high beta → low I.

    ``gradient_rms`` : float
        RMS gradient magnitude [field_units/pixel].
        Proxy for mean Fisher Information density.
        ~0 for flat maps; ~1 for unit-variance white noise.

    ``correlation_length_px`` : float
        1/e autocorrelation length [pixels], measured from the radial ACF.
        Minimum travel distance for statistically independent measurements.
        Note: this is the *measured* correlation of the generated field and
        differs from the ``correlation_length`` input to ``generate_field_spectral``
        (which is a PSD roll-off frequency scale, not the ACF scale).

    ``correlation_length_m`` : float
        1/e autocorrelation length [m] = correlation_length_px * dx.

    ``localization_bound_px`` : float
        Cramér-Rao lower bound on single-measurement position error [pixels].
        Equal to sigma_meas / gradient_rms; inf for flat maps.

    ``localization_bound_m`` : float
        Single-measurement CRB on position error [m].

    ``spatial_diversity_index`` : float in [0, 1]
        Feature fineness relative to map extent: D = 1 - corr_len_px / (N/2).
        1 = correlation_length = 1 pixel (maximum fine-grained structure).
        0 = correlation_length = half the map side (one large feature, no diversity).
        For a 10 km map (N=334, half-side = 5 km):
          D=0.9 → measured corr ≈ 500 m;  D=0.7 → ≈1.5 km;  D=0.5 → ≈2.5 km.
        Achievable lower bound with spectral synthesis ≈ 0.47, because the
        power-law PSD always produces an ACF that falls to 1/e within the map.
        D=0 would require a single smooth feature spanning the entire map extent.
        D and I are anti-correlated: fine-grained maps (high D) are also
        information-rich (high I).

    ``localization_index`` : float in [0, 1]
        Log-scale position accuracy relative to map resolution and extent.
        L = 1 - log10(bound_px) / log10(diag_px)
        1 = localization_bound ≤ 1 pixel (sub-pixel accuracy).
        0 = localization_bound ≥ map diagonal (map provides no useful fix).
        For a 10 km map at dx=30 m (diagonal ≈ 14.1 km), sigma_meas=1.0:
          L=0.9 → bound ≈ 56 m;  L=0.7 → ≈190 m;  L=0.5 → ≈650 m;
          L=0.3 → ≈2.2 km;       L=0.1 → ≈7.7 km.
        Achievable lower bound with spectral synthesis ≈ 0.38.

    Notes
    -----
    ``information_index`` and ``correlation_length_m`` together define
    the information-resolution tradeoff:

    * High index + small correlation → fine-grained, dense information
    * High index + large correlation → coarse but consistent information
    * Low index → poor localizability regardless of spatial scale

    ``localization_bound_m`` is the per-measurement CRB floor. Accumulating
    N independent measurements (spaced >= correlation_length_m apart)
    reduces position uncertainty by sqrt(N).

    The two [0, 1] dial pairs:

    * ``information_index`` / ``spatial_diversity_index``: spectral richness
      and spatial fineness — together describe map quality independent of noise.
    * ``localization_index``: practical navigation accuracy given ``sigma_meas``
      and ``dx``.

    The information dial extremes:

    * ``information_index`` = 0: field is flat — all measurements identical,
      no position information extractable.
    * ``information_index`` = 1: field is white noise — each pixel has a
      statistically independent value; maximum spatial diversity.

    Examples
    --------
    >>> from geo_fixability.mapping import generate_field_spectral
    >>> from geo_fixability.metrics import map_information_content
    >>> flat = np.zeros((256, 256))
    >>> info = map_information_content(flat, dx=10.0)
    >>> info['information_index']
    0.0
    >>> rough = generate_field_spectral((256, 256), beta=0, correlation_length=1.0)
    >>> info = map_information_content(rough, dx=10.0, sigma_meas=0.1)
    >>> info['information_index'] > 0.9
    True
    """
    ny, nx = field.shape

    # --- Gradient RMS (proxy for mean Fisher Information density) ---
    # np.gradient uses 2nd-order central differences; for unit-variance
    # white noise, gradient_rms ≈ 1.0; for flat fields, gradient_rms = 0.
    gy, gx = np.gradient(field)
    gradient_rms = float(np.sqrt(np.mean(gx**2 + gy**2)))

    # --- Power spectral density ---
    psd = np.abs(np.fft.fft2(field)) ** 2
    psd[0, 0] = 0.0  # exclude DC; field should be zero-mean

    # --- Information index: normalized spectral entropy ---
    # Shannon entropy of the PSD treated as a probability distribution over
    # spatial frequencies.  Maximized when PSD is uniform (white noise),
    # minimized (=0) when all power is at a single frequency (flat or pure
    # sinusoid).
    total_power = psd.sum()
    if total_power > 0.0:
        p = psd / total_power
        log_p = np.zeros_like(p)
        nz = p > 0.0
        log_p[nz] = np.log(p[nz])
        h = float(-np.sum(p * log_p))
        # Max entropy: uniform over all non-DC bins → log(N-1)
        information_index = float(np.clip(h / np.log(psd.size - 1), 0.0, 1.0))
    else:
        information_index = 0.0

    # --- Autocorrelation via Wiener-Khinchin theorem ---
    # Use the full PSD (including DC) so the ACF peak at zero-lag reflects
    # total field variance, not just the AC-coupled component.
    psd_full = np.abs(np.fft.fft2(field)) ** 2
    acf = np.fft.fftshift(np.fft.ifft2(psd_full).real)

    center = acf[ny // 2, nx // 2]
    if center > 0.0:
        acf /= center  # normalize to 1 at zero lag

    # --- Radial average of ACF via bincount (O(N)) ---
    cy, cx = ny // 2, nx // 2
    max_r = min(cy, cx)
    y_idx, x_idx = np.indices((ny, nx))
    r_map = np.round(np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)).astype(int)

    mask = r_map <= max_r
    r_flat = r_map[mask]
    acf_flat = acf[mask]
    counts = np.bincount(r_flat, minlength=max_r + 1)
    sums = np.bincount(r_flat, weights=acf_flat, minlength=max_r + 1)
    radial_acf = np.where(counts > 0, sums / counts, 0.0)

    # --- Correlation length: radius of first 1/e crossing ---
    below = np.where(radial_acf < 1.0 / np.e)[0]
    correlation_length_px = float(below[0]) if len(below) > 0 else float(max_r)

    # --- Localization CRB from Fisher Information ---
    # For a single measurement z = h(x,y) + noise, the FIM is |∇h|² / σ².
    # Averaging over trajectory positions: J_avg = gradient_rms² / σ².
    # CRB: σ_pos ≥ σ_meas / gradient_rms  (in each axis; isotropic approx.)
    if gradient_rms > 0.0:
        localization_bound_px = sigma_meas / gradient_rms
    else:
        localization_bound_px = float("inf")

    # --- Spatial diversity index [0, 1] ---
    # How fine the features are relative to the map extent.
    # Max useful correlation = half the shorter map side (larger = one blob).
    half_min_side = min(ny, nx) / 2.0
    spatial_diversity_index = float(
        np.clip(1.0 - correlation_length_px / half_min_side, 0.0, 1.0)
    )

    # --- Localization index [0, 1] ---
    # Log-scale accuracy relative to map diagonal (spans ~3 orders of magnitude).
    # loc_bound = 1 px → 1.0;  loc_bound = map diagonal → 0.0.
    map_diagonal_px = float(np.sqrt(ny**2 + nx**2))
    if np.isfinite(localization_bound_px) and localization_bound_px > 0.0:
        log_ratio = np.log10(localization_bound_px) / np.log10(map_diagonal_px)
        localization_index = float(np.clip(1.0 - log_ratio, 0.0, 1.0))
    else:
        localization_index = 0.0

    return {
        "information_index": information_index,
        "spatial_diversity_index": spatial_diversity_index,
        "localization_index": localization_index,
        "gradient_rms": gradient_rms,
        "correlation_length_px": correlation_length_px,
        "correlation_length_m": correlation_length_px * dx,
        "localization_bound_px": localization_bound_px,
        "localization_bound_m": localization_bound_px * dx,
    }
