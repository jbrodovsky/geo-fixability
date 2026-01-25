"""
Toolkit for creating synthetic geophysical maps.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist 


# Method 1: Spectral synthesis (most control)
def generate_field_spectral(shape, beta=2.0, correlation_length=10.0, 
                           anisotropy_ratio=1.0, angle=0):
    """
    beta: PSD exponent (0=white, 2=brownian)
    correlation_length: in pixels
    anisotropy_ratio: elongation factor
    angle: orientation of anisotropy
    """
    ny, nx = shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    
    # Rotate for anisotropy
    kx_rot = kx * np.cos(angle) - ky * np.sin(angle)
    ky_rot = kx * np.sin(angle) + ky * np.cos(angle)
    
    # Anisotropic frequency
    k = np.sqrt((kx_rot * anisotropy_ratio)**2 + ky_rot**2)
    k[0, 0] = 1  # avoid division by zero
    
    # Power law spectrum with correlation length
    psd = k**(-beta) * np.exp(-(k * correlation_length)**2)
    psd[0, 0] = 0  # zero mean
    
    # Random phases
    phase = np.exp(2j * np.pi * np.random.random(shape))
    
    # Generate field
    field_fft = np.sqrt(psd) * phase
    field = np.fft.ifft2(field_fft).real
    
    return (field - field.mean()) / field.std()

# Method 2: Gaussian Random Field (simpler)
def generate_field_grf(shape, correlation_length=10.0, nu=1.5):
    """
    nu: Matérn smoothness (0.5=exponential, 1.5=decent, ∞=Gaussian)
    """
    
    # Sample points
    n_samples = min(shape[0] * shape[1], 10000)  # for efficiency
    idx = np.random.choice(shape[0]*shape[1], n_samples, replace=False)
    y_sample, x_sample = np.unravel_index(idx, shape)
    coords_sample = np.column_stack([y_sample, x_sample])
    
    # Matérn covariance
    dists = cdist(coords_sample, coords_sample)
    if nu == 0.5:  # Exponential
        K = np.exp(-dists / correlation_length)
    else:  # Matérn
        scaled = dists / correlation_length
        K = (2**(1-nu) / gamma(nu)) * scaled**nu * kv(nu, scaled)
        K[dists == 0] = 1
    
    # Sample from GP
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n_samples))
    z = L @ np.random.randn(n_samples)
    
    # Interpolate to grid
    from scipy.interpolate import griddata
    yi, xi = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    field = griddata(coords_sample, z, (yi, xi), method='cubic')
    
    return (field - np.nanmean(field)) / np.nanstd(field)

# Method 3: Fractional Brownian Motion
def generate_fbm(shape, hurst=0.5):
    """
    hurst: 0.5 = Brownian, <0.5 = rough, >0.5 = smooth
    """
    beta = 2 * hurst + 1
    return generate_field_spectral(shape, beta=beta, correlation_length=1.0)