"""
Real terrain (DEM) data loading via SRTM tiles.

Tiles are downloaded from the AWS Skadi mirror (no auth required) and cached
locally.  SRTM covers ±60° latitude at 1 arc-second (~30 m) resolution.

Tile URL: https://s3.amazonaws.com/elevation-tiles-prod/skadi/{NS}{lat:02d}/{NS}{lat:02d}{EW}{lon:03d}.hgt.gz
"""

import gzip
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# Default cache directory
DEFAULT_CACHE = Path.home() / ".cache" / "geo_fixability" / "srtm"

# SRTM constants
_SRTM_VOID = -32768
_TILE_SIZES = {
    3601 * 3601 * 2: 3601,  # SRTM1, 1 arc-second (~30 m)
    1201 * 1201 * 2: 1201,  # SRTM3, 3 arc-second (~90 m)
}
_ARCSEC = {3601: 1.0, 1201: 3.0}
_EARTH_RADIUS_M = 6_371_008.8


def dx_at_latitude(lat_deg: float, arcsec: float = 1.0) -> float:
    """
    Pixel size in meters for a given latitude and arc-second resolution.

    Returns the geometric mean of east-west and north-south pixel sizes
    (isotropic approximation valid for small patches).

    Parameters
    ----------
    lat_deg : float
        Latitude in decimal degrees.
    arcsec : float
        Resolution in arc-seconds. SRTM1 = 1.0, SRTM3 = 3.0.
    """
    deg_per_arcsec = arcsec / 3600.0
    ns_m = _EARTH_RADIUS_M * np.radians(deg_per_arcsec)
    ew_m = ns_m * np.cos(np.radians(lat_deg))
    return float(np.sqrt(ns_m * ew_m))  # geometric mean


def _tile_url(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_str = f"{ns}{abs(lat):02d}"
    lon_str = f"{ew}{abs(lon):03d}"
    return (
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi"
        f"/{lat_str}/{lat_str}{lon_str}.hgt.gz"
    )


def _tile_path(lat: int, lon: int, cache_dir: Path) -> Path:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt.gz"
    return cache_dir / name


def download_srtm_tile(
    lat: int,
    lon: int,
    cache_dir: Path = DEFAULT_CACHE,
    verbose: bool = True,
) -> np.ndarray:
    """
    Download (and cache) an SRTM HGT tile, return elevation in metres.

    Void pixels (ocean, missing data) are returned as NaN.

    Parameters
    ----------
    lat : int
        Tile south-west corner latitude (floor of decimal latitude).
    lon : int
        Tile south-west corner longitude (floor of decimal longitude).
    cache_dir : Path
        Directory for caching downloaded tiles.
    verbose : bool
        Print download progress.

    Returns
    -------
    ndarray, shape (N, N), float32
        Elevation in metres.  N = 3601 (SRTM1) or 1201 (SRTM3).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _tile_path(lat, lon, cache_dir)

    if not cached.exists():
        url = _tile_url(lat, lon)
        if verbose:
            print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, cached)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(
                f"SRTM tile N{abs(lat):02d}{'EW'[lon<0]}{abs(lon):03d} not available "
                f"(HTTP {e.code}). Tile may be ocean-only or outside SRTM coverage."
            ) from e
    else:
        if verbose:
            print(f"Using cached {cached.name}")

    with gzip.open(cached, "rb") as f:
        raw = f.read()

    n = _TILE_SIZES.get(len(raw))
    if n is None:
        raise ValueError(f"Unexpected tile size {len(raw)} bytes for {cached.name}")

    elev = np.frombuffer(raw, dtype=np.dtype(">i2")).reshape(n, n).astype(np.float32)
    elev[elev == _VOID] = np.nan
    return elev


_VOID = _SRTM_VOID


def load_terrain_patch(
    lat_center: float,
    lon_center: float,
    size_km: float = 50.0,
    cache_dir: Path = DEFAULT_CACHE,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Load a square terrain patch centred on (lat_center, lon_center).

    Downloads the required SRTM tile(s), extracts a square subregion, fills
    any voids by linear interpolation, and normalises to zero-mean unit-variance.

    Parameters
    ----------
    lat_center : float
        Centre latitude [decimal degrees].
    lon_center : float
        Centre longitude [decimal degrees].
    size_km : float
        Side length of the patch [km].
    cache_dir : Path
        Directory for caching downloaded tiles.
    verbose : bool
        Print download progress.

    Returns
    -------
    field : ndarray, shape (N, N), float64
        Normalised elevation patch (zero-mean, unit-variance).
    dx_m : float
        Pixel size in metres at the given latitude.
    """
    # Only handles single-tile patches for now (size_km < ~100 km at mid-latitudes)
    tile_lat = int(np.floor(lat_center))
    tile_lon = int(np.floor(lon_center))

    elev = download_srtm_tile(tile_lat, tile_lon, cache_dir=cache_dir, verbose=verbose)
    n = elev.shape[0]
    arcsec = _ARCSEC[n]
    dx_m = dx_at_latitude(lat_center, arcsec=arcsec)

    # Pixels per km at this latitude
    px_per_km = 1000.0 / dx_m
    half_px = int(round(size_km * px_per_km / 2))

    # Convert centre coordinates to pixel indices
    # SRTM tiles: row 0 = north edge, row N-1 = south edge
    lat_frac = lat_center - tile_lat  # 0..1 within tile
    lon_frac = lon_center - tile_lon

    row_center = int(round((1.0 - lat_frac) * (n - 1)))
    col_center = int(round(lon_frac * (n - 1)))

    row0 = max(0, row_center - half_px)
    row1 = min(n, row_center + half_px)
    col0 = max(0, col_center - half_px)
    col1 = min(n, col_center + half_px)

    patch = elev[row0:row1, col0:col1].astype(float)

    # Fill NaN voids by nearest-neighbour (simple, avoids scipy dependency)
    if np.any(np.isnan(patch)):
        void_mask = np.isnan(patch)
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(void_mask, return_indices=True)
        patch[void_mask] = patch[idx[0][void_mask], idx[1][void_mask]]

    # Make square by trimming to shortest side
    side = min(patch.shape)
    patch = patch[:side, :side]

    std = patch.std()
    if std == 0.0:
        raise ValueError("Terrain patch is flat (constant elevation) — no information content.")
    field = (patch - patch.mean()) / std

    return field, dx_m
