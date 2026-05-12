"""
Download an SRTM terrain patch and report its information metrics.

Usage
-----
    uv run fetch-terrain --lat 36.5 --lon -121.5           # Big Sur, CA
    uv run fetch-terrain --lat 47.0 --lon 10.5 --size 80   # Alps
    uv run fetch-terrain --lat 36.1 --lon -112.1           # Grand Canyon
    uv run fetch-terrain --lat 27.9 --lon 86.9             # Everest region
    uv run fetch-terrain --lat 64.0 --lon -16.0 --size 30  # Iceland glacier

Options
-------
    --lat FLOAT        Centre latitude [decimal degrees]
    --lon FLOAT        Centre longitude [decimal degrees]
    --size FLOAT       Patch side length [km] (default: 50)
    --sigma FLOAT      Measurement noise std dev in field units (default: 1.0)
    --cache-dir PATH   Override default SRTM cache directory
    --save PATH        Save normalised field as .npy to this path
    --quiet            Suppress download messages
"""

import argparse
from pathlib import Path

import numpy as np

from geo_fixability.terrain import load_terrain_patch, DEFAULT_CACHE
from geo_fixability.metrics import map_information_content


_PRESETS = {
    "big-sur":      (36.5,  -121.5, "Big Sur, CA"),
    "alps":         (47.0,    10.5, "Alps, Austria/Italy"),
    "grand-canyon": (36.1,  -112.1, "Grand Canyon, AZ"),
    "everest":      (27.9,    86.9, "Everest region, Nepal"),
    "iceland":      (64.0,   -16.0, "Vatnajokull glacier, Iceland"),
    "amazon":       (-3.5,   -62.0, "Amazon basin, Brazil"),
    "sahara":       (23.0,    10.0, "Sahara, Libya"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch an SRTM terrain patch and report navigation information metrics."
    )
    loc_group = parser.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--lat", type=float, help="Centre latitude [°]")
    loc_group.add_argument(
        "--preset", choices=list(_PRESETS.keys()),
        help="Named location preset"
    )
    parser.add_argument("--lon", type=float, help="Centre longitude [°] (required with --lat)")
    parser.add_argument("--size", type=float, default=50.0, help="Patch side [km] (default: 50)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Measurement noise std dev [field units] (default: 1.0)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"SRTM cache directory (default: {DEFAULT_CACHE})")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save normalised field as .npy to this path")
    parser.add_argument("--quiet", action="store_true", help="Suppress download messages")
    args = parser.parse_args()

    if args.preset:
        lat, lon, label = _PRESETS[args.preset]
    else:
        if args.lon is None:
            parser.error("--lon is required when using --lat")
        lat, lon = args.lat, args.lon
        label = f"({lat:.3f}°, {lon:.3f}°)"

    print(f"\nLocation : {label}")
    print(f"Patch    : {args.size} km × {args.size} km")

    field, dx_m = load_terrain_patch(
        lat, lon,
        size_km=args.size,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
    )

    metrics = map_information_content(field, dx=dx_m, sigma_meas=args.sigma)

    ny, nx = field.shape
    print(f"\nTerrain patch: {ny} × {nx} px  |  dx = {dx_m:.1f} m/px")
    print(f"Elevation stats: mean=0.0 (normalised), std=1.0")
    print()
    print("─" * 48)
    print(f"  {'information_index':30s} {metrics['information_index']:.4f}  [0,1]")
    print(f"  {'spatial_diversity_index':30s} {metrics['spatial_diversity_index']:.4f}  [0,1]")
    print(f"  {'localization_index':30s} {metrics['localization_index']:.4f}  [0,1]")
    print(f"  {'gradient_rms':30s} {metrics['gradient_rms']:.4f}  [field_units/px]")
    print(f"  {'correlation_length':30s} {metrics['correlation_length_px']:.1f} px = {metrics['correlation_length_m']:.0f} m")
    lb_label = f"localization_bound (σ={args.sigma:.1f})"
    print(f"  {lb_label:30s} {metrics['localization_bound_px']:.1f} px = {metrics['localization_bound_m']:.0f} m")
    print("─" * 48)

    if args.save:
        np.save(args.save, field)
        print(f"\nSaved normalised field to {args.save}")


if __name__ == "__main__":
    main()
