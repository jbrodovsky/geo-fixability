"""
Generate a parameter sweep of synthetic geophysical maps.

Usage
-----
    uv run generate-maps                          # 10 km map at 30 m/px, all methods
    uv run generate-maps --outdir data/synthetic --seed 7
    uv run generate-maps --method fbm             # fBm sweep only
    uv run generate-maps --no-save                # print metrics only
    uv run generate-maps --extent-km 10 --dx 100  # 10 km map at 100 m/px → 100×100 grid
    uv run generate-maps --extent-km 10 --dx 10   # 10 km map at 10 m/px  → 1000×1000 grid

Grid size is always derived: N = round(extent_km * 1000 / dx).
Use --extent-km and --dx to define the physical domain; never set grid size directly.

Metric ranges for a 10 km × 10 km map (default: extent-km=10, dx=30 m → 334×334 grid)
---------------------------------------------------------------------------------------
Information Index  (I) : [0.10, 0.97]  — controlled primarily by beta
Spatial Diversity  (D) : [0.47, 0.99]  — controlled primarily by correlation_length_m
Localization Index (L) : [0.38, 1.00]  — derived from gradient RMS

I and D are anti-correlated: fine-grained fields (high D) are information-rich
(high I); smooth fields (low I) also have coarse spatial structure (low D).
D cannot be driven below ~0.47 with the spectral method because the power-law
PSD always produces an ACF that drops to 1/e within the map extent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from geo_fixability.mapping import generate_field_spectral, generate_fbm
from geo_fixability.metrics import map_information_content


# ---------------------------------------------------------------------------
# Parameter grids
#
# For a 10 km × 10 km map at dx=30 m (N=334, half-side = 167 px = 5 km):
#   D_target = 1 - (correlation_length_m / dx) / 167
#
# beta controls spectral slope → primarily drives Information Index (I):
#   beta=0.2 → I≈0.97   beta=0.7 → I≈0.84   beta=1.3 → I≈0.69
#   beta=2.0 → I≈0.55   beta=3.5 → I≈0.32   beta=5.0 → I≈0.15
#
# correlation_length_m controls frequency roll-off → primarily drives D:
#   30 m  → D≈0.99   300 m  → D≈0.97   1000 m → D≈0.90
#   2500 m→ D≈0.77   4500 m → D≈0.68   (lower limit ~0.47 at high beta)
#
# NOTE: correlation_length_m is the Gaussian roll-off scale in the PSD,
# NOT the measured ACF correlation length. The measured correlation is
# roughly 0.35–0.60× the input value for β < 1, and approaches a
# beta-dependent asymptote for β > 2.
# ---------------------------------------------------------------------------

SPECTRAL_GRID = {
    # Even steps across the achievable Information Index range [0.15, 0.97]
    "beta": [0.2, 0.7, 1.3, 2.0, 3.5],
    # Even steps across the achievable Spatial Diversity range [~0.47, 0.99]
    # expressed in physical metres (converted to pixels at runtime)
    "correlation_length_m": [30, 300, 1000, 2500, 4500],
    "anisotropy_ratio": [1.0, 2.0, 5.0],
}

FBM_GRID = {
    # H = 0.1 → β=1.2 (rough);  H = 0.9 → β=2.8 (smooth)
    # Even steps in H give even steps in β = 2H+1 ∈ [1.2, 2.8]
    "hurst": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def _spectral_configs() -> list[dict]:
    configs = []
    for beta in SPECTRAL_GRID["beta"]:
        for cl_m in SPECTRAL_GRID["correlation_length_m"]:
            for ar in SPECTRAL_GRID["anisotropy_ratio"]:
                configs.append(
                    {"method": "spectral", "beta": beta,
                     "correlation_length_m": cl_m, "anisotropy_ratio": ar}
                )
    return configs


def _fbm_configs() -> list[dict]:
    return [{"method": "fbm", "hurst": h} for h in FBM_GRID["hurst"]]


def _generate_one(cfg: dict, size: int, dx: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    if cfg["method"] == "spectral":
        cl_px = cfg["correlation_length_m"] / dx
        return generate_field_spectral(
            (size, size),
            beta=cfg["beta"],
            correlation_length=cl_px,
            anisotropy_ratio=cfg["anisotropy_ratio"],
        )
    elif cfg["method"] == "fbm":
        return generate_fbm((size, size), hurst=cfg["hurst"])
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")


def _config_name(cfg: dict) -> str:
    if cfg["method"] == "spectral":
        return (
            f"spectral_b{cfg['beta']:.1f}"
            f"_cl{cfg['correlation_length_m']:04d}m"
            f"_ar{cfg['anisotropy_ratio']:.1f}"
        )
    elif cfg["method"] == "fbm":
        return f"fbm_h{cfg['hurst']:.2f}"
    return "unknown"


_HEADER = (
    f"{'name':<44} {'I':>5} {'D':>5} {'L':>5} "
    f"{'corr_m':>8} {'bound_m':>8}"
)
_DIVIDER = "-" * len(_HEADER)


def _row(name: str, m: dict) -> str:
    return (
        f"{name:<44} {m['information_index']:5.3f} "
        f"{m['spatial_diversity_index']:5.3f} "
        f"{m['localization_index']:5.3f} "
        f"{m['correlation_length_m']:8.1f} "
        f"{m['localization_bound_m']:8.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic geophysical maps and report information metrics."
    )
    parser.add_argument(
        "--outdir", default="data/synthetic",
        help="Output directory for .npy, .json, and .png files (default: data/synthetic)"
    )
    parser.add_argument(
        "--extent-km", type=float, default=10.0, dest="extent_km",
        help="Physical map side length in km (default: 10.0)"
    )
    parser.add_argument(
        "--dx", type=float, default=30.0,
        help="Pixel size in meters (default: 30.0, matching SRTM1). "
             "Grid size is derived: N = round(extent_km * 1000 / dx)"
    )
    parser.add_argument(
        "--sigma-meas", type=float, default=1.0,
        help="Measurement noise std dev in field units (default: 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--method", choices=["spectral", "fbm", "all"], default="all",
        help="Which generation method to sweep (default: all)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving files, just print metrics"
    )
    args = parser.parse_args()

    size = round(args.extent_km * 1000 / args.dx)
    print(f"Map: {args.extent_km} km × {args.extent_km} km  "
          f"dx={args.dx} m/px  →  {size}×{size} grid\n")

    outdir = Path(args.outdir)
    if not args.no_save:
        outdir.mkdir(parents=True, exist_ok=True)

    configs: list[dict] = []
    if args.method in ("spectral", "all"):
        configs += _spectral_configs()
    if args.method in ("fbm", "all"):
        configs += _fbm_configs()

    units = (f"{'':44} {'[0,1]':>5} {'[0,1]':>5} {'[0,1]':>5} "
             f"{'[m]':>8} {'[m]':>8}")
    print(_HEADER)
    print(units)
    print(_DIVIDER)

    for i, cfg in enumerate(configs):
        name = _config_name(cfg)
        field = _generate_one(cfg, size, args.dx, args.seed + i)
        metrics = map_information_content(field, dx=args.dx, sigma_meas=args.sigma_meas)

        print(_row(name, metrics))

        if not args.no_save:
            np.save(outdir / f"{name}.npy", field)
            meta = {**cfg, "name": name, "extent_km": args.extent_km,
                    "grid_size": size, "dx_m": args.dx,
                    "sigma_meas": args.sigma_meas, "seed": args.seed + i, **metrics}
            (outdir / f"{name}.json").write_text(json.dumps(meta, indent=2))
            plt.imsave(outdir / f"{name}.png", field, cmap="gray")

    print(_DIVIDER)
    print(f"\n{len(configs)} maps", end="")
    if not args.no_save:
        print(f" saved to {outdir}/")
    else:
        print(" (not saved, --no-save set)")


if __name__ == "__main__":
    main()
