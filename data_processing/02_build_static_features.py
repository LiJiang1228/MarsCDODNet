from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter


MARS_RADIUS_M = 3389500
CHANNEL_NAMES = np.asarray(
    [
        "elevation",
        "slope",
        "relief",
        "roughness",
        "sin(latitude)",
        "cos(latitude)",
        "sin(longitude)",
        "cos(longitude)",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MarsCDODNet static terrain features from MOLA data.")
    parser.add_argument("--mola-path", type=Path, required=True, help="Source MOLA NetCDF file.")
    parser.add_argument("--latlon-path", type=Path, required=True, help="Dynamic-data NPZ containing LAT2D and LON2D.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mola-variable", default="alt")
    parser.add_argument("--latitude-name", default="latitude")
    parser.add_argument("--subcells-lat", type=int, default=20)
    parser.add_argument("--subcells-lon", type=int, default=24)
    return parser.parse_args()


def cell_edges_from_centres(centres: np.ndarray, lower: float, upper: float) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = lower
    edges[-1] = upper
    return edges


def load_cdod_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "LAT2D" not in data or "LON2D" not in data:
            raise KeyError(f"{path} must contain LAT2D and LON2D")
        lat_centres = np.asarray(data["LAT2D"][:, 0], dtype=np.float64)
        lon_centres = np.asarray(data["LON2D"][0], dtype=np.float64) % 360.0
    if lat_centres.shape != (36,) or lon_centres.shape != (60,):
        raise ValueError(f"Expected a 36 x 60 CDOD grid, got {lat_centres.shape} x {lon_centres.shape}")
    return lat_centres, lon_centres


def build_high_resolution_grid(lat_centres: np.ndarray, lon_centres: np.ndarray, subcells_lat: int, subcells_lon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse_lat_edges = cell_edges_from_centres(lat_centres, -90.0, 90.0)
    lat_edges = []
    for index in range(lat_centres.size):
        section = np.linspace(coarse_lat_edges[index], coarse_lat_edges[index + 1], subcells_lat + 1)
        lat_edges.extend(section if index == 0 else section[1:])

    lon_step = float(np.median(np.diff(lon_centres)))
    coarse_lon_edges = np.concatenate(([lon_centres[0] - lon_step / 2.0], lon_centres + lon_step / 2.0))
    coarse_lon_edges[0] = 0.0
    coarse_lon_edges[-1] = 360.0
    lon_edges = []
    for index in range(lon_centres.size):
        section = np.linspace(coarse_lon_edges[index], coarse_lon_edges[index + 1], subcells_lon + 1)
        lon_edges.extend(section if index == 0 else section[1:])

    lat_edges = np.asarray(lat_edges, dtype=np.float64)
    lon_edges = np.asarray(lon_edges, dtype=np.float64)
    lat_hr = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_hr = (0.5 * (lon_edges[:-1] + lon_edges[1:])) % 360.0
    return lat_edges, lat_hr.astype(np.float32), lon_hr.astype(np.float32)


def load_mola(path: Path, variable: str, latitude_name: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{path} does not contain variable {variable!r}")
        altitude = ds[variable]
        mola_lat = np.asarray(altitude[latitude_name].values, dtype=np.float64)
        mola_altitude = np.asarray(altitude.values, dtype=np.float32)
    if mola_altitude.ndim != 2:
        raise ValueError(f"MOLA altitude must be two-dimensional, got {mola_altitude.shape}")
    if mola_lat[0] > mola_lat[-1]:
        mola_lat = mola_lat[::-1]
        mola_altitude = mola_altitude[::-1, :]
    return mola_altitude, mola_lat


def aggregate_mola_to_subgrids(mola_altitude: np.ndarray, mola_lat: np.ndarray, lat_edges: np.ndarray, output_width: int) -> tuple[np.ndarray, np.ndarray]:
    if mola_altitude.shape[1] % output_width != 0:
        raise ValueError(f"MOLA longitude size {mola_altitude.shape[1]} is not divisible by output width {output_width}")
    longitude_factor = mola_altitude.shape[1] // output_width
    output_height = lat_edges.size - 1
    elevation = np.empty((output_height, output_width), dtype=np.float32)
    roughness = np.empty_like(elevation)

    for out_row in range(output_height):
        lower, upper = lat_edges[out_row], lat_edges[out_row + 1]
        in_cell = (mola_lat >= lower) & (mola_lat < upper if out_row < output_height - 1 else mola_lat <= upper)
        source_rows = np.flatnonzero(in_cell)
        if source_rows.size == 0:
            source_rows = np.asarray([int(np.argmin(np.abs(mola_lat - 0.5 * (lower + upper))))], dtype=np.int64)
        block = mola_altitude[source_rows, :].reshape(source_rows.size, output_width, longitude_factor)
        elevation[out_row] = np.nanmean(block, axis=(0, 2), dtype=np.float64).astype(np.float32)
        roughness[out_row] = np.nanstd(block, axis=(0, 2), dtype=np.float64).astype(np.float32)
        if (out_row + 1) % 60 == 0 or out_row + 1 == output_height:
            print(f"Aggregated latitude subgrids: {out_row + 1}/{output_height}")

    if not np.all(np.isfinite(elevation)) or not np.all(np.isfinite(roughness)):
        raise ValueError("MOLA aggregation produced NaN or Inf")
    return elevation, roughness


def compute_slope_deg(elevation: np.ndarray, lat_hr: np.ndarray, lon_hr: np.ndarray) -> np.ndarray:
    y_m = MARS_RADIUS_M * np.deg2rad(lat_hr.astype(np.float64))
    dz_dy = np.gradient(elevation.astype(np.float64), y_m, axis=0, edge_order=2)
    dlon_rad = np.deg2rad(float(np.median(np.diff(lon_hr.astype(np.float64)))))
    dx_m = MARS_RADIUS_M * np.clip(np.cos(np.deg2rad(lat_hr.astype(np.float64)))[:, None], 1e-6, None) * dlon_rad
    dz_dx = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / (2.0 * dx_m)
    return np.rad2deg(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)


def compute_relief(elevation: np.ndarray) -> np.ndarray:
    mode = ("nearest", "wrap")
    return (maximum_filter(elevation, size=(3, 3), mode=mode) - minimum_filter(elevation, size=(3, 3), mode=mode)).astype(np.float32)


def zscore(values: np.ndarray, name: str, transform: str = "none") -> tuple[np.ndarray, dict[str, float | str]]:
    working = np.asarray(values, dtype=np.float32)
    if transform == "log1p":
        working = np.log1p(np.maximum(working, 0.0)).astype(np.float32)
    mean, std = float(np.nanmean(working)), float(np.nanstd(working))
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid standard deviation for {name}: {std}")
    return ((working - mean) / std).astype(np.float32), {"mean": mean, "std": std, "transform": transform}


def main() -> None:
    args = parse_args()
    if args.subcells_lat <= 0 or args.subcells_lon <= 0:
        raise ValueError("--subcells-lat and --subcells-lon must be positive")

    cdod_lat, cdod_lon = load_cdod_grid(args.latlon_path)
    lat_edges, lat_hr, lon_hr = build_high_resolution_grid(cdod_lat, cdod_lon, args.subcells_lat, args.subcells_lon)
    expected_shape = (36 * args.subcells_lat, 60 * args.subcells_lon)
    if lat_hr.shape != (expected_shape[0],) or lon_hr.shape != (expected_shape[1],):
        raise ValueError(f"Invalid high-resolution coordinate shapes: {lat_hr.shape}, {lon_hr.shape}")
    print(f"CDOD grid: 36 x 60; static subgrid: {expected_shape[0]} x {expected_shape[1]}")

    mola_altitude, mola_lat = load_mola(args.mola_path, args.mola_variable, args.latitude_name)
    print(f"Raw MOLA altitude shape: {mola_altitude.shape}")
    elevation, roughness = aggregate_mola_to_subgrids(mola_altitude, mola_lat, lat_edges, expected_shape[1])
    slope = compute_slope_deg(elevation, lat_hr, lon_hr)
    relief = compute_relief(elevation)

    elevation_z, elevation_stats = zscore(elevation, "elevation")
    slope_z, slope_stats = zscore(slope, "slope", transform="log1p")
    relief_z, relief_stats = zscore(relief, "relief", transform="log1p")
    roughness_z, roughness_stats = zscore(roughness, "roughness", transform="log1p")
    lat2d, lon2d = np.meshgrid(lat_hr, lon_hr, indexing="ij")
    static_features = np.stack(
        [
            elevation_z,
            slope_z,
            relief_z,
            roughness_z,
            np.sin(np.deg2rad(lat2d)).astype(np.float32),
            np.cos(np.deg2rad(lat2d)).astype(np.float32),
            np.sin(np.deg2rad(lon2d)).astype(np.float32),
            np.cos(np.deg2rad(lon2d)).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if static_features.shape != (8, *expected_shape) or not np.all(np.isfinite(static_features)):
        raise ValueError(f"Invalid static features: shape={static_features.shape}")

    stats = {"elevation": elevation_stats, "slope": slope_stats, "relief": relief_stats, "roughness": roughness_stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        static_features=static_features,
        channel_names=CHANNEL_NAMES,
        lat_hr=lat_hr,
        lon_hr=lon_hr,
        normalization_stats=np.asarray(json.dumps(stats, sort_keys=True)),
    )
    print(f"Saved: {args.output}")
    print(f"static_features: {static_features.shape}, dtype={static_features.dtype}")
    print(f"channels: {CHANNEL_NAMES.tolist()}")


if __name__ == "__main__":
    main()
