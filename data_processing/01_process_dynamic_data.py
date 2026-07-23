from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def parse_number_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("The list must contain at least one number")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EMARS dynamic fields for MarsCDODNet.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing emars_v1.0_back_mean_MY*_Ls*.nc files.")
    parser.add_argument("--output", type=Path, required=True, help="Output dynamic-field NPZ file.")
    parser.add_argument("--my-start", type=int, default=24)
    parser.add_argument("--my-end", type=int, default=33)
    parser.add_argument("--hours", default="0,6,12,18", help="Retained mars_hour values.")
    parser.add_argument("--pressure-levels", default="10,40,100,300,500,610", help="Requested U/V pressure levels in Pa.")
    parser.add_argument("--temperature-source", choices=("near-surface", "surface"), default="near-surface")
    parser.add_argument("--compressed", action="store_true", help="Write a compressed NPZ file.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_my_from_filename(path: Path) -> int | None:
    match = re.search(r"_MY(\d+)_", path.name)
    return int(match.group(1)) if match else None


def find_variable(ds: xr.Dataset, candidates: tuple[str, ...], filename: str) -> str:
    for name in candidates:
        if name in ds:
            return name
    raise KeyError(f"Could not find any of {candidates} in {filename}")


def earth_datetime64(ds: xr.Dataset, time_index: int) -> np.datetime64:
    second_float = float(ds["earth_second"].isel(time=time_index).values)
    second = int(second_float)
    microsecond = int(round((second_float - second) * 1_000_000))
    value = datetime(
        int(ds["earth_year"].isel(time=time_index).values),
        int(ds["earth_month"].isel(time=time_index).values),
        int(ds["earth_day"].isel(time=time_index).values),
        int(ds["earth_hour"].isel(time=time_index).values),
        int(ds["earth_minute"].isel(time=time_index).values),
        second,
        microsecond,
    )
    return np.datetime64(value, "us")


def select_near_surface(da: xr.DataArray, time_index: int) -> np.ndarray:
    if "pfull" in da.dims:
        return np.asarray(da.isel(time=time_index, pfull=-1).values, dtype=np.float32)
    return np.asarray(da.isel(time=time_index).values, dtype=np.float32)


def pfull_to_pa(pfull_values: np.ndarray, units: str) -> np.ndarray:
    values = np.asarray(pfull_values, dtype=np.float64)
    unit = (units or "").strip().lower()
    if unit in ("mb", "mbar", "millibar", "hpa", "hectopascal"):
        return values * 100.0
    if unit in ("pa", "pascal", "pascals"):
        return values
    return values * 100.0 if np.nanmax(values) < 50 else values


def nearest_pfull_indices(ds: xr.Dataset, target_pressures: tuple[float, ...]) -> tuple[list[int], np.ndarray]:
    if "pfull" not in ds:
        raise KeyError("The NetCDF file does not contain pfull")
    pfull_pa = pfull_to_pa(ds["pfull"].values, str(ds["pfull"].attrs.get("units", "")))
    indices = [int(np.argmin(np.abs(pfull_pa - pressure))) for pressure in target_pressures]
    return indices, pfull_pa[indices].astype(np.float32)


def stack_and_sort(values: list[np.ndarray], order: np.ndarray) -> np.ndarray:
    return np.stack(values, axis=0).astype(np.float32)[order]


def main() -> None:
    args = parse_args()
    if args.my_start > args.my_end:
        raise ValueError("--my-start must not exceed --my-end")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Pass --overwrite to replace it.")

    target_hours = parse_number_list(args.hours)
    target_pressures = parse_number_list(args.pressure_levels)
    files = []
    for path in args.input_dir.glob("emars_v1.0_back_mean_MY*_Ls*.nc"):
        my = parse_my_from_filename(path)
        if my is not None and args.my_start <= my <= args.my_end:
            files.append(path)
    files.sort(key=lambda path: (parse_my_from_filename(path), path.name))
    if not files:
        raise FileNotFoundError(f"No EMARS files for MY{args.my_start}--MY{args.my_end} under {args.input_dir}")
    print(f"Found {len(files)} EMARS files for MY{args.my_start}--MY{args.my_end}")

    t_c_list, ps_list = [], []
    u_sfc_list, v_sfc_list, cdod610_list = [], [], []
    u_level_lists = {int(pressure): [] for pressure in target_pressures}
    v_level_lists = {int(pressure): [] for pressure in target_pressures}
    time_list, mars_hour_list, my_list, ls_list = [], [], [], []
    lat_reference = lon_reference = None
    pressure_indices_reference = pressure_used_reference = None

    for file_index, path in enumerate(files, start=1):
        with xr.open_dataset(path) as ds:
            lat = np.asarray(ds["lat"].values, dtype=np.float32)
            lon = np.asarray(ds["lon"].values, dtype=np.float32)
            if lat_reference is None:
                lat_reference, lon_reference = lat.copy(), lon.copy()
            elif lat.shape != lat_reference.shape or lon.shape != lon_reference.shape or not np.allclose(lat, lat_reference) or not np.allclose(lon, lon_reference):
                raise ValueError(f"Grid mismatch in {path.name}")

            t_name = find_variable(ds, ("ts", "TS"), path.name) if args.temperature_source == "surface" else find_variable(ds, ("t", "T"), path.name)
            ps_name = find_variable(ds, ("ps", "PS"), path.name)
            u_name = find_variable(ds, ("u", "U"), path.name)
            v_name = find_variable(ds, ("v", "V"), path.name)
            tod_name = find_variable(ds, ("tod", "TOD"), path.name)
            if "pfull" not in ds[u_name].dims or "pfull" not in ds[v_name].dims:
                raise ValueError(f"U/V fields in {path.name} must contain pfull")

            p_indices, p_used = nearest_pfull_indices(ds, target_pressures)
            if pressure_indices_reference is None:
                pressure_indices_reference = np.asarray(p_indices, dtype=np.int32)
                pressure_used_reference = p_used
                print("Requested pressure levels and selected pfull levels:")
                for pressure, index, used in zip(target_pressures, p_indices, p_used):
                    print(f"  {pressure:7.1f} Pa -> pfull[{index:2d}] = {used:8.3f} Pa")
            elif not np.array_equal(np.asarray(p_indices), pressure_indices_reference):
                raise ValueError(f"Nearest pfull indices changed in {path.name}")

            mars_hour_all = np.asarray(ds["mars_hour"].values, dtype=np.float64)
            keep_mask = np.isclose(mars_hour_all[:, None], np.asarray(target_hours)[None, :], atol=1e-6).any(axis=1)
            time_indices = np.flatnonzero(keep_mask)
            if time_indices.size == 0:
                print(f"Warning: no requested mars_hour values in {path.name}; skipping")
                continue

            for time_index in time_indices:
                if args.temperature_source == "surface":
                    temperature_kelvin = np.asarray(ds[t_name].isel(time=time_index).values, dtype=np.float32)
                else:
                    temperature_kelvin = select_near_surface(ds[t_name], int(time_index))
                ps_2d = np.asarray(ds[ps_name].isel(time=time_index).values, dtype=np.float32)
                u_3d = np.asarray(ds[u_name].isel(time=time_index).values, dtype=np.float32)
                v_3d = np.asarray(ds[v_name].isel(time=time_index).values, dtype=np.float32)
                tod_2d = np.asarray(ds[tod_name].isel(time=time_index).values, dtype=np.float32)

                t_c_list.append(temperature_kelvin - np.float32(273.15))
                ps_list.append(ps_2d)
                u_sfc_list.append(select_near_surface(ds[u_name], int(time_index)))
                v_sfc_list.append(select_near_surface(ds[v_name], int(time_index)))
                cdod610_list.append(tod_2d * (np.float32(610.0) / np.maximum(ps_2d, np.float32(1e-6))))
                for pressure, p_index in zip(target_pressures, p_indices):
                    u_level_lists[int(pressure)].append(u_3d[p_index])
                    v_level_lists[int(pressure)].append(v_3d[p_index])
                time_list.append(earth_datetime64(ds, int(time_index)))
                mars_hour_list.append(np.float32(mars_hour_all[time_index]))
                my_list.append(np.float32(ds["MY"].isel(time=time_index).values))
                ls_list.append(np.float32(ds["Ls"].isel(time=time_index).values))
        print(f"[{file_index:>2d}/{len(files)}] {path.name}: retained {time_indices.size} snapshots")

    if not time_list:
        raise RuntimeError("No snapshots were extracted. Check --hours and the source files.")

    time_array = np.asarray(time_list, dtype="datetime64[us]")
    order = np.argsort(time_array)
    t_c = stack_and_sort(t_c_list, order)
    ps = stack_and_sort(ps_list, order)
    u_sfc = stack_and_sort(u_sfc_list, order)
    v_sfc = stack_and_sort(v_sfc_list, order)
    cdod610 = stack_and_sort(cdod610_list, order)
    u_levels = {pressure: stack_and_sort(values, order) for pressure, values in u_level_lists.items()}
    v_levels = {pressure: stack_and_sort(values, order) for pressure, values in v_level_lists.items()}
    time_array = time_array[order]
    mars_hour = np.asarray(mars_hour_list, dtype=np.float32)[order]
    my = np.asarray(my_list, dtype=np.float32)[order]
    ls = np.asarray(ls_list, dtype=np.float32)[order]
    lon2d, lat2d = np.meshgrid(lon_reference, lat_reference)

    print("Extracted dynamic data:")
    print(f"  Shape: {cdod610.shape}")
    print(f"  Time range: {time_array[0]} -> {time_array[-1]}")
    print(f"  mars_hour values: {np.unique(np.round(mars_hour, 6)).tolist()}")
    print(f"  CDOD610 range: {float(np.nanmin(cdod610)):.6f} -> {float(np.nanmax(cdod610)):.6f}")

    save_dict = {
        "T_C": t_c,
        "ps": ps,
        "u_sfc": u_sfc,
        "v_sfc": v_sfc,
        "cdod610": cdod610,
        "time_dt": time_array,
        "lat": lat_reference.astype(np.float32),
        "lon": lon_reference.astype(np.float32),
        "LAT2D": lat2d.astype(np.float32),
        "LON2D": (lon2d % 360.0).astype(np.float32),
        "mars_hour": mars_hour,
        "sin_mars_hour": np.sin(2.0 * np.pi * mars_hour / 24.0).astype(np.float32),
        "cos_mars_hour": np.cos(2.0 * np.pi * mars_hour / 24.0).astype(np.float32),
        "MY": my,
        "Ls": ls,
        "uv_pfull_idx_used": pressure_indices_reference,
        "uv_pfull_pa_used": pressure_used_reference,
        "uv_target_pa": np.asarray(target_pressures, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps({
            "temperature_source": args.temperature_source,
            "target_hours": list(target_hours),
            "cdod610_formula": "tod * 610 / ps",
            "wind_levels": "nearest pfull level; no vertical interpolation",
        })),
    }
    for pressure in target_pressures:
        save_dict[f"u_{int(pressure)}Pa"] = u_levels[int(pressure)]
        save_dict[f"v_{int(pressure)}Pa"] = v_levels[int(pressure)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if args.compressed else np.savez
    save(args.output, **save_dict)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
