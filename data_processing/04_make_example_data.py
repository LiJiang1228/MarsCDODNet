from __future__ import annotations

import argparse
import importlib.util
import pickle
import shutil
from pathlib import Path

import numpy as np


MY25_GLOBAL_STORM = ((25, 142.894), (26, 20.490))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one MY25 global-dust-storm example for MarsCDODNet.")
    parser.add_argument("--dynamic-path", type=Path, required=True)
    parser.add_argument("--static-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data_example")
    parser.add_argument("--storm-my", type=int, default=25)
    parser.add_argument("--input-steps", type=int, default=40)
    parser.add_argument("--output-steps", type=int, default=12)
    parser.add_argument("--sza-chunk", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dataset_builder():
    path = Path(__file__).with_name("03_build_dataset.py")
    spec = importlib.util.spec_from_file_location("marscdodnet_dataset_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_my25_window(my: np.ndarray, ls: np.ndarray, required_steps: int, storm_my: int) -> int:
    if storm_my != 25:
        raise ValueError("This exporter currently supports the MY25 global dust storm only")
    (start_my, start_ls), (end_my, end_ls) = MY25_GLOBAL_STORM
    continuous_ls = my.astype(np.float64) * 360.0 + ls.astype(np.float64)
    lower = start_my * 360.0 + start_ls
    upper = end_my * 360.0 + end_ls
    selected = (continuous_ls >= lower) & (continuous_ls <= upper)
    indices = np.flatnonzero(selected)
    if indices.size < required_steps:
        raise ValueError(f"MY{storm_my} global dust storm has only {indices.size} time steps")

    run_start = 0
    for position in range(1, indices.size + 1):
        is_end = position == indices.size or indices[position] != indices[position - 1] + 1
        if is_end:
            run = indices[run_start:position]
            if run.size >= required_steps:
                return int(run[0])
            run_start = position
    raise ValueError(f"MY{storm_my} global dust storm has no contiguous {required_steps}-step window")


def normalise(values: np.ndarray, name: str, stats: dict[str, object], log_vars: set[str]) -> np.ndarray:
    if name not in stats:
        raise KeyError(f"Normalization statistics are missing {name}")
    entry = stats[name]
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid normalization entry for {name}")
    working = np.log(values.astype(np.float32) + np.float32(1e-6)) if name in log_vars else values.astype(np.float32)
    return ((working - float(entry["mean"])) / float(entry["std"])).astype(np.float32)


def validate_destination(output_dir: Path, overwrite: bool) -> None:
    targets = (
        "X_dynamic_example.npy",
        "y_cdod_example.npy",
        "static_terrain_example.npz",
        "normalization_stats.pkl",
        "metadata_example.npz",
    )
    existing = [str(output_dir / name) for name in targets if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("Example files already exist. Pass --overwrite to replace: " + ", ".join(existing))


def main() -> None:
    args = parse_args()
    if args.input_steps <= 0 or args.output_steps <= 0:
        raise ValueError("--input-steps and --output-steps must be positive")
    if args.sza_chunk <= 0:
        raise ValueError("--sza-chunk must be positive")
    if not args.dynamic_path.is_file() or not args.static_path.is_file() or not args.stats_path.is_file():
        raise FileNotFoundError("--dynamic-path, --static-path, and --stats-path must all exist")

    validate_destination(args.output_dir, args.overwrite)
    builder = load_dataset_builder()
    total_steps = args.input_steps + args.output_steps
    with args.stats_path.open("rb") as handle:
        stats_file = pickle.load(handle)
    stats = stats_file.get("stats")
    log_vars = set(stats_file.get("log_vars", ()))
    if not isinstance(stats, dict):
        raise TypeError("Statistics file must contain a 'stats' dictionary")

    with np.load(args.dynamic_path, allow_pickle=False) as data:
        required = (*builder.DYNAMIC_FIELDS, "lat", "lon", "time_dt", "mars_hour", "MY", "Ls")
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError("Dynamic source is missing: " + ", ".join(missing))
        my = np.asarray(data["MY"], dtype=np.int64)
        ls = np.asarray(data["Ls"], dtype=np.float32)
        start = find_my25_window(my, ls, total_steps, args.storm_my)
        stop = start + total_steps
        lat1d = np.asarray(data["lat"], dtype=np.float32)
        lon1d = np.asarray(data["lon"], dtype=np.float32)
        lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
        time_dt = np.asarray(data["time_dt"][start:stop])
        window_my = my[start:stop]
        window_ls = ls[start:stop]
        window_hour = np.asarray(data["mars_hour"][start:stop], dtype=np.float32)
        normalised = {
            name: normalise(np.asarray(data[name][start:stop], dtype=np.float32), name, stats, log_vars)
            for name in builder.DYNAMIC_FIELDS
        }

    auxiliary = builder.compute_astronomical_features(lat2d, lon2d, window_ls, window_hour, args.sza_chunk)
    fields = {**normalised, **auxiliary}
    full_inputs = np.stack([fields[name] for name in builder.CHANNEL_NAMES], axis=-1).astype(np.float32)
    x_example = full_inputs[: args.input_steps][None, ...]
    y_example = normalised["cdod610"][args.input_steps :][..., None][None, ...]
    if x_example.shape[1:] != (args.input_steps, 36, 60, len(builder.CHANNEL_NAMES)):
        raise ValueError(f"Unexpected dynamic example shape: {x_example.shape}")
    if y_example.shape[1:] != (args.output_steps, 36, 60, 1):
        raise ValueError(f"Unexpected target example shape: {y_example.shape}")
    if not np.all(np.isfinite(x_example)) or not np.all(np.isfinite(y_example)):
        raise ValueError("Example data contains non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X_dynamic_example.npy", x_example)
    np.save(args.output_dir / "y_cdod_example.npy", y_example)
    shutil.copy2(args.static_path, args.output_dir / "static_terrain_example.npz")
    shutil.copy2(args.stats_path, args.output_dir / "normalization_stats.pkl")
    np.savez_compressed(
        args.output_dir / "metadata_example.npz",
        channel_names=np.asarray(builder.CHANNEL_NAMES),
        input_steps=np.asarray(args.input_steps, dtype=np.int64),
        output_steps=np.asarray(args.output_steps, dtype=np.int64),
        sample_kind=np.asarray("global_dust_storm"),
        global_storm_my=np.asarray(args.storm_my, dtype=np.int64),
        source_dynamic_file=np.asarray(args.dynamic_path.name),
        window_start_index=np.asarray(start, dtype=np.int64),
        window_stop_index=np.asarray(stop, dtype=np.int64),
        input_time_dt=time_dt[: args.input_steps],
        target_time_dt=time_dt[args.input_steps :],
        input_MY=window_my[: args.input_steps],
        target_MY=window_my[args.input_steps :],
        input_Ls=window_ls[: args.input_steps],
        target_Ls=window_ls[args.input_steps :],
        input_mars_hour=window_hour[: args.input_steps],
        target_mars_hour=window_hour[args.input_steps :],
        LAT2D=lat2d,
        LON2D=np.mod(lon2d, 360.0).astype(np.float32),
    )
    print(f"Selected MY{args.storm_my} global dust-storm window: {start}:{stop}")
    print(f"Saved dynamic input: {x_example.shape}")
    print(f"Saved CDOD target: {y_example.shape}")
    print(f"Saved example directory: {args.output_dir}")


if __name__ == "__main__":
    main()
