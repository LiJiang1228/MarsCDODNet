from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


DYNAMIC_FIELDS = (
    "T_C",
    "ps",
    "u_sfc",
    "v_sfc",
    "cdod610",
    "u_10Pa",
    "v_10Pa",
    "u_40Pa",
    "v_40Pa",
    "u_100Pa",
    "v_100Pa",
    "u_300Pa",
    "v_300Pa",
    "u_500Pa",
    "v_500Pa",
    "u_610Pa",
    "v_610Pa",
)
CHANNEL_NAMES = (
    "cdod610",
    "T_C",
    "ps",
    "u_sfc",
    "v_sfc",
    "u_610Pa",
    "v_610Pa",
    "u_500Pa",
    "v_500Pa",
    "u_300Pa",
    "v_300Pa",
    "u_100Pa",
    "v_100Pa",
    "u_40Pa",
    "v_40Pa",
    "u_10Pa",
    "v_10Pa",
    "mars_sun_distance",
    "cos_sza",
    "sin_Ls",
    "cos_Ls",
    "sin_LTST",
    "cos_LTST",
)
DUST_SEASON_DEFINITIONS = (
    ((24, 140.637), (25, 14.950)),
    ((25, 142.894), (26, 20.490)),
    ((26, 145.168), (27, 22.127)),
    ((27, 132.162), (28, 5.113)),
    ((28, 147.667), (29, 25.855)),
    ((29, 138.599), (30, 16.904)),
    ((30, 152.834), (31, 19.525)),
    ((31, 146.207), (32, 18.265)),
    ((32, 152.197), (33, 19.428)),
    ((33, 150.821), (34, 22.521)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 23-channel MarsCDODNet dataset with dust/calm block splitting.")
    parser.add_argument("--dynamic-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--input-steps", type=int, default=40)
    parser.add_argument("--output-steps", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--special-years", type=int, nargs=2, default=(25, 28))
    parser.add_argument("--sza-chunk", type=int, default=2048)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class MarsDataNormalizer:
    def __init__(self) -> None:
        self.stats: dict[str, dict[str, float | str]] = {}
        self.log_vars = ("cdod610",)
        self.epsilon = np.float32(1e-6)

    def fit_transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalised: dict[str, np.ndarray] = {}
        print(f"{'Variable':<12} {'Shape':<20} {'Mean':>12} {'Std':>12} Method")
        for name in DYNAMIC_FIELDS:
            values = np.asarray(data[name], dtype=np.float32)
            if name in self.log_vars:
                working = np.log(values + self.epsilon)
                method = "Log + Z-Score"
            else:
                working = values
                method = "Z-Score"
            mean = np.mean(working)
            std = np.std(working)
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid standard deviation for {name}: {std}")
            self.stats[name] = {"mean": float(mean), "std": float(std), "method": method}
            normalised[name] = ((working - mean) / std).astype(np.float32)
            print(f"{name:<12} {str(values.shape):<20} {float(mean):>12.6f} {float(std):>12.6f} {method}")
        return normalised

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"stats": self.stats, "log_vars": list(self.log_vars)}, handle)
        print(f"Saved normalization statistics: {path}")


def compute_astronomical_features(lat2d: np.ndarray, lon2d: np.ndarray, ls: np.ndarray, mars_hour: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    if lat2d.shape != lon2d.shape:
        raise ValueError(f"Latitude/longitude shape mismatch: {lat2d.shape} vs {lon2d.shape}")
    if ls.shape != mars_hour.shape:
        raise ValueError(f"Ls/mars_hour length mismatch: {ls.shape} vs {mars_hour.shape}")
    if chunk_size <= 0:
        raise ValueError("--sza-chunk must be positive")

    height, width = lat2d.shape
    n_time = ls.size
    ls_rad = np.deg2rad(ls.astype(np.float64))
    distance_au = 1.52367934 * (1.0 - 0.09341233**2) / (1.0 + 0.09341233 * np.cos(ls_rad - np.deg2rad(250.99)))
    mars_sun_distance = np.broadcast_to(((distance_au - 1.38) / (1.67 - 1.38))[:, None, None], (n_time, height, width)).astype(np.float32)

    lat_rad = np.deg2rad(lat2d.astype(np.float64))
    lon_rad = np.deg2rad(np.mod(lon2d, 360.0).astype(np.float64))
    solar_declination = np.arcsin(np.sin(np.deg2rad(25.1919)) * np.sin(ls_rad))
    hour_angle = np.deg2rad(15.0 * (mars_hour.astype(np.float64) - 12.0))
    cos_sza = np.empty((n_time, height, width), dtype=np.float32)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)

    for start in range(0, n_time, chunk_size):
        stop = min(start + chunk_size, n_time)
        current = (
            np.sin(solar_declination[start:stop, None, None]) * sin_lat[None, :, :]
            + np.cos(solar_declination[start:stop, None, None])
            * cos_lat[None, :, :]
            * np.cos(hour_angle[start:stop, None, None] + lon_rad[None, :, :])
        )
        cos_sza[start:stop] = np.clip(current, -1.0, 1.0).astype(np.float32)
        print(f"Computed cos_sza: {start}:{stop}")

    sin_ls = np.broadcast_to(np.sin(ls_rad)[:, None, None], (n_time, height, width)).astype(np.float32)
    cos_ls = np.broadcast_to(np.cos(ls_rad)[:, None, None], (n_time, height, width)).astype(np.float32)
    ltst = (mars_hour.astype(np.float64)[:, None, None] + lon2d[None, :, :] / 15.0) % 24.0
    phase = 2.0 * np.pi * ltst / 24.0
    return {
        "mars_sun_distance": mars_sun_distance,
        "cos_sza": cos_sza,
        "sin_Ls": sin_ls,
        "cos_Ls": cos_ls,
        "sin_LTST": np.sin(phase).astype(np.float32),
        "cos_LTST": np.cos(phase).astype(np.float32),
    }


def split_dust_and_calm_blocks(my: np.ndarray, ls: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    continuous_ls = my.astype(np.float64) * 360.0 + ls.astype(np.float64)
    intervals = tuple(
        (start_my * 360.0 + start_ls, end_my * 360.0 + end_ls)
        for (start_my, start_ls), (end_my, end_ls) in DUST_SEASON_DEFINITIONS
    )
    dusty_blocks: list[np.ndarray] = []
    calm_blocks: list[np.ndarray] = []
    start = 0
    current_is_dusty = any(lower <= continuous_ls[0] <= upper for lower, upper in intervals)

    for index in range(1, continuous_ls.size):
        is_dusty = any(lower <= continuous_ls[index] <= upper for lower, upper in intervals)
        if is_dusty != current_is_dusty:
            target = dusty_blocks if current_is_dusty else calm_blocks
            target.append(np.arange(start, index, dtype=np.int64))
            start = index
            current_is_dusty = is_dusty

    target = dusty_blocks if current_is_dusty else calm_blocks
    target.append(np.arange(start, continuous_ls.size, dtype=np.int64))
    print(f"Dusty blocks: {len(dusty_blocks)}")
    print(f"Calm blocks: {len(calm_blocks)}")
    return dusty_blocks, calm_blocks


def split_blocks_capacity_constrained(
    blocks: list[np.ndarray],
    my: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
    special_years: tuple[int, ...] = (),
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not blocks:
        raise ValueError("No blocks are available for splitting")
    total_length = sum(len(block) for block in blocks)
    target_train_length = int(total_length * train_fraction)
    special_blocks = {year: [] for year in special_years}
    ordinary_blocks: list[np.ndarray] = []

    for block in blocks:
        start_my = int(my[block[0]])
        if start_my in special_blocks:
            special_blocks[start_my].append(block)
        else:
            ordinary_blocks.append(block)

    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    train_length = 0
    if special_years:
        missing = [year for year in special_years if not special_blocks[year]]
        if missing:
            names = ", ".join(f"MY{year}" for year in missing)
            raise ValueError(f"Global dust-storm blocks were not found for {names}")
        special_order = list(rng.permutation(np.asarray(special_years, dtype=np.int64)))
        train_year = int(special_order[0])
        test_year = int(special_order[1])
        train_blocks.extend(special_blocks[train_year])
        test_blocks.extend(special_blocks[test_year])
        train_length += sum(len(block) for block in special_blocks[train_year])
        for year in special_order[2:]:
            ordinary_blocks.extend(special_blocks[int(year)])
        print(f"Separated global dust storms: MY{train_year} -> train, MY{test_year} -> test")

    for index in rng.permutation(len(ordinary_blocks)):
        block = ordinary_blocks[int(index)]
        if train_length + len(block) <= target_train_length:
            train_blocks.append(block)
            train_length += len(block)
        else:
            test_blocks.append(block)

    if not train_blocks or not test_blocks:
        raise ValueError("Capacity-constrained splitting produced an empty train or test block set")
    return train_blocks, test_blocks


def build_dataset_from_blocks(
    blocks: list[np.ndarray],
    inputs: np.ndarray,
    target: np.ndarray,
    input_steps: int,
    output_steps: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    total_steps = input_steps + output_steps
    height, width, channels = inputs.shape[1:]
    sample_count = sum(max(0, len(block) - total_steps + 1) for block in blocks)
    valid_blocks = sum(len(block) >= total_steps for block in blocks)
    if sample_count == 0:
        raise ValueError(f"{name} has no block long enough for {total_steps} time steps")
    print(f"{name}: {valid_blocks}/{len(blocks)} valid blocks, {sample_count} samples")

    x = np.empty((sample_count, input_steps, height, width, channels), dtype=np.float32)
    y = np.empty((sample_count, output_steps, height, width, 1), dtype=np.float32)
    output_index = 0
    for block in blocks:
        if len(block) < total_steps:
            continue
        block_start = int(block[0])
        for offset in range(len(block) - total_steps + 1):
            start = block_start + offset
            x[output_index] = inputs[start : start + input_steps]
            y[output_index, ..., 0] = target[start + input_steps : start + total_steps]
            output_index += 1
    return x, y


def block_table(blocks: list[np.ndarray], my: np.ndarray, ls: np.ndarray) -> np.ndarray:
    rows = [
        (int(block[0]), int(block[-1]), len(block), int(my[block[0]]), int(my[block[-1]]), float(ls[block[0]]), float(ls[block[-1]]))
        for block in blocks
    ]
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.input_steps <= 0 or args.output_steps <= 0:
        raise ValueError("--input-steps and --output-steps must be positive")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Pass --overwrite to replace it.")
    if args.stats_output.exists() and not args.overwrite:
        raise FileExistsError(f"Statistics file already exists: {args.stats_output}. Pass --overwrite to replace it.")

    with np.load(args.dynamic_path, allow_pickle=False) as data:
        required = (*DYNAMIC_FIELDS, "LAT2D", "LON2D", "Ls", "mars_hour", "MY")
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Dynamic input is missing: {', '.join(missing)}")
        raw = {name: np.asarray(data[name], dtype=np.float32) for name in DYNAMIC_FIELDS}
        lat2d = np.asarray(data["LAT2D"], dtype=np.float32)
        lon2d = np.asarray(data["LON2D"], dtype=np.float32)
        ls = np.asarray(data["Ls"], dtype=np.float32)
        mars_hour = np.asarray(data["mars_hour"], dtype=np.float32)
        my = np.asarray(data["MY"], dtype=np.int64)

    dynamic_shape = raw["cdod610"].shape
    if len(dynamic_shape) != 3 or dynamic_shape[1:] != lat2d.shape or lat2d.shape != lon2d.shape:
        raise ValueError("Dynamic fields must be [time, lat, lon] and match LAT2D/LON2D")
    if any(values.shape != dynamic_shape for values in raw.values()):
        raise ValueError("All dynamic fields must have the same shape")
    if ls.shape != (dynamic_shape[0],) or mars_hour.shape != (dynamic_shape[0],) or my.shape != (dynamic_shape[0],):
        raise ValueError("Ls, mars_hour, and MY must have one value per dynamic time step")

    normalizer = MarsDataNormalizer()
    normalised = normalizer.fit_transform(raw)
    auxiliary = compute_astronomical_features(lat2d, lon2d, ls, mars_hour, args.sza_chunk)
    fields = {**normalised, **auxiliary}
    inputs = np.stack([fields[name] for name in CHANNEL_NAMES], axis=-1).astype(np.float32)

    dusty_blocks, calm_blocks = split_dust_and_calm_blocks(my, ls)
    rng = np.random.default_rng(args.split_seed)
    special_years = tuple(int(year) for year in args.special_years)
    train_dusty_blocks, test_dusty_blocks = split_blocks_capacity_constrained(
        dusty_blocks, my, args.train_fraction, rng, special_years
    )
    train_calm_blocks, test_calm_blocks = split_blocks_capacity_constrained(
        calm_blocks, my, args.train_fraction, rng
    )
    train_blocks = train_dusty_blocks + train_calm_blocks
    test_blocks = test_dusty_blocks + test_calm_blocks
    train_storm_years = {int(my[block[0]]) for block in train_dusty_blocks if int(my[block[0]]) in special_years}
    test_storm_years = {int(my[block[0]]) for block in test_dusty_blocks if int(my[block[0]]) in special_years}
    if len(train_storm_years) != 1 or len(test_storm_years) != 1 or train_storm_years == test_storm_years:
        raise ValueError("MY25 and MY28 global dust storms were not assigned to separate train and test sets")

    train_steps = sum(len(block) for block in train_blocks)
    test_steps = sum(len(block) for block in test_blocks)
    print(f"Train blocks: {len(train_dusty_blocks)} dusty + {len(train_calm_blocks)} calm, {train_steps} steps")
    print(f"Test blocks: {len(test_dusty_blocks)} dusty + {len(test_calm_blocks)} calm, {test_steps} steps")
    print(f"Global dust storms: train={sorted(train_storm_years)}, test={sorted(test_storm_years)}")

    x_train, y_train = build_dataset_from_blocks(train_blocks, inputs, normalised["cdod610"], args.input_steps, args.output_steps, "train")
    x_test, y_test = build_dataset_from_blocks(test_blocks, inputs, normalised["cdod610"], args.input_steps, args.output_steps, "test")
    print(f"Full input tensor: {inputs.shape}")
    print(f"Train tensor: {x_train.shape} -> {y_train.shape}")
    print(f"Test tensor: {x_test.shape} -> {y_test.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compressed else np.savez
    saver(
        args.output,
        X_train=x_train,
        y_train=y_train,
        X_val=x_test,
        y_val=y_test,
        X_test=x_test,
        y_test=y_test,
        channel_names=np.asarray(CHANNEL_NAMES),
        input_steps=np.asarray(args.input_steps, dtype=np.int64),
        output_steps=np.asarray(args.output_steps, dtype=np.int64),
        split_seed=np.asarray(args.split_seed, dtype=np.int64),
        train_fraction=np.asarray(args.train_fraction, dtype=np.float32),
        validation_source=np.asarray("test_blocks"),
        train_dusty_blocks=block_table(train_dusty_blocks, my, ls),
        test_dusty_blocks=block_table(test_dusty_blocks, my, ls),
        train_calm_blocks=block_table(train_calm_blocks, my, ls),
        test_calm_blocks=block_table(test_calm_blocks, my, ls),
    )
    normalizer.save(args.stats_output)
    print(f"Saved dataset: {args.output}")


if __name__ == "__main__":
    main()
