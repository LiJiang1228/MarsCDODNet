from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


FILE_DESCRIPTIONS = {
    "X_dynamic_example.npy": "Normalized dynamic model input: [sample, history, latitude, longitude, channel].",
    "y_cdod_example.npy": "Normalized future CDOD610 target: [sample, forecast, latitude, longitude, channel].",
    "static_terrain_example.npz": "Eight high-resolution static terrain and geographic input channels.",
    "normalization_stats.pkl": "Global normalization statistics for the dynamic EMARS variables.",
    "metadata_example.npz": "Sample timing, Mars-year, solar-longitude, grid, and channel metadata.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MarsCDODNet example-data files.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--show-ranges", action="store_true", help="Print min/max values for numeric arrays.")
    return parser.parse_args()


def format_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def describe_array(name: str, array: np.ndarray, show_ranges: bool) -> None:
    print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
    if show_ranges and np.issubdtype(array.dtype, np.number) and array.size:
        print(f"    finite={np.isfinite(array).all()}, min={float(np.nanmin(array)):.6g}, max={float(np.nanmax(array)):.6g}")


def inspect_npy(path: Path, show_ranges: bool) -> None:
    array = np.load(path, allow_pickle=False)
    describe_array(path.stem, array, show_ranges)


def inspect_npz(path: Path, show_ranges: bool) -> None:
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            describe_array(name, archive[name], show_ranges)


def inspect_stats(path: Path) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    log_vars = payload.get("log_vars", []) if isinstance(payload, dict) else []
    print(f"  Variables with statistics ({len(stats)}): {', '.join(stats)}")
    print(f"  Log-transformed variables: {', '.join(log_vars)}")


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)

    paths = sorted(path for path in args.data_dir.iterdir() if path.is_file() and path.name != Path(__file__).name)
    if not paths:
        raise FileNotFoundError(f"No files found in {args.data_dir}")

    print(f"Example-data directory: {args.data_dir}")
    print(f"Files: {len(paths)}")
    for path in paths:
        print(f"\n{path.name} ({format_size(path.stat().st_size)})")
        description = FILE_DESCRIPTIONS.get(path.name)
        if description:
            print(f"  {description}")
        if path.suffix == ".npy":
            inspect_npy(path, args.show_ranges)
        elif path.suffix == ".npz":
            inspect_npz(path, args.show_ranges)
        elif path.suffix == ".pkl":
            inspect_stats(path)
        else:
            print("  Text or auxiliary file.")


if __name__ == "__main__":
    main()
