# Example data

This folder contains one MY25 global-dust-storm example for checking the data
and model workflow. It is not a training dataset.

```text
X_dynamic_example.npy       [1, 40, 36, 60, 23]
y_cdod_example.npy          [1, 12, 36, 60, 1]
static_terrain_example.npz  static_features=[8, 720, 1440]
```

The dynamic channels are CDOD, temperature, pressure, surface and pressure-
level winds, and six astronomical features. Inspect the files with:

```bash
python inspect_example_data.py --show-ranges
```
