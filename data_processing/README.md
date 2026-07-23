# Data processing

Run the scripts in order:

```bash
python 01_process_dynamic_data.py --input-dir /path/to/emars --output dynamic.npz
python 02_build_static_features.py --mola-path /path/to/mola.nc --latlon-path dynamic.npz --output static.npz
python 03_build_dataset.py --dynamic-path dynamic.npz --output dataset.npz --stats-output stats.pkl
python 04_make_example_data.py --dynamic-path dynamic.npz --static-path static.npz --stats-path stats.pkl --output-dir ../data_example --overwrite
```

`01` extracts 6-hourly EMARS fields and derives CDOD at 610 Pa. 
`02` creates eight terrain/geographic channels at 720 x 1440.
`03` adds six astronomical features and builds 40-to-12 samples. 
`04` exports one MY25 example.

Do not upload the full EMARS or MOLA source data. Obtain and cite them from their official providers.
