# ML_Assignment_2

Dataset
-
Place the obesity dataset CSV at `data/ObesityDataSet_raw_and_data_sinthetic.csv` so the loader can find it.

You can either:
- copy the file into `data/` yourself, or
- run the helper script to download or copy it:

```bash
python scripts/fetch_dataset.py --url <DIRECT_CSV_URL>
# or
python scripts/fetch_dataset.py --local /path/to/ObesityDataSet_raw_and_data_sinthetic.csv
```

After placing the file, run the model training:

```bash
PYTHONPATH=. python3 -m model.xgboost_model
```

ML Assignment 2. This Repository is for implementation of multiple classification models. It contains complete, requirements.txt,  source code and README.md
