"""Creates data/iris_messy.csv: iris plus missing values, outliers and a text column.

Use it to see the automatic preprocessing at work:
    python scripts/make_messy_data.py
    python -m mlbuilder train --config configs/messy_iris.yaml
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlbuilder.adapters import SklearnDatasetLoader  # noqa: E402

rng = np.random.default_rng(0)
df = SklearnDatasetLoader("iris").load().rename(columns={"target": "species"})
df["region"] = rng.choice(["north", "south", "east", "west"], len(df))      # categorical feature
for col in ["sepal_length_cm", "petal_width_cm"]:                            # missing numeric values
    df.loc[rng.choice(len(df), 8, replace=False), col] = np.nan
df.loc[rng.choice(len(df), 3, replace=False), "sepal_width_cm"] *= 10        # outliers
df.loc[rng.choice(len(df), 4, replace=False), "region"] = np.nan             # missing categorical
out = ROOT / "data" / "iris_messy.csv"
df.to_csv(out, index=False)
df.drop(columns="species").head(6).to_csv(ROOT / "data" / "iris_messy_new_rows.csv", index=False)
print(f"Wrote {out} ({len(df)} rows) and data/iris_messy_new_rows.csv (unlabelled rows to score)")
