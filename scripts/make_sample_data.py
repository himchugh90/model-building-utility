"""Export a built-in sklearn dataset to data/<name>.csv.

Usage (from anywhere):
    python scripts/make_sample_data.py                 # iris -> data/iris.csv
    python scripts/make_sample_data.py diabetes        # -> data/diabetes.csv
    python scripts/make_sample_data.py wine --target-name label

The label column is called 'species' for iris and 'target' for everything else,
unless --target-name is given. Also writes <name>_features_only.csv: the same
rows without the label, handy for testing the `predict` command.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets this script run from anywhere

from mlbuilder.adapters import SklearnDatasetLoader  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default="iris",
                    choices=["iris", "wine", "breast_cancer", "diabetes"])
parser.add_argument("--target-name", default=None)
args = parser.parse_args()

target = args.target_name or ("species" if args.dataset == "iris" else "target")
out_dir = ROOT / "data"
out_dir.mkdir(exist_ok=True)

df = SklearnDatasetLoader(args.dataset).load().rename(columns={"target": target})
df.to_csv(out_dir / f"{args.dataset}.csv", index=False)
df.drop(columns=target).to_csv(out_dir / f"{args.dataset}_features_only.csv", index=False)
print(f"Wrote data/{args.dataset}.csv ({len(df)} rows, label column '{target}') "
      f"and data/{args.dataset}_features_only.csv")
