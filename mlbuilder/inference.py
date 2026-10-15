"""Scoring helpers shared by the training run and the `predict` command."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import pandas as pd
import sklearn

from .errors import DataError

logger = logging.getLogger("mlbuilder")


def predict_frame(
    model: Any, features: List[str], df: pd.DataFrame, categorical: Sequence[str] = ()
) -> pd.DataFrame:
    """Score a DataFrame. Columns are matched by NAME (order is irrelevant) and
    extra columns, such as the label, are ignored. `categorical` lists the columns that were
    one-hot encoded at training time (allowed to be text). Bad input raises DataError."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise DataError(
            f"Input is missing required columns: {missing}. "
            f"Input has columns: {list(df.columns)}"
        )
    X = df[features]
    non_numeric = [
        c for c in features if c not in categorical and not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric:
        raise DataError(f"Non-numeric values found in feature columns: {non_numeric}")
    try:
        out = pd.DataFrame({"prediction": model.predict(X)}, index=X.index)
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    except ValueError as exc:  # e.g. NaN values and no imputer in the saved pipeline
        first_line = str(exc).splitlines()[0]
        raise DataError(
            f"Could not score input: {first_line} "
            "(if this is about missing values, retrain with a 'simple_imputer' step)"
        ) from exc
    except AttributeError:  # estimator exposes predict_proba but cannot compute it
        proba = None
    if proba is not None:
        for cls, col in zip(model.classes_, proba.T):
            out[f"proba_{cls}"] = col
    return out


def load_run(run_dir: "str | Path") -> Tuple[Any, Dict[str, Any]]:
    run_dir = Path(run_dir)
    model_path, meta_path = run_dir / "model.joblib", run_dir / "run_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise DataError(f"'{run_dir}' is not a valid run directory (model or metadata missing)")
    meta = json.loads(meta_path.read_text())
    trained_with = meta.get("versions", {}).get("scikit-learn")
    if trained_with != sklearn.__version__:
        logger.warning(
            "Model was trained with scikit-learn %s but %s is installed; results may differ",
            trained_with, sklearn.__version__,
        )
    return joblib.load(model_path), meta
