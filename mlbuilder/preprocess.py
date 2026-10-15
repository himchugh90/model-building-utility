"""Automatic preprocessing: outliers, missing values, categorical encoding.

Everything here is fit on TRAINING data only and lives inside the sklearn Pipeline, so
(a) there is no leakage, including inside cross-validation folds, and (b) scoring a new
file applies exactly the rules learned at training time. It only ever touches features,
never the target, so it works the same for classification and regression.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .config import AutoPreprocessConfig


class OutlierReplacer(BaseEstimator, TransformerMixin):
    """IQR rule per column. Values outside [Q1 - factor*IQR, Q3 + factor*IQR] are replaced
    with the mean (or median) of the NON-outlier training values.

    * Bounds and replacement values are learned in `fit` (train only); `transform` reuses them.
    * Using the mean of inliers matters: the plain mean is dragged by the very outliers
      being removed.
    * Columns with IQR == 0 (constants, rare-event 0/1 flags) are never treated, otherwise
      every "1" in a mostly-zero flag would be called an outlier.
    * NaNs pass through untouched, to be filled by the imputer that follows.
    """

    def __init__(self, factor: float = 1.5, replace_with: str = "mean") -> None:
        self.factor = factor
        self.replace_with = replace_with

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            active = iqr > 0
            self.lower_ = np.where(active, q1 - self.factor * iqr, -np.inf)
            self.upper_ = np.where(active, q3 + self.factor * iqr, np.inf)
            outlier = (X < self.lower_) | (X > self.upper_)
            self.n_outliers_ = outlier.sum(axis=0)
            inliers = np.where(outlier, np.nan, X)
            stat = np.nanmedian if self.replace_with == "median" else np.nanmean
            self.fill_ = stat(inliers, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "fill_")
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore"):
            outlier = (X < self.lower_) | (X > self.upper_)
        return np.where(outlier, self.fill_, X)


def build_auto_preprocessor(ap: AutoPreprocessConfig) -> ColumnTransformer:
    """Numeric columns: outliers -> scale -> KNN imputation.
    Non-numeric columns: most-frequent fill -> one-hot (unseen categories ignored at scoring).

    Scaling is part of the numeric branch because KNN imputation is distance based and
    would otherwise be dominated by the largest-scale column.
    """
    numeric = []
    if ap.outliers:
        numeric.append(("outliers", OutlierReplacer(ap.outlier_factor, ap.replace_with)))
    numeric += [
        ("scale", StandardScaler()),
        ("impute", KNNImputer(n_neighbors=ap.knn_neighbors)),
    ]
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric), make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ],
        sparse_threshold=0,  # always dense output
    )


def describe_auto_preprocessing(model: Any, X_train: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """What the automatic step did on the training data (stored in run_meta.json)."""
    auto = getattr(model, "named_steps", {}).get("auto")
    if auto is None:
        return None
    numeric = list(X_train.select_dtypes("number").columns)
    report: Dict[str, Any] = {
        "numeric_columns": numeric,
        "categorical_columns": list(X_train.select_dtypes(exclude="number").columns),
        "missing_values_in_train": {c: int(n) for c, n in X_train.isna().sum().items() if n},
    }
    try:
        rep = auto.named_transformers_["num"].named_steps["outliers"]
        report["outliers_replaced_in_train"] = {
            col: {"count": int(n), "lower": float(lo), "upper": float(hi), "replaced_with": float(fill)}
            for col, n, lo, hi, fill in zip(numeric, rep.n_outliers_, rep.lower_, rep.upper_, rep.fill_)
            if n > 0
        }
    except (KeyError, AttributeError):
        pass  # outlier treatment disabled, or no numeric columns
    return report
