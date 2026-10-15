"""Built-in components. Importing this module populates the registries.

To extend the utility, create your own module with the same pattern and list
it under `plugins:` in the config (see README).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .preprocess import OutlierReplacer
from .registry import METRICS, MODELS, PREPROCESSORS, SPLITTERS

# ---- preprocessors -------------------------------------------------------
PREPROCESSORS.add("standard_scaler", StandardScaler)
PREPROCESSORS.add("minmax_scaler", MinMaxScaler)
PREPROCESSORS.add("robust_scaler", RobustScaler)
# handles_missing=True tells the NaN guard that this step fills missing values
PREPROCESSORS.add("simple_imputer", SimpleImputer, handles_missing=True)
PREPROCESSORS.add("knn_imputer", KNNImputer, handles_missing=True)
PREPROCESSORS.add("outlier_replacer", OutlierReplacer)

# ---- models (task -> estimator class) -----------------------------------
MODELS.add("logistic_regression", {"classification": LogisticRegression})
MODELS.add("linear_regression", {"regression": LinearRegression})
MODELS.add("ridge", {"regression": Ridge})
MODELS.add(
    "decision_tree",
    {"classification": DecisionTreeClassifier, "regression": DecisionTreeRegressor},
)
MODELS.add(
    "random_forest",
    {"classification": RandomForestClassifier, "regression": RandomForestRegressor},
)
MODELS.add(
    "gradient_boosting",
    {"classification": GradientBoostingClassifier, "regression": GradientBoostingRegressor},
)
MODELS.add(
    "knn", {"classification": KNeighborsClassifier, "regression": KNeighborsRegressor}
)
MODELS.add("svm", {"classification": SVC, "regression": SVR})

# ---- metrics: fn(y_true, y_pred) -> float --------------------------------
CLS = ("classification",)
REG = ("regression",)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRICS.add("accuracy", accuracy_score, tasks=CLS, greater_is_better=True)
METRICS.add(
    "f1_macro",
    lambda yt, yp: f1_score(yt, yp, average="macro"),
    tasks=CLS,
    greater_is_better=True,
)
METRICS.add(
    "f1_weighted",
    lambda yt, yp: f1_score(yt, yp, average="weighted"),
    tasks=CLS,
    greater_is_better=True,
)
METRICS.add(
    "precision_macro",
    lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
    tasks=CLS,
    greater_is_better=True,
)
METRICS.add(
    "recall_macro",
    lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
    tasks=CLS,
    greater_is_better=True,
)
METRICS.add("r2", r2_score, tasks=REG, greater_is_better=True)
METRICS.add("rmse", _rmse, tasks=REG, greater_is_better=False)
METRICS.add("mae", mean_absolute_error, tasks=REG, greater_is_better=False)


# ---- split strategies: fn(X, y, cfg) -> X_train, X_test, y_train, y_test ---
def _random_split(X, y, cfg):
    stratify = y if (cfg.task == "classification" and cfg.split.stratify) else None
    return train_test_split(
        X, y, test_size=cfg.split.test_size, random_state=cfg.seed, stratify=stratify
    )


def _ordered_split(X, y, cfg):
    """No shuffling: the last rows become the test set (use for time-ordered data)."""
    n_test = max(1, int(round(len(X) * cfg.split.test_size)))
    cut = len(X) - n_test
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


SPLITTERS.add("random", _random_split)
SPLITTERS.add("ordered", _ordered_split)
