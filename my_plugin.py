"""Example plugin: adds a model and a metric WITHOUT touching the core package.

Enable it in a config with:   plugins: [my_plugin]
"""
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score

from mlbuilder import METRICS, MODELS

MODELS.add(
    "extra_trees",
    {"classification": ExtraTreesClassifier, "regression": ExtraTreesRegressor},
)
METRICS.add(
    "balanced_accuracy", balanced_accuracy_score,
    tasks=("classification",), greater_is_better=True,
)
