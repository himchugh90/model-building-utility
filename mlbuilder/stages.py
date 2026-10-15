"""Pipeline stages: small functions of one argument, `ctx`.

* The run is just the stages listed under `stages:` in the config, in order.
* To add a step: write a function, decorate it with @stage(...), list its name in
  the config. Nothing else changes.
* `requires` / `provides` name RunContext attributes. They are checked before any
  data is read, so a misordered or incomplete stage list fails immediately.
"""
from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from . import components  # noqa: F401  (registers built-ins)
from .config import Config, StepConfig
from .errors import ConfigError, DataError
from .inference import predict_frame
from .pipeline import build_pipeline, candidate_labels
from .preprocess import describe_auto_preprocessing
from .ports import ArtifactStore, DataLoader, Tracker
from .registry import METRICS, PREPROCESSORS, SPLITTERS, STAGES

logger = logging.getLogger("mlbuilder")


@dataclass
class RunContext:
    """Shared state passed from stage to stage."""

    cfg: Config
    loader: DataLoader
    store: ArtifactStore
    tracker: Tracker
    started: float = field(default_factory=time.time)
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    features: Optional[List[str]] = None
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    selected: Optional[str] = None
    selected_step: Optional[StepConfig] = None
    cv_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model: Any = None
    metrics: Optional[Dict[str, float]] = None
    test_predictions: Optional[pd.DataFrame] = None
    inference_predictions: Optional[pd.DataFrame] = None
    preprocessing_report: Optional[Dict[str, Any]] = None
    categorical_features: List[str] = field(default_factory=list)


def stage(name: str, requires: Tuple[str, ...] = (), provides: Tuple[str, ...] = ()):
    return STAGES.register(name, requires=requires, provides=provides)


def validate_stage_order(names: List[str]) -> None:
    available: set = set()
    for name in names:
        meta = STAGES.get(name).meta
        missing = [r for r in meta["requires"] if r not in available]
        if missing:
            raise ConfigError(
                f"Stage '{name}' needs {missing}, but no earlier stage provides it. "
                f"Current order: {names}"
            )
        available.update(meta["provides"])
    if "save" not in names:
        logger.warning("No 'save' stage configured: nothing will be written to disk")


# ---- data preparation helper --------------------------------------------
def prepare_xy(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    d = cfg.data
    if d.target not in df.columns:
        raise DataError(f"Target column '{d.target}' not found. Columns: {list(df.columns)}")
    unknown_drop = [c for c in d.drop_columns if c not in df.columns]
    if unknown_drop:
        raise DataError(f"drop_columns not found in data: {unknown_drop}")

    excluded = set(d.drop_columns) | {d.target}
    features = d.features or [c for c in df.columns if c not in excluded]
    if d.target in features:
        raise DataError("The target column cannot also be a feature")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise DataError(f"Feature columns not found in data: {missing}")
    if not features:
        raise DataError("No feature columns left after exclusions")

    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric and not cfg.auto_preprocess.enabled:
        raise DataError(
            f"Non-numeric feature columns {non_numeric}. Enable 'auto_preprocess' (it one-hot "
            "encodes them) or list numeric columns under data.features / data.drop_columns."
        )

    n_before = len(df)
    df = df.dropna(subset=[d.target])
    if len(df) < n_before:
        logger.warning("Dropped %d rows with a missing target", n_before - len(df))
    if df.empty:
        raise DataError("No rows left after dropping rows with a missing target")
    if cfg.task == "regression" and not pd.api.types.is_numeric_dtype(df[d.target]):
        raise DataError(f"Target '{d.target}' must be numeric for regression")

    X, y = df[features], df[d.target]
    has_imputer = any(PREPROCESSORS.get(s.name).meta.get("handles_missing") for s in cfg.preprocessing)
    if X.isna().any().any() and not cfg.auto_preprocess.enabled and not has_imputer:
        raise DataError(
            "Features contain missing values. Enable 'auto_preprocess' or add an imputer step "
            "('simple_imputer' or 'knn_imputer') to 'preprocessing'."
        )
    return X, y, features


# ---- stages --------------------------------------------------------------
@stage("load", provides=("X", "y", "features"))
def load(ctx: RunContext) -> None:
    ctx.X, ctx.y, ctx.features = prepare_xy(ctx.cfg, ctx.loader.load())
    logger.info("Loaded %d rows, %d features", len(ctx.X), len(ctx.features))
    if ctx.cfg.auto_preprocess.enabled:
        for col in ctx.X.select_dtypes(exclude="number").columns:
            if ctx.X[col].nunique() > 50:
                logger.warning(
                    "Column '%s' has %d distinct values and will create that many one-hot "
                    "columns; consider data.drop_columns", col, ctx.X[col].nunique(),
                )


@stage("split", requires=("X", "y"), provides=("X_train", "X_test", "y_train", "y_test"))
def split(ctx: RunContext) -> None:
    splitter = SPLITTERS.get(ctx.cfg.split.strategy).obj
    try:
        ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test = splitter(ctx.X, ctx.y, ctx.cfg)
    except ValueError as exc:
        raise DataError(f"Could not split data: {exc}") from exc
    logger.info("Split: %d train / %d test rows", len(ctx.X_train), len(ctx.X_test))


@stage("select", requires=("X_train", "y_train"), provides=("selected",))
def select(ctx: RunContext) -> None:
    """Pick among candidate models by cross-validation on TRAIN only (test stays untouched)."""
    cfg = ctx.cfg
    labels = candidate_labels(cfg)
    best = labels[0]
    if len(cfg.models) > 1:
        entry = METRICS.get(cfg.selection.metric)
        greater = entry.meta["greater_is_better"]
        scorer = make_scorer(entry.obj, greater_is_better=greater)
        splitter_cls = StratifiedKFold if cfg.task == "classification" else KFold
        cv = splitter_cls(n_splits=cfg.selection.cv_folds, shuffle=True, random_state=cfg.seed)
        sign = 1 if greater else -1  # make_scorer negates "lower is better" metrics
        for label, step in zip(labels, cfg.models):
            scores = cross_val_score(
                build_pipeline(cfg, step), ctx.X_train, ctx.y_train, cv=cv, scoring=scorer
            )
            ctx.cv_results[label] = {"mean": float(scores.mean() * sign), "std": float(scores.std())}
            logger.info("CV %s %s: %.4f", label, cfg.selection.metric, ctx.cv_results[label]["mean"])
        pick = max if greater else min
        best = pick(ctx.cv_results, key=lambda k: ctx.cv_results[k]["mean"])
    ctx.selected = best
    ctx.selected_step = cfg.models[labels.index(best)]
    logger.info("Selected model: %s", best)


@stage("train", requires=("X_train", "y_train", "selected"), provides=("model",))
def train(ctx: RunContext) -> None:
    model = build_pipeline(ctx.cfg, ctx.selected_step)
    try:
        model.fit(ctx.X_train, ctx.y_train)
    except ValueError as exc:
        raise DataError(f"Training failed: {exc}") from exc
    ctx.model = model
    ctx.preprocessing_report = describe_auto_preprocessing(model, ctx.X_train)
    ctx.categorical_features = (ctx.preprocessing_report or {}).get("categorical_columns", [])
    if ctx.preprocessing_report and ctx.preprocessing_report.get("outliers_replaced_in_train"):
        logger.info("Outliers replaced (train): %s", ctx.preprocessing_report["outliers_replaced_in_train"])


@stage("evaluate", requires=("model", "X_test", "y_test"), provides=("metrics",))
def evaluate(ctx: RunContext) -> None:
    y_pred = ctx.model.predict(ctx.X_test)
    ctx.metrics = {
        m: float(METRICS.get(m).obj(ctx.y_test, y_pred)) for m in ctx.cfg.evaluation.metrics
    }
    ctx.tracker.log_metrics(ctx.metrics)


@stage("predict", requires=("model", "X_test", "y_test", "features"),
       provides=("test_predictions",))
def predict(ctx: RunContext) -> None:
    """Inference output: predictions for the test split, plus an optional extra file."""
    preds = predict_frame(ctx.model, ctx.features, ctx.X_test, ctx.categorical_features)
    preds.insert(0, "y_true", ctx.y_test)
    ctx.test_predictions = preds
    path = ctx.cfg.inference.input_path
    if path:
        try:
            extra = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise DataError(f"inference.input_path not found: {path}") from exc
        ctx.inference_predictions = predict_frame(
            ctx.model, ctx.features, extra, ctx.categorical_features
        )


def _build_meta(ctx: RunContext) -> Dict[str, Any]:
    cfg = ctx.cfg
    numeric_train = ctx.X_train.select_dtypes("number") if ctx.X_train is not None else None
    stats = (
        numeric_train.agg(["mean", "std", "min", "max"])
        if numeric_train is not None and not numeric_train.empty else None
    )
    return {
        "run_id": ctx.store.run_id,
        "task": cfg.task,
        "target": cfg.data.target,
        "features": ctx.features,
        "classes": [str(c) for c in getattr(ctx.model, "classes_", [])],
        "selected_model": ctx.selected,
        "metrics": ctx.metrics,
        "cv_results": ctx.cv_results,
        "stages": cfg.stages,
        "categorical_features": ctx.categorical_features,
        "preprocessing": ctx.preprocessing_report,
        "rows": {
            "total": len(ctx.X),
            "train": len(ctx.X_train) if ctx.X_train is not None else None,
            "test": len(ctx.X_test) if ctx.X_test is not None else None,
        },
        # Training-data profile: the baseline you would compare live data against for drift.
        "feature_stats": None if stats is None else {
            c: {k: float(np.nan_to_num(stats.loc[k, c])) for k in stats.index} for c in stats.columns
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - ctx.started, 3),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


@stage("save", requires=("model",))
def save(ctx: RunContext) -> None:
    store = ctx.store
    store.save_model(ctx.model)
    if ctx.metrics is not None:
        store.save_json(ctx.metrics, "metrics.json")
    if ctx.test_predictions is not None:
        store.save_frame(ctx.test_predictions, "predictions.csv")
    if ctx.inference_predictions is not None:
        store.save_frame(ctx.inference_predictions, "inference_predictions.csv")
    store.save_text(yaml.safe_dump(ctx.cfg.model_dump(), sort_keys=False), "config.resolved.yaml")
    store.save_json(_build_meta(ctx), "run_meta.json")
