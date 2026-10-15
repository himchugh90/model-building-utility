"""Turns a validated Config into sklearn Pipelines.

Preprocessing lives *inside* the Pipeline, so it is fit on training data only
and is persisted with the model. That prevents leakage and train/serve skew.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

from sklearn.pipeline import Pipeline

from .config import Config, StepConfig
from .errors import ConfigError
from .preprocess import build_auto_preprocessor
from .registry import LOADERS, METRICS, MODELS, PREPROCESSORS, SPLITTERS


def _instantiate(kind: str, name: str, cls: Any, params: Dict[str, Any], seed: int):
    params = dict(params)
    try:
        accepted = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        accepted = {}
    if "random_state" in accepted and "random_state" not in params:
        params["random_state"] = seed  # reproducibility by default
    try:
        return cls(**params)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid params for {kind} '{name}': {exc}") from exc


def build_pipeline(cfg: Config, model_step: StepConfig) -> Pipeline:
    steps = []
    if cfg.auto_preprocess.enabled:
        steps.append(("auto", build_auto_preprocessor(cfg.auto_preprocess)))
    for i, step in enumerate(cfg.preprocessing):
        cls = PREPROCESSORS.get(step.name).obj
        steps.append(
            (f"{i}_{step.name}", _instantiate("preprocessor", step.name, cls, step.params, cfg.seed))
        )
    model_cls = MODELS.get(model_step.name).obj[cfg.task]
    steps.append(
        ("model", _instantiate("model", model_step.name, model_cls, model_step.params, cfg.seed))
    )
    return Pipeline(steps)


def validate_config(cfg: Config) -> None:
    """Semantic checks that the schema alone cannot express. Runs before any data is read."""
    LOADERS.get(cfg.data.source)
    SPLITTERS.get(cfg.split.strategy)
    for step in cfg.preprocessing:
        PREPROCESSORS.get(step.name)

    for step in cfg.models:
        supported = MODELS.get(step.name).obj
        if cfg.task not in supported:
            raise ConfigError(
                f"Model '{step.name}' does not support task '{cfg.task}' "
                f"(supports: {sorted(supported)})"
            )
        build_pipeline(cfg, step)  # surfaces bad params now, not mid-run

    for metric in list(cfg.evaluation.metrics) + [cfg.selection.metric]:
        entry = METRICS.get(metric)
        if cfg.task not in entry.meta["tasks"]:
            raise ConfigError(
                f"Metric '{metric}' is not valid for task '{cfg.task}' "
                f"(valid for: {list(entry.meta['tasks'])})"
            )


def candidate_labels(cfg: Config) -> list:
    """Unique labels, so two random_forest candidates with different params stay distinct."""
    names = [m.name for m in cfg.models]
    return [n if names.count(n) == 1 else f"{n}_{i}" for i, n in enumerate(names)]
