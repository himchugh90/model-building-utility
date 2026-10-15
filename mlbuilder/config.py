"""Config schema, validation and loading.

The schema is the contract between users and the utility. `extra="forbid"`
turns typos like `test_sise` into immediate, readable errors instead of
silently ignored settings.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ConfigError

DEFAULT_METRICS = {
    "classification": ["accuracy", "f1_macro"],
    "regression": ["r2", "rmse", "mae"],
}
DEFAULT_SELECTION_METRIC = {"classification": "accuracy", "regression": "r2"}
DEFAULT_STAGES = ["load", "split", "select", "train", "evaluate", "predict", "save"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StepConfig(_Strict):
    """A named component (preprocessor or model) plus its constructor params."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DataConfig(_Strict):
    source: str = "csv"  # any name in the LOADERS registry (csv, sklearn, or a plugin's)
    path: Optional[str] = None  # csv: file path
    dataset: Optional[str] = None  # sklearn: iris | wine | breast_cancer | diabetes
    read_options: Dict[str, Any] = Field(default_factory=dict)  # passed to pandas.read_csv
    target: str
    features: Optional[List[str]] = None  # None -> every column except target / drop_columns
    drop_columns: List[str] = Field(default_factory=list)  # e.g. an "Id" column


class SplitConfig(_Strict):
    strategy: str = "random"  # any name in the SPLITTERS registry (random, ordered, ...)
    test_size: float = Field(0.2, gt=0, lt=1)
    stratify: bool = True  # only applied for classification


class SelectionConfig(_Strict):
    """Used only when more than one model is listed."""

    metric: Optional[str] = None
    cv_folds: int = Field(5, ge=2)


class EvaluationConfig(_Strict):
    metrics: List[str] = Field(default_factory=list)


class InferenceConfig(_Strict):
    input_path: Optional[str] = None  # optional extra file to score after training


class OutputConfig(_Strict):
    dir: str = "runs"


class AutoPreprocessConfig(_Strict):
    """Automatic, train-only preprocessing. Set `enabled: false` to turn it all off."""

    enabled: bool = True
    outliers: bool = True  # IQR rule; outliers -> mean of the non-outlier training values
    outlier_factor: float = Field(1.5, gt=0)  # IQR multiplier (3.0 = only extreme outliers)
    replace_with: Literal["mean", "median"] = "mean"
    knn_neighbors: int = Field(5, ge=1)  # KNN imputation of missing numeric values


class Config(_Strict):
    run_name: str
    task: Literal["classification", "regression"]
    seed: int = 42
    plugins: List[str] = Field(default_factory=list)  # modules that register extra components
    stages: List[str] = Field(default_factory=lambda: list(DEFAULT_STAGES))  # steps to run, in order
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    auto_preprocess: AutoPreprocessConfig = Field(default_factory=AutoPreprocessConfig)
    preprocessing: List[StepConfig] = Field(default_factory=list)  # extra steps, run AFTER auto
    models: List[StepConfig] = Field(min_length=1)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _fill_task_defaults(self) -> "Config":
        if not self.evaluation.metrics:
            self.evaluation.metrics = list(DEFAULT_METRICS[self.task])
        if not self.selection.metric:
            self.selection.metric = DEFAULT_SELECTION_METRIC[self.task]
        return self


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Dicts merge recursively; every other type (including lists) is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_with_extends(path: Path, seen: Tuple[Path, ...] = ()) -> Dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular 'extends' detected at {path}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        base = _read_with_extends(path.parent / parent, seen + (path,))
        raw = _deep_merge(base, raw)
    return raw


def parse_config(raw: Dict[str, Any]) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc


def _index(key: str, lst: list, original: str) -> int:
    try:
        i = int(key)
    except ValueError:
        raise ConfigError(f"Override '{original}': '{key}' must be a list index") from None
    if not -len(lst) <= i < len(lst):
        raise ConfigError(f"Override '{original}': index {i} is out of range")
    return i


def _step(node: Any, key: str, original: str) -> Any:
    if isinstance(node, list):
        return node[_index(key, node, original)]
    if isinstance(node, dict):
        return node.setdefault(key, {})
    raise ConfigError(f"Override '{original}': cannot descend into '{key}'")


def apply_overrides(raw: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply `dotted.path=value` overrides, e.g. `models.0.params.max_iter=500`.

    Values are parsed as YAML (so 0.3, true, [a, b] work). Numeric path parts index lists.
    """
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like dotted.path=value")
        path, text = item.split("=", 1)
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError:
            value = text
        keys = path.strip().split(".")
        node = raw
        for key in keys[:-1]:
            node = _step(node, key, item)
        if isinstance(node, list):
            node[_index(keys[-1], node, item)] = value
        elif isinstance(node, dict):
            node[keys[-1]] = value
        else:
            raise ConfigError(f"Override '{item}': cannot set '{keys[-1]}' here")
    return raw


def load_config(path: "str | Path", overrides: Optional[List[str]] = None) -> Config:
    raw = _read_with_extends(Path(path))
    if overrides:
        apply_overrides(raw, overrides)
    return parse_config(raw)
