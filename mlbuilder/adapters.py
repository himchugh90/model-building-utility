"""Default adapters for the ports: local CSV / sklearn data, local disk, logging."""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn import datasets

from .config import DataConfig
from .errors import ConfigError, DataError
from .registry import LOADERS

logger = logging.getLogger("mlbuilder")


# ---- data loaders --------------------------------------------------------
class CsvLoader:
    def __init__(self, path: str, read_options: Dict[str, Any] = None) -> None:
        self.path = Path(path)
        self.read_options = read_options or {}  # e.g. {sep: ";", encoding: "latin-1"}

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataError(f"Data file not found: {self.path.resolve()}")
        try:
            return pd.read_csv(self.path, **self.read_options)
        except Exception as exc:  # pandas raises many error types
            raise DataError(f"Could not read CSV {self.path}: {exc}") from exc


_SKLEARN_DATASETS = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
    "diabetes": datasets.load_diabetes,
}


class SklearnDatasetLoader:
    """Built-in toy datasets. The target column is always called 'target'."""

    def __init__(self, name: str) -> None:
        if name not in _SKLEARN_DATASETS:
            raise ConfigError(
                f"Unknown sklearn dataset '{name}'. Available: {sorted(_SKLEARN_DATASETS)}"
            )
        self.name = name

    def load(self) -> pd.DataFrame:
        bunch = _SKLEARN_DATASETS[self.name](as_frame=True)
        df = bunch.frame.copy()
        df.columns = [re.sub(r"\W+", "_", c).strip("_") for c in df.columns]
        names = getattr(bunch, "target_names", None)
        if names is not None and len(names) > 1:
            df["target"] = df["target"].map(dict(enumerate(names)))
        return df


def _csv_factory(cfg: DataConfig):
    if not cfg.path:
        raise ConfigError("data.path is required when data.source is 'csv'")
    return CsvLoader(cfg.path, cfg.read_options)


def _sklearn_factory(cfg: DataConfig):
    if not cfg.dataset:
        raise ConfigError("data.dataset is required when data.source is 'sklearn'")
    return SklearnDatasetLoader(cfg.dataset)


# To support a new source (parquet, database, ...): write a loader class and
# register a factory here or in a plugin. Neither the schema nor the runner changes.
LOADERS.add("csv", _csv_factory)
LOADERS.add("sklearn", _sklearn_factory)


def build_loader(cfg: DataConfig):
    return LOADERS.get(cfg.source).obj(cfg)


# ---- artifact store ------------------------------------------------------
class LocalArtifactStore:
    """Writes into a hidden temp directory and renames it on commit, so a
    failed run never leaves a half-written run directory behind."""

    def __init__(self, base_dir: str, run_id: str) -> None:
        self.run_id = run_id
        base = Path(base_dir)
        self._final = base / run_id
        self._tmp = base / f".{run_id}.partial"
        self._tmp.mkdir(parents=True, exist_ok=False)

    @property
    def location(self) -> Path:
        return self._final

    def save_model(self, model: Any, name: str = "model.joblib") -> None:
        joblib.dump(model, self._tmp / name)

    def save_json(self, obj: Any, name: str) -> None:
        (self._tmp / name).write_text(json.dumps(obj, indent=2, default=str))

    def save_text(self, text: str, name: str) -> None:
        (self._tmp / name).write_text(text)

    def save_frame(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self._tmp / name, index=True)

    def commit(self) -> None:
        self._tmp.rename(self._final)

    def abort(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


# ---- tracker -------------------------------------------------------------
class LoggingTracker:
    """Simplest tracker. An MLflowTracker would implement the same two methods."""

    def log_params(self, params: Dict[str, Any]) -> None:
        logger.info("params: %s", params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        logger.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
