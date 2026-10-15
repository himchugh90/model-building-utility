"""Ports: the interfaces the core depends on.

The core never imports a concrete loader, store or tracker. Swapping CSV for
a feature store, local disk for S3, or logging for MLflow means writing a new
adapter, not editing the runner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol

import pandas as pd


class DataLoader(Protocol):
    def load(self) -> pd.DataFrame: ...


class ArtifactStore(Protocol):
    run_id: str

    @property
    def location(self) -> Path: ...

    def save_model(self, model: Any, name: str = "model.joblib") -> None: ...
    def save_json(self, obj: Any, name: str) -> None: ...
    def save_text(self, text: str, name: str) -> None: ...
    def save_frame(self, df: pd.DataFrame, name: str) -> None: ...
    def commit(self) -> None: ...
    def abort(self) -> None: ...


class Tracker(Protocol):
    def log_params(self, params: Dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: Dict[str, float]) -> None: ...
