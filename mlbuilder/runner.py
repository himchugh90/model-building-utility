"""Entry point for one run: validate, build the context, execute the configured stages."""
from __future__ import annotations

import importlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import components  # noqa: F401  (registers built-ins)
from .adapters import LocalArtifactStore, LoggingTracker, build_loader
from .config import Config
from .pipeline import validate_config
from .ports import ArtifactStore, DataLoader, Tracker
from .registry import STAGES
from .stages import RunContext, validate_stage_order

logger = logging.getLogger("mlbuilder")


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    selected_model: Optional[str]
    metrics: Dict[str, float]
    cv_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model: Any = None


def load_plugins(modules: List[str]) -> None:
    """Plugins are explicit module imports (not arbitrary dotted paths from config)."""
    for name in modules:
        importlib.import_module(name)


def make_run_id(run_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{run_name}-{stamp}-{uuid.uuid4().hex[:6]}"


def run(
    cfg: Config,
    loader: Optional[DataLoader] = None,
    store: Optional[ArtifactStore] = None,
    tracker: Optional[Tracker] = None,
) -> RunResult:
    load_plugins(cfg.plugins)
    validate_config(cfg)  # fail fast, before touching data
    validate_stage_order(cfg.stages)

    loader = loader or build_loader(cfg.data)
    tracker = tracker or LoggingTracker()
    store = store or LocalArtifactStore(cfg.output.dir, make_run_id(cfg.run_name))
    ctx = RunContext(cfg=cfg, loader=loader, store=store, tracker=tracker)

    tracker.log_params(
        {"run_id": store.run_id, "task": cfg.task, "seed": cfg.seed,
         "models": [m.name for m in cfg.models], "stages": cfg.stages}
    )
    try:
        for name in cfg.stages:
            logger.info("stage: %s", name)
            STAGES.get(name).obj(ctx)
        store.commit()
    except Exception:
        store.abort()
        raise
    logger.info("Run complete: %s", store.location)
    return RunResult(
        run_id=store.run_id,
        run_dir=store.location,
        selected_model=ctx.selected,
        metrics=ctx.metrics or {},
        cv_results=ctx.cv_results,
        model=ctx.model,
    )
