"""Generic name -> component registries.

Adding a new model / preprocessor / metric means registering one object.
No other module changes (open/closed principle).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .errors import ConfigError


@dataclass(frozen=True)
class Entry:
    obj: Any
    meta: Dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Entry] = {}

    def add(self, name: str, obj: Any, **meta: Any) -> Any:
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = Entry(obj, meta)
        return obj

    def register(self, name: str, **meta: Any) -> Callable[[Any], Any]:
        """Decorator form of `add`."""

        def decorator(obj: Any) -> Any:
            return self.add(name, obj, **meta)

        return decorator

    def get(self, name: str) -> Entry:
        if name not in self._items:
            hint = difflib.get_close_matches(name, self._items, n=1)
            msg = f"Unknown {self.kind} '{name}'."
            if hint:
                msg += f" Did you mean '{hint[0]}'?"
            msg += f" Available: {self.names()}"
            raise ConfigError(msg)
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)


PREPROCESSORS = Registry("preprocessor")
MODELS = Registry("model")  # entry.obj is a dict: task -> estimator class
METRICS = Registry("metric")  # entry.meta: tasks, greater_is_better
LOADERS = Registry("data source")  # entry.obj: factory(DataConfig) -> DataLoader
SPLITTERS = Registry("split strategy")  # entry.obj: fn(X, y, cfg) -> X_tr, X_te, y_tr, y_te
STAGES = Registry("stage")  # entry.obj: fn(ctx); entry.meta: requires, provides
