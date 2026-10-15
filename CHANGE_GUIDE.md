# Change guide: "if I'm asked to change X, where do I go?"

Every row is one place to edit. After any change run `python -m pytest -q` and
`python -m mlbuilder list` (shows everything registered).

| Request | What to do | Where |
|---|---|---|
| Add a model | `MODELS.add("name", {"classification": Cls, "regression": Reg})` | `components.py` (or a plugin) |
| Add a preprocessing / featurisation step (PCA, polynomial...) | `PREPROCESSORS.add("pca", PCA)`, then list it under `preprocessing:` | `components.py` |
| Add a metric | `METRICS.add("name", fn, tasks=("classification",), greater_is_better=True)` | `components.py` |
| Change test size / seed / strategy | edit YAML or `--set split.test_size=0.3` | config |
| New split strategy (time-based, group...) | `SPLITTERS.add("name", fn(X, y, cfg))` returning X_tr, X_te, y_tr, y_te | `components.py` |
| Read another format (parquet, JSON, DB) | write a loader class with `load()`; `LOADERS.add("parquet", factory)`; set `data.source` | `adapters.py` |
| CSV oddities (separator, encoding) | `data.read_options: {sep: ";"}` | config |
| Skip / reorder steps | `stages: [load, split, train, save]` | config |
| Add a new step (feature importance, plots, tuning) | write `@stage("name", requires=(...), provides=(...))` function, list it in `stages:` | `stages.py` (or plugin) |
| Hyperparameter tuning | new stage using `GridSearchCV` on `ctx.X_train`; set `ctx.selected_step` / `ctx.model`; use it instead of `select` | `stages.py` |
| Turn automatic preprocessing off, or tune it | `auto_preprocess:` block, or `--set auto_preprocess.outliers=false` / `.enabled=false` / `.outlier_factor=3` | config |
| Different imputer / encoder / outlier method in the automatic step | edit `build_auto_preprocessor` (the one place it is assembled) | `preprocess.py` |
| Remove outlier ROWS (not just replace values) | new stage between `split` and `select` that filters `ctx.X_train` / `ctx.y_train` (train only!) | `stages.py` |
| Per-column preprocessing beyond numeric vs non-numeric | extend the `ColumnTransformer` in `build_auto_preprocessor`, or wrap steps in `build_pipeline` | `preprocess.py` / `pipeline.py` |
| Save to S3 / registry | class with the 6 methods of `ArtifactStore`; call `run(cfg, store=MyStore())` | `adapters.py` |
| Track in MLflow | class with `log_params` / `log_metrics`; call `run(cfg, tracker=MyTracker())` | `adapters.py` |
| Different default metrics | `DEFAULT_METRICS` / `DEFAULT_SELECTION_METRIC` | `config.py` |
| Try a value without editing files | `--set models.0.params.n_estimators=300` | CLI |
| Add a plugin without touching the package | put registrations in `my_plugin.py`, `plugins: [my_plugin]` | config |

Rule of thumb: registry entry for "a thing", stage for "a step", adapter for "infrastructure".
