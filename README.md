# mlbuilder: config-driven model building utility

Build a baseline ML model from a YAML file. No code changes needed for new datasets,
models, or hyperparameters.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_sample_data.py                      # optional: data/iris.csv from sklearn
python scripts/make_sample_data.py diabetes             # optional: data/diabetes.csv
```

## Use
```bash
python -m mlbuilder train --config configs/iris.yaml
python -m mlbuilder train --config configs/wine.yaml          # multiclass, built-in dataset
python -m mlbuilder train --config configs/diabetes.yaml      # regression
python -m mlbuilder train --config configs/iris_plugin.yaml   # model added via plugin

python -m mlbuilder predict --run-dir runs/<run_id> --input new_data.csv --output preds.csv
# new_data.csv needs the same feature column NAMES the model was trained on (order and extra columns are fine)
python -m mlbuilder train --config configs/iris.yaml --set split.test_size=0.3 --set models.0.params.max_iter=500
python -m mlbuilder train --config configs/iris.yaml --set "stages=[load, split, select, train, save]"
python scripts/make_messy_data.py && python -m mlbuilder train --config configs/messy_iris.yaml   # NaNs, outliers, text column
python -m mlbuilder list            # everything registered: stages, sources, splits, models, metrics...
python -m pytest -q    # for running all tests
python -m pytest -q -k test_regression_end_to_end  #for running test by name 
```
Each run writes `runs/<run_id>/`: `model.joblib`, `metrics.json`, `predictions.csv`
(test split), `config.resolved.yaml`, `run_meta.json` (features, versions, CV scores, row counts).

## Layout
```
mlbuilder/
  config.py       typed schema (pydantic, extra="forbid"), `extends:` inheritance
  registry.py     name -> component registries (+ did-you-mean errors)
  components.py   built-in preprocessors, models, metrics
  ports.py        interfaces: DataLoader, ArtifactStore, Tracker
  adapters.py     default adapters: CSV/sklearn data, local disk, logging
  preprocess.py   automatic preprocessing: outliers, KNN imputation, scaling, one-hot encoding
  pipeline.py     config -> sklearn Pipeline, semantic validation
  stages.py       the steps (load, split, select, train, evaluate, predict, save) + RunContext
  inference.py    predict_frame / load_run, shared by training and the predict command
  runner.py       validate, build context, run the configured stages
  cli.py          train / predict commands
```

## Design notes
* Preprocessing is inside the sklearn Pipeline: fit on train only, saved with the model.
* Split happens before anything is fit. With several candidate models, selection uses
  cross-validation on the training split; the test set is evaluated once.
* Validation runs before data is read: unknown names, task/metric mismatches, bad params.
* Runs are atomic: written to a temp dir and renamed on success.
* Reproducible: global seed injected into every estimator that accepts `random_state`.
* Automatic preprocessing (on by default, one switch to disable): numeric columns get IQR outlier
  replacement (mean of non-outliers), scaling and KNN imputation; non-numeric columns get one-hot
  encoding. Fit on train only, features only (never the target), identical for classification and
  regression. What it did is recorded under `preprocessing` in `run_meta.json`.
* The steps to run are config (`stages:`); order is validated before any data is read.
* Extend by registering a component (see `my_plugin.py`); the core is not modified.
* Swap infrastructure by writing an adapter for a port (e.g. MLflow `Tracker`, S3 `ArtifactStore`).

See `CHANGE_GUIDE.md` for where each kind of change goes.

## Known limitations
Tabular, in-memory data only (KNN imputation is O(n^2): for large data use `simple_imputer`); numeric and low-cardinality categorical features (high-cardinality text columns would explode one-hot, so a warning is logged); outlier replacement can hurt when outliers are the signal (fraud, anomalies), so disable it there; one split strategy (random,
stratified for classification); no time-series splits; no hyperparameter search;
joblib artifacts are only safe to load from trusted sources and are sklearn-version sensitive.
