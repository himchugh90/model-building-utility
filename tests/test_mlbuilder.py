import pandas as pd
import pytest

from mlbuilder import (
    ConfigError, DataError, load_config, load_run, parse_config, predict_frame, run,
)


def make_cfg(tmp_path, **overrides):
    raw = {
        "run_name": "t",
        "task": "classification",
        "data": {"source": "sklearn", "dataset": "iris", "target": "target"},
        "preprocessing": [{"name": "standard_scaler"}],
        "models": [{"name": "logistic_regression", "params": {"max_iter": 300}}],
        "output": {"dir": str(tmp_path)},
    }
    raw.update(overrides)
    return parse_config(raw)


@pytest.mark.parametrize("dataset", ["iris", "wine", "breast_cancer"])
def test_classification_end_to_end(tmp_path, dataset):
    cfg = make_cfg(tmp_path, data={"source": "sklearn", "dataset": dataset, "target": "target"})
    result = run(cfg)
    assert result.metrics["accuracy"] > 0.85
    for name in ["model.joblib", "metrics.json", "predictions.csv",
                 "config.resolved.yaml", "run_meta.json"]:
        assert (result.run_dir / name).exists(), name


def test_regression_end_to_end(tmp_path):
    cfg = make_cfg(
        tmp_path,
        task="regression",
        data={"source": "sklearn", "dataset": "diabetes", "target": "target"},
        models=[{"name": "ridge"}],
        split={"stratify": False},
    )
    assert run(cfg).metrics["r2"] > 0.3


def test_model_selection_uses_cv_and_picks_a_candidate(tmp_path):
    cfg = make_cfg(
        tmp_path,
        models=[{"name": "logistic_regression", "params": {"max_iter": 300}},
                {"name": "random_forest", "params": {"n_estimators": 20}}],
    )
    result = run(cfg)
    assert set(result.cv_results) == {"logistic_regression", "random_forest"}
    assert result.selected_model in result.cv_results


def test_same_seed_is_reproducible(tmp_path):
    cfg = make_cfg(tmp_path, models=[{"name": "random_forest", "params": {"n_estimators": 20}}])
    assert run(cfg).metrics == run(cfg).metrics


def test_unknown_model_gets_suggestion(tmp_path):
    cfg = make_cfg(tmp_path, models=[{"name": "random_forst"}])
    with pytest.raises(ConfigError, match="Did you mean 'random_forest'"):
        run(cfg)


def test_typo_in_config_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="test_sise"):
        make_cfg(tmp_path, split={"test_sise": 0.3})


def test_model_task_mismatch_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path, models=[{"name": "linear_regression"}])
    with pytest.raises(ConfigError, match="does not support task"):
        run(cfg)


def test_metric_task_mismatch_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path, evaluation={"metrics": ["rmse"]})
    with pytest.raises(ConfigError, match="not valid for task"):
        run(cfg)


def test_bad_model_params_fail_before_data_is_read(tmp_path):
    cfg = make_cfg(tmp_path, models=[{"name": "random_forest", "params": {"n_trees": 5}}])
    with pytest.raises(ConfigError, match="Invalid params"):
        run(cfg)


def test_failed_run_leaves_no_partial_directory(tmp_path):
    cfg = make_cfg(tmp_path, data={"source": "csv", "path": "does_not_exist.csv", "target": "y"})
    with pytest.raises(DataError):
        run(cfg)
    assert list(tmp_path.iterdir()) == []


def test_missing_values_need_auto_preprocess_or_an_imputer_step(tmp_path):
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0] * 10, "b": range(40), "y": [0, 1] * 20})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    data = {"source": "csv", "path": str(csv), "target": "y"}
    off = {"enabled": False}
    with pytest.raises(DataError, match="imputer"):
        run(make_cfg(tmp_path, data=data, preprocessing=[], auto_preprocess=off))
    for imputer in ("simple_imputer", "knn_imputer"):  # any step flagged handles_missing works
        cfg = make_cfg(tmp_path, data=data, preprocessing=[{"name": imputer}], auto_preprocess=off)
        assert run(cfg).metrics
    assert run(make_cfg(tmp_path, data=data, preprocessing=[])).metrics  # auto handles it


def test_predict_roundtrip_and_missing_columns(tmp_path):
    result = run(make_cfg(tmp_path))
    model, meta = load_run(result.run_dir)
    new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=meta["features"])
    out = predict_frame(model, meta["features"], new)
    assert out.loc[0, "prediction"] == "setosa"
    with pytest.raises(DataError, match="missing required columns"):
        predict_frame(model, meta["features"], new.drop(columns=meta["features"][0]))


def test_extends_merges_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 7\nsplit: {test_size: 0.3}\n")
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nrun_name: c\ntask: classification\n"
        "data: {source: sklearn, dataset: iris, target: target}\n"
        "models: [{name: knn}]\nsplit: {stratify: false}\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.seed == 7 and cfg.split.test_size == 0.3 and cfg.split.stratify is False


def test_scoring_ignores_extra_columns_and_column_order(tmp_path):
    result = run(make_cfg(tmp_path))
    model, meta = load_run(result.run_dir)
    row = {"species": "x", "petal_width_cm": 0.2, "petal_length_cm": 1.4,
           "sepal_width_cm": 3.5, "sepal_length_cm": 5.1}
    out = predict_frame(model, meta["features"], pd.DataFrame([row]))
    assert out.loc[0, "prediction"] == "setosa"


def test_scoring_wrong_dataset_and_bad_values_raise_data_error(tmp_path):
    result = run(make_cfg(tmp_path))
    model, meta = load_run(result.run_dir)
    feats = meta["features"]
    with pytest.raises(DataError, match="missing required columns"):
        predict_frame(model, feats, pd.DataFrame({"age": [1.0], "bmi": [2.0]}))
    with pytest.raises(DataError, match="Non-numeric"):
        predict_frame(model, feats, pd.DataFrame([["abc", 1, 1, 1]], columns=feats))
    nan_row = pd.DataFrame([[float("nan"), 3.0, 1.4, 0.2]], columns=feats)
    assert len(predict_frame(model, feats, nan_row)) == 1  # auto_preprocess imputes it
    plain = run(make_cfg(tmp_path, auto_preprocess={"enabled": False}))
    plain_model, _ = load_run(plain.run_dir)
    with pytest.raises(DataError, match="Could not score input"):
        predict_frame(plain_model, feats, nan_row)


# ---- parametrization features -------------------------------------------
import json

from mlbuilder import LOADERS, SPLITTERS, STAGES, stage
from mlbuilder.config import apply_overrides
from mlbuilder.registry import Entry


def test_stages_are_configurable_skip_evaluate_and_predict(tmp_path):
    cfg = make_cfg(tmp_path, stages=["load", "split", "select", "train", "save"])
    result = run(cfg)
    assert result.metrics == {}
    assert not (result.run_dir / "metrics.json").exists()
    assert not (result.run_dir / "predictions.csv").exists()
    assert (result.run_dir / "model.joblib").exists()


def test_misordered_or_incomplete_stages_fail_before_data_is_read(tmp_path):
    with pytest.raises(ConfigError, match="Stage 'train' needs"):
        run(make_cfg(tmp_path, stages=["train", "load"]))
    with pytest.raises(ConfigError, match="Unknown stage 'evalute'"):
        run(make_cfg(tmp_path, stages=["load", "evalute"]))
    assert list(tmp_path.iterdir()) == []


def test_custom_stage_can_be_added_without_touching_the_core(tmp_path, monkeypatch):
    seen = {}

    def my_stage(ctx):
        seen["rows"] = len(ctx.X_train)

    monkeypatch.setitem(
        STAGES._items, "my_stage",
        Entry(my_stage, {"requires": ("X_train",), "provides": ()}),
    )
    run(make_cfg(tmp_path, stages=["load", "split", "my_stage", "select", "train", "save"]))
    assert seen["rows"] == 120  # 80% of iris


def test_ordered_split_keeps_row_order(tmp_path):
    cfg = make_cfg(tmp_path, split={"strategy": "ordered", "test_size": 0.2})
    result = run(cfg)
    preds = pd.read_csv(result.run_dir / "predictions.csv", index_col=0)
    assert list(preds.index) == list(range(120, 150))


def test_unknown_split_strategy_gets_suggestion(tmp_path):
    with pytest.raises(ConfigError, match="Did you mean 'random'"):
        run(make_cfg(tmp_path, split={"strategy": "randon"}))


def test_csv_read_options_and_custom_loader(tmp_path, monkeypatch):
    df = pd.DataFrame({"a": range(40), "b": range(40, 80), "y": [0, 1] * 20})
    csv = tmp_path / "semi.csv"
    df.to_csv(csv, index=False, sep=";")
    cfg = make_cfg(tmp_path, data={"source": "csv", "path": str(csv), "target": "y",
                                   "read_options": {"sep": ";"}})
    assert run(cfg).metrics

    class InMemory:
        def load(self):
            return df

    monkeypatch.setitem(LOADERS._items, "memory", Entry(lambda cfg: InMemory(), {}))
    cfg = make_cfg(tmp_path, data={"source": "memory", "target": "y"})
    assert run(cfg).metrics


def test_overrides_change_nested_values_and_list_items():
    raw = {"split": {"test_size": 0.2}, "models": [{"name": "knn", "params": {}}]}
    apply_overrides(raw, ["split.test_size=0.3", "models.0.params.n_neighbors=7",
                          "stages=[load, split]", "seed=1"])
    assert raw["split"]["test_size"] == 0.3
    assert raw["models"][0]["params"]["n_neighbors"] == 7
    assert raw["stages"] == ["load", "split"] and raw["seed"] == 1
    with pytest.raises(ConfigError, match="dotted.path=value"):
        apply_overrides(raw, ["oops"])
    with pytest.raises(ConfigError, match="out of range"):
        apply_overrides(raw, ["models.5.name=x"])


def test_run_meta_has_feature_stats_for_drift_baselines(tmp_path):
    result = run(make_cfg(tmp_path))
    meta = json.loads((result.run_dir / "run_meta.json").read_text())
    stats = meta["feature_stats"]["sepal_length_cm"]
    assert set(stats) == {"mean", "std", "min", "max"} and stats["min"] < stats["max"]


# ---- automatic preprocessing ----------------------------------------------
import numpy as np

from mlbuilder.preprocess import OutlierReplacer


def test_outlier_replacer_learns_on_train_and_replaces_with_inlier_mean():
    train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [1000.0]])
    rep = OutlierReplacer().fit(train)
    assert rep.n_outliers_[0] == 1
    assert rep.fill_[0] == pytest.approx(3.0)  # mean of 1..5, not dragged by 1000
    out = rep.transform(np.array([[2.0], [999.0], [np.nan], [-500.0]]))
    assert out[0, 0] == 2.0 and out[1, 0] == pytest.approx(3.0)
    assert np.isnan(out[2, 0]) and out[3, 0] == pytest.approx(3.0)  # NaN untouched; new data uses TRAIN bounds
    assert OutlierReplacer(replace_with="median").fit(train).fill_[0] == 3.0


def test_outlier_replacer_never_touches_binary_or_constant_columns():
    X = np.array([[0.0, 7.0]] * 18 + [[1.0, 7.0]] * 2)  # 10% ones, IQR = 0
    rep = OutlierReplacer().fit(X)
    assert rep.n_outliers_.tolist() == [0, 0]
    assert (rep.transform(X) == X).all()


def _messy_frame(task, n=240, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 2, n)
    x1[:4] = 500  # outliers
    x2 = rng.normal(0, 1, n)
    x2[rng.choice(n, 12, replace=False)] = np.nan  # missing numeric
    region = rng.choice(["north", "south", "east"], n).astype(object)
    region[5] = None  # missing categorical
    signal = x2.copy()
    signal[np.isnan(signal)] = 0
    signal = signal + (region == "north") * 1.5 + rng.normal(0, 0.4, n)
    y = (signal > 0.5).astype(int) if task == "classification" else signal * 10
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "y": y})


@pytest.mark.parametrize("task,metric,floor", [("classification", "accuracy", 0.7), ("regression", "r2", 0.5)])
def test_auto_preprocess_handles_messy_data_for_both_task_types(tmp_path, task, metric, floor):
    csv = tmp_path / "messy.csv"
    _messy_frame(task).to_csv(csv, index=False)
    models = [{"name": "logistic_regression"}] if task == "classification" else [{"name": "ridge"}]
    cfg = make_cfg(tmp_path, task=task, models=models, preprocessing=[],
                   data={"source": "csv", "path": str(csv), "target": "y"},
                   split={"stratify": task == "classification"})
    result = run(cfg)
    assert result.metrics[metric] > floor

    meta = json.loads((result.run_dir / "run_meta.json").read_text())
    report = meta["preprocessing"]
    assert report["categorical_columns"] == ["region"] and meta["categorical_features"] == ["region"]
    assert report["outliers_replaced_in_train"]["x1"]["count"] >= 1
    assert report["missing_values_in_train"]["x2"] > 0

    # score a file with an UNSEEN category, a missing value and an outlier
    model, meta = load_run(result.run_dir)
    new = pd.DataFrame({"x1": [9.0, 800.0], "x2": [np.nan, 0.3], "region": ["west", "north"]})
    out = predict_frame(model, meta["features"], new, meta["categorical_features"])
    assert len(out) == 2 and out["prediction"].notna().all()


def test_auto_preprocess_off_rejects_text_features_with_a_helpful_message(tmp_path):
    csv = tmp_path / "messy.csv"
    _messy_frame("classification").to_csv(csv, index=False)
    cfg = make_cfg(tmp_path, data={"source": "csv", "path": str(csv), "target": "y"},
                   auto_preprocess={"enabled": False})
    with pytest.raises(DataError, match="auto_preprocess"):
        run(cfg)


def test_outlier_treatment_can_be_disabled_independently(tmp_path):
    csv = tmp_path / "messy.csv"
    _messy_frame("classification").to_csv(csv, index=False)
    cfg = make_cfg(tmp_path, preprocessing=[], auto_preprocess={"outliers": False},
                   data={"source": "csv", "path": str(csv), "target": "y"})
    meta = json.loads((run(cfg).run_dir / "run_meta.json").read_text())
    assert "outliers_replaced_in_train" not in meta["preprocessing"]


def test_user_steps_run_after_the_automatic_step(tmp_path):
    cfg = make_cfg(tmp_path, preprocessing=[{"name": "standard_scaler"}])
    names = [n for n, _ in __import__("mlbuilder.pipeline", fromlist=["x"]).build_pipeline(cfg, cfg.models[0]).steps]
    assert names == ["auto", "0_standard_scaler", "model"]

def test_plugin_adds_a_model_and_a_metric(tmp_path):
    cfg = make_cfg(
        tmp_path,
        plugins=["my_plugin"],
        models=[{"name": "extra_trees", "params": {"n_estimators": 20}}],
        evaluation={"metrics": ["accuracy", "balanced_accuracy"]},
    )
    result = run(cfg)
    assert result.selected_model == "extra_trees"
    assert set(result.metrics) == {"accuracy", "balanced_accuracy"}
    assert result.metrics["accuracy"] > 0.85