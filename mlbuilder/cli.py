"""Command line interface:  python -m mlbuilder train|predict|list ..."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import load_config
from .errors import MlBuilderError
from .inference import load_run, predict_frame
from .registry import LOADERS, METRICS, MODELS, PREPROCESSORS, SPLITTERS, STAGES
from .runner import load_plugins, run


def _train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.set)
    result = run(cfg)
    print(f"\nRun directory : {result.run_dir}")
    print(f"Selected model: {result.selected_model}")
    if result.cv_results:
        print("CV results    :", {k: round(v["mean"], 4) for k, v in result.cv_results.items()})
    if result.metrics:
        print("Test metrics  :", {k: round(v, 4) for k, v in result.metrics.items()})
    return 0


def _predict(args: argparse.Namespace) -> int:
    model, meta = load_run(args.run_dir)
    df = pd.read_csv(args.input)
    out = predict_frame(model, meta["features"], df, meta.get("categorical_features", []))
    out.to_csv(args.output, index=True)
    print(f"Wrote {len(out)} predictions to {args.output}")
    return 0


def _list(args: argparse.Namespace) -> int:
    load_plugins(args.plugin)
    for title, reg in [("stages", STAGES), ("data sources", LOADERS), ("split strategies", SPLITTERS),
                       ("preprocessors", PREPROCESSORS), ("models", MODELS), ("metrics", METRICS)]:
        print(f"{title:18}: {', '.join(reg.names())}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlbuilder", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="build a model from a config file")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="override a config value, e.g. --set split.test_size=0.3 "
                              "--set models.0.params.max_iter=500 (repeatable)")
    p_train.set_defaults(func=_train)

    p_pred = sub.add_parser("predict", help="score a CSV with a previously trained run")
    p_pred.add_argument("--run-dir", required=True)
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", default="predictions.csv")
    p_pred.set_defaults(func=_predict)

    p_list = sub.add_parser("list", help="show every registered component")
    p_list.add_argument("--plugin", action="append", default=[], help="plugin module to import first")
    p_list.set_defaults(func=_list)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.func(args)
    except MlBuilderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
