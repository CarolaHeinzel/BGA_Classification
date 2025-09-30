from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from feature_selection_experiments.experiments_code import (
    N_FEATURES,
    N_FOLDS,
    N_REPEATS,
    get_data,
)
from feature_selection_experiments.fe_methods.ag_fe_method import AGFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


def run_feature_selection_for_split(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    n_repeats: int,
    split_index: int,
    time_limit: int,
    eval_metric: str,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Run cross-validation for the models and compute various metrics."""
    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    split = list(kf.split(X, y))[split_index]

    # Shuffle the order of the features to ensure fair comparison
    X = X[np.random.RandomState(split_index).permutation(X.columns)]
    train_index, _ = split
    X_train, y_train = X.iloc[train_index].copy(), y.iloc[train_index].copy()

    ag_fe = AGFeatureSelector(
        time_limit=time_limit,
        eval_metric=eval_metric,
        problem_type="multiclass",
        n_features=N_FEATURES,
    ).fit(X_train, y_train)
    feature_names = ag_fe.selected_feature_names

    return {
        "fold": split_index,
        "feautre_names": feature_names,
        "n_features_target": N_FEATURES,
        "eval_metric": eval_metric,
        "time_limit": time_limit,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
    }, ag_fe.feature_importance


def run_experiments(split_index: int, time_limit: int, eval_metric: str) -> None:
    """Run our experiments."""
    output_path = Path(__file__).parent / "output_data" / "fe_results"
    output_path.mkdir(exist_ok=True, parents=True)
    X, y = get_data()
    results, feature_importance = run_feature_selection_for_split(
        X=X,
        y=y,
        n_folds=N_FOLDS,
        n_repeats=N_REPEATS,
        split_index=split_index,
        time_limit=time_limit,
        eval_metric=eval_metric,
    )

    feature_importance.to_csv(output_path / f"ag_feature_importance_{split_index}.csv")
    with (output_path / f"ag_feature_selection_{split_index}.json").open("w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    # ------------- CLI parsing -------------
    parser = argparse.ArgumentParser(
        description="Run the experiments for one data split."
    )

    parser.add_argument(
        "--split-index",
        "-s",
        type=int,
        required=True,
        help="Zero-based split to run (e.g. 0, 1, 2 …).",
    )

    parser.add_argument(
        "--time-limit",
        "-t",
        type=int,
        default=10 * 60 * 60,
        help="Time limit in seconds.",
    )
    args = parser.parse_args()

    run_experiments(
        split_index=args.split_index, time_limit=args.time_limit, eval_metric="log_loss"
    )
