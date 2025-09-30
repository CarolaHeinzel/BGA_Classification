from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

from feature_selection_experiments.experiments_code import (
    N_FEATURES,
    N_FOLDS,
    N_REPEATS,
    get_data,
    run_cross_val,
)
from feature_selection_experiments.fe_methods.ag_fe_method import (
    AGFeatureSelectorCached,TabMSklearnInterface,
)
from feature_selection_experiments.fe_methods.baseline_fe_methods import (
    RandomFeatureSelector,
    StaticExpertFeatureSelector,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def get_models(n_features: int) -> dict[str, BaseEstimator]:
    """Sklearn models to compare.

    Parameters
    ----------
    n_features : int
        The maximum number of features to select for the models with feature selection.
    """
    models = {
        "TabPFN": TabPFNClassifier(
            random_state=np.random.RandomState(42), ignore_pretraining_limits=True
        ),
        "NaiveBayes": Pipeline(
            [
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("classifier", CategoricalNB()),
            ],
        ),
    }

    models_with_fe = {}
    for fe_method in [
        AGFeatureSelectorCached(
            base_path_to_jsons=str(
                Path(__file__).parent / "output_data" / "fe_results"
            ),
            n_features=n_features,
        ),
        # RandomFeatureSelector(
        #     random_state=np.random.RandomState(42), n_features=n_features
        # ),
        StaticExpertFeatureSelector(),
    ]:
        for model_name, model in models.items():
            pipe = Pipeline(
                steps=[
                    ("select", deepcopy(fe_method)),
                    ("model", deepcopy(model)),
                ]
            )
            models_with_fe[f"{fe_method.method_name}_{model_name}"] = pipe

    return {**models_with_fe} # , **models


def run_experiments():
    """Run our experiments."""
    X, y = get_data()
    models = get_models(n_features=N_FEATURES)
    results_df, raw_predictions = run_cross_val(
        X=X, y=y, models=models, n_repeats=N_REPEATS, n_folds=N_FOLDS
    )

    # Save results to disk
    results_df.to_csv(
        Path(__file__).parent / "output_data" / "baseline_results.csv", index=False
    )
    with (Path(__file__).parent / "output_data" / "baseline_results.json").open(
        "w"
    ) as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
