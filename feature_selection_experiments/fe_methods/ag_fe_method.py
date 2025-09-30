from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.data import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular.models.tabm.tabm_model import TabMModel
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,  # or RegressorMixin
    TransformerMixin,
)

from feature_selection_experiments.fe_methods.ag_fe_new import logger


class TabMSklearnInterface(
    BaseEstimator, ClassifierMixin
):  # swap in RegressorMixin if needed
    def __init__(self):
        self._feature_generator = None
        self._label_cleaner = None
        # self.model_ = None
        self.problem_type = "multiclass"  # or "binary", "regression", etc.

    def fit(self, X, y):
        """Learn from the data; store whatever you need for prediction."""
        torch.cuda.empty_cache()
        # ---- minimal example: remember the training target's mean ----
        self._feature_generator, self._label_cleaner = (
            AutoMLPipelineFeatureGenerator(verbosity=0),
            LabelCleaner.construct(problem_type=self.problem_type, y=y),
        )
        X, y = (
            self._feature_generator.fit_transform(X),
            self._label_cleaner.transform(y),
        )

        fit_kwargs = {
            # "num_cpus": 8,
            # "num_gpus": 1,
            "k_fold": 4,
        }
        model = BaggedEnsembleModel(
            TabMModel(eval_metric="log_loss", problem_type=self.problem_type),
            # hyperparameters={"fold_fitting_strategy": "sequential_local"},
        )
        model.fit(X=X, y=y, **fit_kwargs)
        self.model_ = model
        return self  # always return self

    def predict(self, X):
        """Return predictions with the same length as X."""
        # dummy rule: pick the majority class seen in training
        X = self._feature_generator.transform(X)
        pred = self.model_.predict(X=X)
        return self._label_cleaner.inverse_transform(pred)

    def predict_proba(self, X):
        X = self._feature_generator.transform(X)
        return self.model_.predict_proba(X=X)


class AGFeatureSelector(BaseEstimator, TransformerMixin):
    """Use clever FE to select as many features as one needs."""

    method_name = "AGFeatures"
    ag_model = TabMModel

    def __init__(
        self, time_limit: int, eval_metric: str, problem_type: str, n_features: int
    ):
        """Initialize the AutoGluon feature selector.

        Parameters
        ----------
        time_limit: int
            Time in seconds to use for the feature selection.
        eval_metric: str
            Name of the (AutoGluon) evaluation metric to use for the model.
        problem_type: str
            The problem type, e.g. "binary", "multiclass", "regression"
        """
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.selected_feature_names: list[str] | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.n_features = n_features

    def fit(self, X, y):
        X, y = X.copy(), y.copy()
        st_time = time.time()
        time_limit = self.time_limit

        set_logger_verbosity(verbosity=2)
        logger.setLevel(10)  # Set to debug level for detailed output

        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)

        n_before_simple_pruning = X.shape[1]
        feature_generator, label_cleaner = (
            AutoMLPipelineFeatureGenerator(verbosity=0),
            LabelCleaner.construct(problem_type=self.problem_type, y=y),
        )
        X, y = (
            feature_generator.fit_transform(X),
            label_cleaner.transform(y),
        )
        logger.info(
            f"AutoGluon Useless Feature Pruning:\tPruned from {n_before_simple_pruning} to {X.shape[1]} features.",
        )

        candidate_features = X.columns.to_list()

        # Add guard that for many features, we avoid to stop right away due to time
        #   limit but prune once.
        rest_time = time_limit - (time.time() - st_time)
        logger.info(
            f"AutoGluon Feature Pruning"
            f"\n\t Time Left: {rest_time:.2f} seconds."
            f"\n\t #features: {len(candidate_features)}"
        )
        fit_kwargs = {
            # "num_cpus": 8,
            # "num_gpus": 1,
            "k_fold": 4,
        }
        model = BaggedEnsembleModel(
            self.ag_model(eval_metric=self.eval_metric, problem_type=self.problem_type),
            # hyperparameters={"fold_fitting_strategy": "sequential_local"},
        )
        model.fit(X=X, y=y, **fit_kwargs)
        model.persist_child_models()
        feature_importance = model.compute_feature_importance(
            X=X,
            y=y,
            is_oof=True,
            num_shuffle_sets=5,
        )
        candidate_features = feature_importance.index.to_list()[: self.n_features]

        self.feature_importance = feature_importance
        self.selected_feature_names = candidate_features
        return self

    def transform(self, X):
        return X[self.selected_feature_names]


class AGFeatureSelectorCached(BaseEstimator, TransformerMixin):
    """Use clever FE to select as many features as one needs."""

    method_name = "AGFeaturesC"

    def __init__(self, base_path_to_jsons: str, n_features: int):
        """Initialize the AutoGluon feature selector.

        Parameters
        ----------
        base_path_to_jsons: str
            Path to where the feature selection results are stored.
        """
        self.base_path_to_jsons = base_path_to_jsons
        self.selected_feature_names: list[str] | None = None
        self.fold: int | None = None
        self.n_features = n_features

    def fit(self, X, y):
        path = Path(self.base_path_to_jsons) / f"aggregated_feature_importance.csv"
        feature_importance = pd.read_csv(path, index_col=0)

        # importance_f = feature_importance.sort_values(by="importance", ascending=False).index[:self.n_features].tolist()
        feature_importance["pessimistic_importance"] = (
            feature_importance["importance_mean"]
        )

        self.selected_feature_names = (
            feature_importance.sort_values(by="pessimistic_importance", ascending=False)
            .index[: self.n_features]
            .tolist()
        )
        self.selected_feature_names = [str(x) for x in self.selected_feature_names]
        print(self.selected_feature_names)
        return self

    def transform(self, X):
        return X[self.selected_feature_names]
