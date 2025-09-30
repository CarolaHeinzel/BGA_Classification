from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from tqdm import tqdm

from feature_selection_experiments.fe_methods.ag_fe_method import (
    AGFeatureSelectorCached,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

# ------ Constants ------
N_FOLDS = 3
"""Nmber of folds for cross-validation."""
N_REPEATS = 20
"""Number of repeats for cross-validation."""
N_FEATURES = 100
"""Max number of features to select for the models with feature selection."""


def run_cross_val(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator],
    n_folds: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run cross-validation for the models and compute various metrics."""
    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    # Store results
    results = []
    raw_predictions = []
    # Cross-validation loop
    original_column_order = X.columns.tolist()
    for fold, (train_index, test_index) in tqdm(
        enumerate(kf.split(X, y), start=1), total=n_folds * n_repeats
    ):
        # Shuffle the order of the features to ensure fair comparison
        X = X[np.random.RandomState(fold).permutation(original_column_order)]

        for model_name, model in models.items():
            X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
            y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

            if (
                isinstance(model, Pipeline)
                and ("select" in model.named_steps)
                and isinstance(model.steps[0][1], AGFeatureSelectorCached)
            ):
                model.steps[0][1].fold = fold - 1

            # Ensure all columns are treated as categorical features
            if isinstance(model, TabPFNClassifier):
                # This code only works here, as all columns are categorical.
                model.categorical_features_indices = list(range(len(X_train.columns)))

            print(f"Running {model_name}")
            # Fit and predict
            fit_time = time.time()
            model.fit(X=X_train, y=y_train)
            fit_time = time.time() - fit_time
            predict_time = time.time()
            y_pred_proba = model.predict_proba(X=X_test)
            predict_time = time.time() - predict_time
            y_pred = model.predict(X=X_test)

            if isinstance(model, Pipeline) and ("select" in model.named_steps):
                feature_names = model.named_steps["select"].selected_feature_names
            else:
                feature_names = "all"

            # Store the experiment results
            results.append(
                {
                    "Fold": fold,
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                    "ROC AUC": roc_auc_score(y_test, y_pred_proba, multi_class="ovr"),
                    "Log Loss": log_loss(y_test, y_pred_proba),
                },
            )
            raw_predictions.append(
                {
                    "Fold": fold,
                    "Model": model_name,
                    "Predictions": y_pred.tolist(),
                    "Predictions Probabilities": y_pred_proba.tolist(),
                    "True Labels": y_test.tolist(),
                    "Fit Time": fit_time,
                    "Predict Time": predict_time,
                    "feature_names": feature_names,
                },
            )

    return pd.DataFrame(results), raw_predictions


def get_data():
    # Get data
    full_data_path = Path(__file__).parent / "input_data" / "full_data.csv"
    data = pd.read_csv(full_data_path).sample(frac=1, random_state=42)
    X = data.drop(columns=["Population"]).copy().astype("category")
    y = data["Population"].copy()
    del data

    # Sanity check
    n_cols = len(X.columns)
    X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
    print("Dropped constant columns:", n_cols - len(X.columns))
    return X, y
