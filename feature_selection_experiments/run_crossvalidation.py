
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
#!pip install tabpfn
from tabpfn import TabPFNClassifier
from sklearn.base import BaseEstimator
from scipy.special import comb
import math
from pathlib import Path


# ------ Constants ------
N_FOLDS = 3
"""Number of folds for cross-validation."""
N_REPEATS = 20
"""Number of repeats for cross-validation."""
N_FEATURES = 100
"""Max number of features to select for the models with feature selection."""



def get_models(categorical_features_indices: list[int]) -> dict[str, BaseEstimator]:
    """Sklearn models to compare."""
    return {
        "TabPFN": TabPFNClassifier(
            random_state=np.random.RandomState(42),
            ignore_pretraining_limits=True,
            categorical_features_indices=categorical_features_indices
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


def run_cross_val(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int,
    n_folds: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run cross-validation for the models and compute various metrics."""
    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    # Store results
    results = []
    raw_predictions = []

    # Cross-validation loop
    original_column_order = X.columns.tolist()

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"==== Fold {fold}")

        # Shuffle the order of the features to ensure fair comparison
        X = X[np.random.RandomState(fold).permutation(original_column_order)]

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # all data is categorical
        categorical_features_indices = [i for i in range(X.shape[1])]

        models = get_models(categorical_features_indices)

        for model_name, model in models.items():
            print(f"Running {model_name}")
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

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
                    #"feature_names": markers,
                },
            )

    return pd.DataFrame(results), raw_predictions


def get_data(file_name):
    data_path = Path(__file__).parent.parent / "data" / "input_data"
    data = pd.read_csv(data_path / file_name).sample(frac=1, random_state=42)
    #data = pd.read_csv(file_name).sample(frac=1, random_state=42)

    X = data.drop(columns=["Population", "ID"]).copy().astype("category")
    y = data["Population"].copy()
    del data

    # Sanity check
    n_cols = len(X.columns)
    X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
    print("Dropped constant columns:", n_cols - len(X.columns))

    return X, y


def run_experiments():
    """Run our experiments."""

    #file_name = "1-s2.0-S1872497323000285-mmc5_EUR.csv"
    #file_name = "all_1000G_Enhanced.csv"
    # file_name = "EUR_5_markerselection_try2.csv"
    file_name = "all_1000G_new_markers.csv"

    X, y = get_data(file_name)
    print(X)

    results_df, raw_predictions = run_cross_val(X=X, y=y, n_repeats=N_REPEATS, n_folds=N_FOLDS)

    # Save results to disk
    #path = Path(__file__).parent.parent
    results_df.to_csv(f"results_{file_name}.csv", index=False)
    with open(f"results_{file_name}.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
