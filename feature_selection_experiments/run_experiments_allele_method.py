
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


def compute_allelefrequencies(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> dict[str, dict[str, float]]:
    """Compute the allelefrequencies for each population from a given Dataframe with DNA data"""

    frequencies = {}

    X_freq = pd.DataFrame(index=X.index, columns=X.columns)
    for col in X.columns:
        idx = X[col].first_valid_index()

        # whole column has no entry
        if idx == None:
            X_freq[col] = 0
        else:
            char = X[col][idx][0]
            X_freq[col] = X[col].astype(str).str.count(char)

    for pop in populations:
        indices = y[y == pop].index
        X_pop = X_freq.loc[indices]
        allele_counts = X_pop.sum(axis=0)

        # Frequenz berechnen: (Summe der '1'-Allele) / (Anzahl Individuen mit gültigen Einträgen * 2)
        #marker_frequencies = (allele_counts / (len(indices) * 2)).to_dict()
        marker_frequencies = {col: (allele_counts[col] / (X_pop[col].count() * 2))
                                if X_pop[col].count() > 0 else 0 for col in X_pop.columns}

        frequencies[pop] = marker_frequencies

    return frequencies


def find_markers_by_difference_of_allelefrequencies(frequencies: dict[str, dict[str, float]], pop1: str, pop2: str, top_n: int) -> pd.Index:
    freq1, freq2 = pd.Series(frequencies[pop1]), pd.Series(frequencies[pop2])
    diff = (freq1 - freq2).abs()

    top_marker = diff.sort_values(ascending=False).head(top_n)
    #print(top_marker.index)
    return top_marker.index


def get_markers(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> [list[str], list[int]]:
    """Select features by the maximum difference in allele frequencies between different populations"""

    print("Computation of frequencies...")

    frequencies = compute_allelefrequencies(X, y, populations)

    markers = set()  # each marker should only occur ones

    # ensure that exactly N markers are selected
    n = len(populations)
    top_n = math.floor(N_FEATURES / comb(n, 2, exact=True))
    find = False

    while len(markers) < N_FEATURES:
        for i in range(n):
            for j in range(i + 1, n):
                pop1, pop2 = populations[i], populations[j]
                markers.update(find_markers_by_difference_of_allelefrequencies(frequencies, pop1, pop2, top_n))
                if len(markers) == 100:
                    find = True
                    break
            if find:
                break
        if find:
            break
        top_n += 1

    markers = list(markers)

    print(f"Number of selected features: {len(markers)}")

    # all data is categorical
    categorical_features_indices = [i for i in range(len(markers))]

    return markers, categorical_features_indices


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
    data: pd.DataFrame,
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
    original_column_order = data.columns.tolist()

    for fold, (train_index, test_index) in enumerate(kf.split(data, y), start=1):
        print(f"==== Fold {fold}")

        # Shuffle the order of the features to ensure fair comparison
        data = data[np.random.RandomState(fold).permutation(original_column_order)]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #################### feature selection ####################
        data_train = data.iloc[train_index]

        # Select features based on allele frequency differences computed from the training data
        populations = y_train.unique().tolist()
        markers, categorical_features_indices = get_markers(data_train, y_train, populations)

        X = data[markers]
        ###########################################################

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

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
                    "feature_names": markers,
                },
            )

    return pd.DataFrame(results), raw_predictions


def get_data():
    data_path = Path(__file__).parent / "data" / "input_data" / "full_data.csv"
    data = pd.read_csv(data_path).sample(frac=1, random_state=42)
    X = data.drop(columns=["Population"]).copy()#.astype("category")
    y = data["Population"].copy()
    del data

    merge_map = {
        "CA": "AC",
        "GA": "AG",
        "TA": "AT",
        "GC": "CG",
        "TC": "CT",
        "TG": "GT"
    }

    # ensure that "1|0" and "0|1" are treated as the same"
    X = X.replace(merge_map).astype("category")

    # Sanity check
    n_cols = len(X.columns)
    X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
    print("Dropped constant columns:", n_cols - len(X.columns))

    return X, y


def run_experiments():
    """Run our experiments."""

    data, y = get_data()

    results_df, raw_predictions = run_cross_val(data=data, y=y, n_repeats=N_REPEATS, n_folds=N_FOLDS)

    # Save results to disk
    path = Path(__file__).parent / "data" / "output_data"
    results_df.to_csv(path / "results_allele.csv", index=False)
    with open(path / "results_allele.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
