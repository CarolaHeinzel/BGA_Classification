
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
"""Number of features to select for the models with feature selection."""


def compute_allelefrequencies(X: pd.DataFrame, y: pd.Series,
                              populations: list[str]) -> dict[str, dict[str, float]]:
    """
    Compute the allele frequencies for each population from a given Dataframe with DNA data

    Resulting dictionary:
    {pop1: {marker1: freq1, marker2: freq2, ...}, pop2: {marker1: ...}}
    """
    frequencies = {}

    for pop in populations:
        indices = y[y == pop].index
        X_pop = X.loc[indices]

        marker_frequencies = {}

        for col in X_pop.columns:
            valid = X_pop[col].dropna()

            if len(valid) > 0:
                # Frequency calculation: (sum of '1'-alleles) / (number of individuals with valid records * 2)
                allele_frequency = valid.sum() / (len(valid) * 2)
            else:
                allele_frequency = 0

            marker_frequencies[col] = allele_frequency

        frequencies[pop] = marker_frequencies

    return frequencies


def find_topn_markers(frequencies: dict[str, dict[str, float]], pop1: str,
                      pop2: str, top_n: int, metric: str) -> pd.Index:
    """Find the top_n markers for a given metric and two populations."""

    freq1, freq2 = pd.Series(frequencies[pop1]), pd.Series(frequencies[pop2])

    if metric == "delta":
        # absolute allele frequency difference
        diff = (freq1 - freq2).abs()
        top_markers = diff.sort_values(ascending=False).head(top_n)
    elif metric == "f_st":
        # F Statistic
        denominator = (freq1 + freq2) * (2 - (freq1 + freq2))
        F_st = ((freq1 - freq2) ** 2 / denominator).where(
            denominator > 0,
            0
        )
        top_markers = F_st.sort_values(ascending=False).head(top_n)

    #print(top_marker.index)
    return top_markers.index


def xlogx(x):
    """
    Avoid log(0) when calculating x*log(x)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, x * np.log(x), 0.0)


def get_markers(X: pd.DataFrame, y: pd.Series, populations: list[str]) \
        -> [[list[str], list[str], list[str]], list[int]]:
    """
    Select features based on three Informativeness measures:
    1) Absolute allele frequency differences (delta)
    2) Informativeness for Assignment Measure (I_n)
    3) F-Statistics (F_st)

    The function returns a list containing a list with N_FEATURES markers for each measure.
    """

    print("Computation of frequencies...")

    frequencies = compute_allelefrequencies(X, y, populations)

    marker_sets = {}

    #################### Markers by absolute allele frequency differences ####################
    top_markers = set()  # each marker should only occur ones

    # ensure that exactly N markers are selected
    K = len(populations)
    top_n = math.floor(N_FEATURES / comb(K, 2, exact=True))

    while len(top_markers) < N_FEATURES:
        for i in range(K):
            for j in range(i + 1, K):
                pop1, pop2 = populations[i], populations[j]
                top_markers.update(find_topn_markers(frequencies, pop1, pop2, top_n, metric="delta"))
                if len(top_markers) == N_FEATURES:
                    break
            if len(top_markers) >= N_FEATURES:
                break
        if len(top_markers) >= N_FEATURES:
            break
        top_n += 1

    marker_sets["delta"] = list(top_markers)[:N_FEATURES]
    ################################################################################

    ###### Markers by Informativeness for Assignment Measure (Rosenberg et al., 2003) ######
    ###### + global F-Statistics (F_{ST}) #########
    markers = frequencies[populations[0]].keys()

    results_I_n = {}
    results_F_st = {}

    for marker in markers:
        # p_i für diesen Marker aus allen Populationen
        p_i = np.array([
            frequencies[pop][marker]
            for pop in populations
        ])

        p_bar = np.mean(p_i)

        I_n = (
                -xlogx(p_bar)
                + np.sum(xlogx(p_i) / K)
                - xlogx(1 - p_bar)
                + np.sum(xlogx(1 - p_bar) / K)
        )

        # F_{ST} = (H_T - H_S) / H_T
        F_st = (2*p_bar*(1-p_bar) - np.sum(2*p_i*(1-p_i)) / K) / (2*p_bar*(1-p_bar))

        results_I_n[marker] = I_n
        results_F_st[marker] = F_st

    top_markers_i_n = sorted(
        results_I_n,
        key=results_I_n.get,
        reverse=True
    )[:N_FEATURES]

    marker_sets["informativeness"] = list(top_markers_i_n)

    top_markers_f_st = sorted(
        results_F_st,
        key=results_F_st.get,
        reverse=True
    )[:N_FEATURES]

    marker_sets["f_st"] = list(top_markers_f_st)
    ################################################################################

    # all data is categorical
    categorical_features_indices = [i for i in range(len(markers))]

    return marker_sets, categorical_features_indices


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

    for fold, (train_index, test_index) in enumerate(kf.split(data, y), start=1):
        print(f"==== Fold {fold}")

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        data_train = data.iloc[train_index]

        populations = y_train.unique().tolist()
        print(populations)
        marker_sets, categorical_features_indices = get_markers(data_train, y_train, populations)

        for marker_method in ["delta", "informativeness", "f_st"]:
            # feature selection
            markers = marker_sets[marker_method]
            X = data[markers]

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
                        "FS Method": marker_method,
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
                        "FS Method": marker_method,
                        "Predictions": y_pred.tolist(),
                        "Predictions Probabilities": y_pred_proba.tolist(),
                        "True Labels": y_test.tolist(),
                        "feature_names": markers,
                    },
                )

    return pd.DataFrame(results), raw_predictions


def get_data():
    """
    Read a gzip-compressed CSV file of the following structure:
    Position,HG00096,HG00097,HG00099,...
    Population,GBR,GBR,GBR,...
    1:10583,0,1,2,...
    1:10611,1,0,0,...
    2:10133,0,2,1,...
    ...

    Missing values are written as "NA".
    """

    data_path = Path(__file__).parent.parent / "Data_revision" / "EUR.MAF01.biallelicSNP.allchr.dosage_test.csv.gz"

    df = pd.read_csv(
        data_path,
        index_col=0,
        na_values=["NA"]  # decode missing values as NaN
    )

    df = df.T  # transform the file so that features are columns

    X = df.drop(columns=["Population"]).copy()#.astype("category")
    y = df["Population"].copy()
    del df

    # Sanity check
    n_cols = len(X.columns)
    X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
    print("Dropped constant columns:", n_cols - len(X.columns))

    # Convert genotypes into numerical values
    X = X.apply(pd.to_numeric, errors="coerce")

    print(X)
    print(y)

    return X, y


def run_experiments():
    """Run our experiments."""

    data, y = get_data()

    results_df, raw_predictions = run_cross_val(data=data, y=y, n_repeats=N_REPEATS, n_folds=N_FOLDS)

    #print(results_df)

    # Save results to disk
    path = Path(__file__).parent.parent / "data" / "output_data"
    results_df.to_csv(path / "results_allele.csv", index=False)
    with open(path / "results_allele.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
