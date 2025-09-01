
import json
#import io
import os
import glob
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
#from xgboost import XGBClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from scipy.special import comb
from cyvcf2 import VCF
import math


def vcf_to_df(vcf_file):
    vcf = VCF(vcf_file)
    samples = vcf.samples
    records = []

    for variant in vcf:
        gts = np.array(variant.genotypes)

        # True für fehlende Werte (hier: drittes Element == 0 oder gt_types==3)
        missing_mask = ~gts[:, 2].astype(bool)

        # Strings für gültige Genotypen
        # ALT-Allele immer als 1 codieren
        gt_strings = np.char.add(np.where(
            gts[:, 0].astype(str) != "0",
            "1", "0"), "|" + np.where(gts[:, 1].astype(str) != "0",
                                      "1", "0"))

        # Fehlende Werte setzen
        gt_strings[missing_mask] = "-1"

        # Record zusammenbauen: Markerposition + alle Genotyp-Strings
        record = np.concatenate(([str(variant.CHROM) + "_" + str(variant.POS)], gt_strings))
        records.append(record)

    df = pd.DataFrame(records, columns=['marker'] + samples)
    df = df.set_index('marker').astype('category')
    #print(df)
    return df


def load_data(data_path_vcf, data_path_y) -> [pd.DataFrame, pd.Series]:
    """Read all vcf files and create a Dataframe in the right format for TabPFN-
    Classification. And create a series with the matching y-values (populations)."""

    files = glob.glob(os.path.join(data_path_vcf, "*.vcf"))
    data = pd.concat([vcf_to_df(f) for f in files])
    data = data.T

    # read in the y-values
    populations = pd.read_csv(data_path_y, sep="\t", header=None)
    populations = populations.set_index(populations.columns[0])
    y = populations.loc[data.index] # only evaluate the populations occurring in the files
    y = y[1]

    return data, y


def compute_allelefrequencies(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> dict[str, dict[str, float]]:
    """Compute the allelefrequencies for each population from a given Dataframe with DNA data"""

    frequencies = {}
    for pop in populations:
        marker_frequencies = {}
        indices = y[y == pop].index
        X_pop = X.loc[indices]
        # Schneller: Zähle '1' in allen Genotypen direkt vektorisert für alle Marker
        allele_counts = X_pop.apply(lambda col: col.str.count('1'), axis=0).sum(axis=0)
        # Frequenz berechnen: (Summe der '1'-Allele) / (Anzahl Individuen * 2)
        marker_frequencies = (allele_counts / (len(indices) * 2)).to_dict()

        frequencies[pop] = marker_frequencies

    return frequencies


def find_markers_by_difference_of_allelefrequencies(frequencies: dict[str, dict[str, float]], pop1: str, pop2: str, top_n: int) -> pd.Index:
    freq1, freq2 = pd.Series(frequencies[pop1]), pd.Series(frequencies[pop2])
    diff = (freq1 - freq2).abs()

    top_marker = diff.sort_values(ascending=False).head(top_n)
    #print(top_marker.index)
    return top_marker.index


def get_dataset_allele_features(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> [list[str], list[int]]:
    """Select features by the maximum difference in allele frequencies between different populations"""
    #populations = y.unique().tolist()
    n = len(populations)
    print("Computation of frequencies...")
    frequencies = compute_allelefrequencies(X, y, populations)

    # ensure that the maximum number of features for TabPFN is maintained
    top_n = math.floor(500/comb(n, 2, exact=True)) # Number of markers selected when selecting markers between two populations

    markers = set()
    for i in range(n):
        for j in range(i + 1, n):
            pop1, pop2 = populations[i], populations[j]
            markers.update(find_markers_by_difference_of_allelefrequencies(frequencies, pop1, pop2, top_n))
    #X_allel = X[list(markers)] # each marker should only occur once
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
            categorical_features_indices=categorical_features_indices
        ),
        "Naive Bayes": Pipeline(
            [
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("classifier", CategoricalNB()),
            ],
        ),
    }


def run_cross_val(
    *,
    data: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run cross-validation for the models and compute various metrics."""
    # Cross-validation setup
    # Folds: 5, because this allows us to have for the minor class (Russian) at least 5 sample in each test fold
    # Repeats: 10, to have a more robust estimate of the model performance
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Store results
    results = []
    raw_predictions = []

    populations = y.unique().tolist()

    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(data, y), start=1):
        print(f"==== Fold {fold}")

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        data_train = data.iloc[train_index]

        # Select features based on allele frequency differences computed from the training data
        markers, categorical_features_indices = get_dataset_allele_features(data_train, y_train, populations)

        X = data[markers]  # feature selection
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        print(X)

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
                },
            )

    return pd.DataFrame(results), raw_predictions


def run_experiments():
    """Run our experiments."""

    data_path_vcf = "AFR"  # folder containing the vcf files with DNA data
    data_path_populations = "1000G_SampleListWithLocations.txt" # csv file with y-values (populations) for all three data sets

    data, y = load_data(data_path_vcf, data_path_populations)
    print(data, y)

    results_df, raw_predictions = run_cross_val(data=data, y=y)

    # Save results to disk
    results_df.to_csv(f"results_TabPFN_NB_allele_{data_path_vcf}.csv", index=False)
    with open(f"results_TabPFN_NB_allele_{data_path_vcf}.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
