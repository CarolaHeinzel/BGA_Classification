
import json
import io
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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator



def get_dataset(data_path) -> [pd.DataFrame, pd.Series, list[int]]:
    """Read in data for experiment."""

    data = pd.read_excel(data_path)
    # Convert features to categorical but leave the target variable as is
    X = data.drop(data.columns[:6], axis=1).astype("category")  # Only convert features
    y = data["Population"]  # Leave target variable as is

    categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == "category"]
    return X, y, categorical_features_indices


def vcf_to_df(vcf_file) -> pd.DataFrame:
    with open(vcf_file, 'r') as vcf:
        lines = [line for line in vcf if not line.startswith('##')]

    header = lines[0].strip().split('\t')
    data = [line.strip().split('\t') for line in lines[1:]]

    df = pd.DataFrame(data, columns=header)
    df = df.drop(['#CHROM', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis=1)
    df = df.set_index(df.columns[0]) # marker as indices
    return df


def get_dataset_vcf(data_path) -> [pd.DataFrame, pd.Series]:
    """Read all vcf files and create a Dataframe in the right format
    for Classification. And create a series with the matching y-values (populations)"""

    # find all vcf files
    files = glob.glob(os.path.join(data_path, "*.vcf"))
    ###files = glob.glob("*.vcf")
    #print(files)

    data = pd.concat([vcf_to_df(f) for f in files])
    X = data.astype("category").T

    # read in the y-values
    populations = pd.read_csv("1000G_SampleListWithLocations.txt", sep="\t", header=None)
    populations = populations.set_index(populations.columns[0])
    y = populations.loc[X.index]
    y = y[1]

    return X, y


def get_features_random(X: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, pd.Series, list[int]]:
    # 500 random markers
    X_random = X.sample(n=500, axis=1, random_state=42)
    #print(X_random)
    categorical_features_indices = [i for i, col in enumerate(X_random.columns) if X_random[col].dtype.name == "category"]
    return X_random, y, categorical_features_indices


def compute_allelefrequencies(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> dict[str, dict[str, float]]:
    """Compute the allelefrequencies for each population from a given Dataframe with DNA data"""

    frequencies = {}
    for pop in populations:
        marker_frequencies = {}
        indices = y[y == pop].index
        X_pop = X.loc[indices]
        for marker in X_pop.columns:
            alleles = 0
            for gt in X_pop[marker]:
                #a = gt.replace('|', '/').split('/') # Einträge des Genotyp als Liste (und falls auch '/' vorkommt)
                alleles += gt.count('1')
            marker_frequencies[marker] = alleles/len(indices) #/2

        frequencies[pop] = marker_frequencies

    return frequencies


def find_markers_by_difference_of_allelefrequencies(frequencies: dict[str, dict[str, float]], pop1: str, pop2: str, top_n=83) -> pd.Index:
    top_n = 122
    freq1, freq2 = pd.Series(frequencies[pop1]), pd.Series(frequencies[pop2])
    diff = (freq1 - freq2).abs()

    top_marker = diff.sort_values(ascending=False).head(top_n)
    #print(top_marker.index)
    return top_marker.index


def get_features_allele(X: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, pd.Series, list[int]]:
    """Select features by the maximum difference in allele frequencies between different populations"""
    populations = ['GBR', 'TSI', 'FIN', 'IBS']
    frequencies = compute_allelefrequencies(X, y, populations)

    markers = []
    for i in range(len(populations)):
        for j in range(i + 1, len(populations)):
            pop1, pop2 = populations[i], populations[j]
            markers += find_markers_by_difference_of_allelefrequencies(frequencies, pop1, pop2).tolist()

    X_allel = X[list(set(markers))]
    categorical_features_indices = [i for i, col in enumerate(X_allel.columns) if X_allel[col].dtype.name == "category"]
    return X_allel, y, categorical_features_indices


def get_models(categorical_features_indices: list[int]) -> dict[str, BaseEstimator]:
    """Sklearn models to compare."""
    return {
        "TabPFN": TabPFNClassifier(
            random_state=np.random.RandomState(42),
            categorical_features_indices=categorical_features_indices
        ),
    }


def run_cross_val(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, BaseEstimator],
) -> tuple[pd.DataFrame, list[dict]]:
    """Run cross-validation for the models and compute various metrics."""
    # Cross-validation setup
    # Folds: 5, because this allows us to have for the minor class (Russian) at least 5 sample in each test fold
    # Repeats: 10, to have a more robust estimate of the model performance
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Store results
    results = []
    raw_predictions = []
    print(results)
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        print(f"==== Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
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

    ####### Select which data should be evaluated #######

    ##### data from an excel file #####
    #data_path = "filtered_population_eur.xlsx"
    #X, y, categorical_features_indices = get_dataset(data_path)

    ##### vcf data #####
    X, y = get_dataset_vcf("data_vcf") # Folder containing the vcf files with DNA data
    ### feature selection ###
    # select 500 random features
    X, y, categorical_features_indices = get_features_random(X, y)
    # select features by the maximum difference in allele frequencies
    #X, y, categorical_features_indices = get_features_allele(X, y)

    print(X, y)

    models = get_models(categorical_features_indices=categorical_features_indices)
    results_df, raw_predictions = run_cross_val(X=X, y=y, models=models)

    # Save results to disk
    results_df.to_csv("results_TabPFN.csv", index=False)
    with open("results_TabPFN.json", "w") as f:
        json.dump(raw_predictions, f)


if __name__ == "__main__":
    run_experiments()
