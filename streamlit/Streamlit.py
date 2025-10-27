import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier
from pathlib import Path
import os
import sys
import json
import glob

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from feature_selection_experiments.run_experiment_feature_selection import run_feature_selection_for_split
from feature_selection_experiments.run_experiments_baselines import run_cross_val
from feature_selection_experiments.fe_methods.ag_fe_method import AGFeatureSelectorCached

from sklearn.base import BaseEstimator
from copy import deepcopy


# filepath in AGFeatureSelector changed
def get_models(n_features: int) -> dict[str, BaseEstimator]:
    """(Function from feature_selection_experiments.run_experiments_baselines, just the
    filepath is changed, and only AGFeaturesC is used.)
    Sklearn models to compare.

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
    fe_method = AGFeatureSelectorCached(
            base_path_to_jsons=str(
                Path(os.getcwd()) / "streamlit" / "output_data" / "fe_results"
            ),
            n_features=n_features,
        )
    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("select", deepcopy(fe_method)),
                ("model", deepcopy(model)),
            ]
        )
        models_with_fe[f"{fe_method.method_name}_{model_name}"] = pipe

    return {**models_with_fe}





#########################################################################################################
# ------ Constants ------
N_FOLDS = 3
#Number of folds for cross-validation.
N_REPEATS = 20
#Number of repeats for cross-validation.

########################################

st.set_page_config(page_title="BGA Classification", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.header("Permutation Importance–Based Feature Selection for Tabular Machine Learning in BGA Classification")
st.write("See ... for a reference what this webpage is about.")

################################
## Choose which input is used ##
################################

###### Example data ######

# Get data
full_data_path = Path(os.getcwd()) / "data" / "input_data" / "full_data.csv"
data = pd.read_csv(full_data_path).sample(frac=1, random_state=42)

# Select only 15 features and 50 individuals for testing to avoid long runtimes
data = data.sample(n=50, random_state=42)
y = data["Population"].copy()
data = data.drop(columns=["Population"])
data = data.sample(n=15, axis=1, random_state=42)
X = data.astype("category")
del data


###### Choose data ######
data_options = [
    "Example data",
    "Upload data"]
option = st.radio(
    "Which data should be used?",
    options=data_options,
    index=0,
)

if option == data_options[0]:
    st.write(pd.concat([y, X], axis=1))

    # Sanity check
    n_cols = len(X.columns)
    X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
    # st.write("Dropped constant columns:", n_cols - len(X.columns))

if option == data_options[1]:
    st.session_state.file = st.file_uploader("Uploading data: ", type=["csv"])
    if st.session_state.file:

        data = pd.read_csv(st.session_state.file).sample(frac=1, random_state=42)
        st.write(data)
        X = data.drop(columns=["Population"]).copy().astype("category")
        y = data["Population"].copy()
        del data

        # Sanity check
        n_cols = len(X.columns)
        X = X[[col for col in X.columns if X[col].nunique(dropna=False) > 1]]
        #st.write("Dropped constant columns:", n_cols - len(X.columns))


###### Select Number of features ######
N_FEATURES = st.number_input(
    "Number of features to select: ",
    min_value=1,       # optional: untere Grenze
    max_value=500,    # optional: obere Grenze
    value=100,          # 🔹 Standardwert
    step=1,            # nur ganze Zahlen
)



####################
## Output results ##
####################

submit = st.button("Submit")
if submit:

    if N_FEATURES > X.shape[0]:
        st.error("More features selected then available")
        st.stop()

    ############## 1) Calculate feature importances ##############
    with st.spinner('Calculation of feature importances...'):

        output_path = Path(os.getcwd()) / "streamlit" / "output_data" / "fe_results"
        output_path.mkdir(exist_ok=True, parents=True)

        for split_index in range(N_FOLDS * N_REPEATS):
            with st.spinner(f"==== Split {split_index}"):
                #st.write(f"==== Split {split_index}")
                results, feature_importance = run_feature_selection_for_split(
                    X=X,
                    y=y,
                    n_folds=N_FOLDS,
                    n_repeats=N_REPEATS,
                    split_index=split_index,
                    time_limit=10 * 60 * 60,
                    eval_metric="log_loss",
                )

                feature_importance.to_csv(output_path / f"ag_feature_importance_{split_index}.csv")
                with (output_path / f"ag_feature_selection_{split_index}.json").open("w") as f:
                    json.dump(results, f)

        st.write(f"Feature importances saved to {output_path}")

    ############## 2) Aggregate features ##############

    with st.spinner('Aggregation of features...'):

        # Path to your folder
        folder_path = Path(os.getcwd()) / "streamlit" / "output_data" / "fe_results"  # Change this to your folder path

        # Pattern for matching files
        file_pattern = os.path.join(folder_path, "ag_feature_importance_*.csv")

        # Collect all matching files
        files = glob.glob(file_pattern)

        # Initialize an empty DataFrame
        agg_df = pd.DataFrame()

        # Process each file
        for file in files:
            df = pd.read_csv(file, index_col=0)  # Index is the feature ID
            # If agg_df is empty, initialize with this file
            if agg_df.empty:
                agg_df = df[['importance']].copy()
                agg_df.rename(columns={'importance': f"importance_{os.path.basename(file)}"}, inplace=True)
            else:
                agg_df = agg_df.join(
                    df[['importance']].rename(columns={'importance': f"importance_{os.path.basename(file)}"}),
                    how='outer')

        # Replace NaN with 0 for aggregation
        agg_df = agg_df.fillna(0)

        # Compute aggregated importance (sum across all files)
        agg_df['importance_sum'] = agg_df.sum(axis=1)

        # Or compute mean importance
        agg_df['importance_mean'] = agg_df.mean(axis=1)

        # Sort by sum (or mean) in descending order
        agg_df = agg_df.sort_values('importance_sum', ascending=False)

        # Save aggregated results
        output_path = os.path.join(folder_path, "aggregated_feature_importance.csv")
        agg_df.to_csv(output_path)

        #st.write(f"Aggregated feature importance saved to {output_path}")

    ############## 3) Evaluation of the selected features with Cross-validation ##############

    with st.spinner('Running Crossvalidation...'):

        models = get_models(n_features=N_FEATURES)

        results_df, raw_predictions = run_cross_val(
            X=X, y=y, models=models, n_repeats=N_REPEATS, n_folds=N_FOLDS
        )


    ############## Output ##############

    st.subheader("Results")

    for model_name in ["TabPFN", "NaiveBayes"]:
        st.write(f"**Metrics for {model_name}**")

        full_name = "AGFeaturesC_" + model_name

        results = results_df[results_df["Model"] == full_name]
        #st.write(results)
        roc_auc = results["ROC AUC"].mean()
        accuracy = results["Accuracy"].mean()

        st.write(f"ROC AUC Score: {roc_auc:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")

    # Selected features
    model = models["AGFeaturesC_TabPFN"]
    feature_names = model.named_steps["select"].selected_feature_names
    st.write(f"\n **Selected features:**", list(feature_names))

    #st.write(f"**ROC AUC Score:** {roc_auc:.2f}")

    #fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # DataFrame for Altair
    #roc_df = pd.DataFrame({
    #    "False Positive Rate": fpr,
    #    "True Positive Rate": tpr
    #})
    # Altair Plot
    #roc_chart = alt.Chart(roc_df).mark_line().encode(
    #    x=alt.X("False Positive Rate"),    #,scale=alt.Scale(domain=(0, 1))),
    #    y=alt.Y("True Positive Rate"),   #, scale=alt.Scale(domain=(0, 1))),
    #    tooltip=["False Positive Rate", "True Positive Rate"]
    #).properties(
    #    title="Receiver Operating Characteristic (ROC) Curve"
    #)

    #st.altair_chart(roc_chart, use_container_width=True)"""