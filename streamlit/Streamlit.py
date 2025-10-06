import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap.umap_ as umap
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="BGA Classification", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.header("BGA Classification")
#st.write("See [biorxiv](https://www.biorxiv.org/content/10.1101/2024.10.18.619150v1) for a reference what this webpage is about.")

################################
## Choose which input is used ##
################################

###### Example data ######
#file_path = "Path(__file__).parent.parent / "data" / "input_data" / "full_data.csv"
#data = pd.read_csv(file_path).sample(frac=1, random_state=42)
#X = data.drop(columns=["Population"]).copy()#.astype("category")
#y = data["Population"].copy()
#del data

### Simple example data #####################
data_train = pd.DataFrame({
    'rs1': ['AA', 'AG', 'AA', 'GG', 'AG'],
    'rs2': ['CC', 'CT', 'CC', 'TT', 'CT'],
    'target': [0, 1, 0, 1, 0]
})

data_test = pd.DataFrame({
    'rs1': ['AG', 'AG', 'AA', 'GG', 'AG'],
    'rs2': ['CT', 'CT', 'CC', 'TT', 'CT'],
    'target': [1, 1, 0, 1, 0]
})

# Target: True Class
X_train = data_train[['rs1', 'rs2']]
y_train = data_train['target']
X_test = data_test[['rs1', 'rs2']]
y_test = data_test['target']
###############################################


###### Training data ######
train_options = [
    "Example data",
    "Upload data"]
train = st.radio(
    "Which data should be used?",
    options=train_options,
    index=0,
)

if train == train_options[1]:
    st.session_state.train_file = st.file_uploader("Uploading training data: ", type=["csv"])
    if st.session_state.train_file:
        #train_data = pd.read_csv(st.session_state.train_file)
        #X_train = train_data.drop(train_data.columns[0], axis=1)
        #y_train = train_data.iloc[:, 0]

        data = pd.read_csv(st.session_state.train_file).sample(frac=1,random_state=42)
        X = data.drop(columns=["Population"]).copy()#.astype("category")
        y = data["Population"].copy()
        del data


###### Testing data ######
"""test_options = [
    "Example data",
    "Upload data"]
test = st.radio(
    "Which testing data should be used?",
    options=test_options,
    index=0,
)

if test == test_options[1]:
    st.session_state.test_file = st.file_uploader("Uploading testing data: ", type=["xlsx", "xls"])
    if st.session_state.test_file:
        test_data = pd.read_excel(st.session_state.test_file)
        X_test = test_data.drop(test_data.columns[0], axis=1)
        y_test = test_data.iloc[:, 0]"""


###### Metric ######
metric_options = [
    "ROC AUC",
    "Accuracy"]
metric = st.radio(
    "Metric: ",
    options=metric_options,
    index=0,
)


####################
## Output results ##
####################

submit = st.button("Submit")
if submit:
    with st.spinner('The computer is calculating...'):
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train).toarray()
        X_test_encoded = encoder.transform(X_test).toarray()

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)

        # UMAP with 2 dimensions
        n_components = 2
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_train_umap = reducer.fit_transform(X_train_scaled)
        X_test_umap = reducer.transform(X_test_scaled)

        # Create and train model
        model = RandomForestClassifier()
        model.fit(X_train_encoded, y_train)

        # Determine ROC AUC
        y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        st.write(f"**ROC AUC Score:** {roc_auc:.2f}")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # DataFrame for Altair
        roc_df = pd.DataFrame({
            "False Positive Rate": fpr,
            "True Positive Rate": tpr
        })
        # Altair Plot
        roc_chart = alt.Chart(roc_df).mark_line().encode(
            x=alt.X("False Positive Rate"),    #,scale=alt.Scale(domain=(0, 1))),
            y=alt.Y("True Positive Rate"),   #, scale=alt.Scale(domain=(0, 1))),
            tooltip=["False Positive Rate", "True Positive Rate"]
        ).properties(
            title="Receiver Operating Characteristic (ROC) Curve"
        )

        st.altair_chart(roc_chart, use_container_width=True)

        # Confusion matrix
        y_pred = model.predict(X_test_encoded)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix:")
        st.dataframe(cm)

        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_encoded)

        feature_names = encoder.get_feature_names_out()
        n_classes = shap_values.shape[2]
        class_names = model.classes_
        n_features = len(feature_names)

        # mean shap value over all samples
        mean_shap_values = np.zeros((n_features, n_classes))

        for sample_shap_values in shap_values:
            mean_shap_values += np.abs(sample_shap_values)

        mean_shap_values /= len(shap_values)

        # DataFrame for each class
        shap_df_list = []
        for class_index in range(n_classes):
            class_shap_df = pd.DataFrame({
                "Feature": feature_names,
                "MeanSHAP": mean_shap_values[:, class_index],
                "Class": f"Class {class_names[class_index]}"
            })
            shap_df_list.append(class_shap_df)

        shap_all_classes_df = pd.concat(shap_df_list, ignore_index=True)

        # Altair Plot
        shap_chart = alt.Chart(shap_all_classes_df).mark_bar().encode(
            x=alt.X("MeanSHAP:Q", title="Mean SHAP Value"),
            y=alt.Y("Feature:N", sort="-x", title="Feature"),
            color=alt.Color("Class:N", title="Class"),
            tooltip=["Class", "Feature", "MeanSHAP"]
        ).properties(
            title="SHAP-Werte für Features nach Klassen",
            width=800,
            height=600
        )

        st.altair_chart(shap_chart, use_container_width=True)