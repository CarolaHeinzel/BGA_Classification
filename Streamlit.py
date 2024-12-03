import streamlit as st
#import numpy as np
import altair as alt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split # für Stratifizierung der Beispiels-trainings-/testdaten

st.set_page_config(page_title="BGA Classification", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.header("BGA Classification")
#st.write("See [biorxiv](https://www.biorxiv.org/content/10.1101/2024.10.18.619150v1) for a reference what this webpage is about.")

################################
## Choose which input is used ##
################################

###### Example data ######
#file_path = "filtered_population.xlsx" #####################
#data = pd.read_excel(file_path, nrows=212) # just use the first two categories of the file as example data
#X = data.drop(data.columns[0], axis=1)
#y = data.iloc[:, 0]  # population as target

# 20% as testing data, remaining as training data
# Splitting the data while maintaining the distribution of population categories
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
    "Which training data should be used?",
    options=train_options,
    index=0,
)

if train == train_options[1]:
    st.session_state.train_file = st.file_uploader("Uploading training data: ", type=["xlsx", "xls"])
    if st.session_state.train_file:
        #try:
            #train_data = pd.read_excel(st.session_state.train_file)
        #except Exception as e:
            #st.error(f"Error reading the file: {e}")
        train_data = pd.read_excel(st.session_state.train_file)
        X_train = train_data.drop(train_data.columns[0], axis=1)
        y_train = train_data.iloc[:, 0]
        #st.write("Training data:")
        #st.dataframe(train_data)


###### Testing data ######
test_options = [
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
        y_test = test_data.iloc[:, 0]
        #st.write("Testing data:")
        #st.dataframe(test_data)


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

#Classifier is not yet applied to the selected data, but first to simple example data

submit = st.button("Submit")
if submit:
    with st.spinner('The computer is calculating...'):
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train).toarray()
        X_test_encoded = encoder.transform(X_test).toarray()

        # Create and train model
        model = RandomForestClassifier()
        model.fit(X_train_encoded, y_train)

        # Determine ROC AUC
        y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        #print(f"ROC AUC: {roc_auc}")
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

        # Display the Altair chart in Streamlit
        st.altair_chart(roc_chart, use_container_width=True)