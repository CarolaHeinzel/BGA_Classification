from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

# Example Data
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


encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train).toarray()
X_test_encoded = encoder.transform(X_test).toarray()

# Creae and train model
model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)

# Determine ROC AUC
y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]  
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC AUC: {roc_auc}")

# Plot ROC AUC 
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="best")
plt.show()
