from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path



def get_labels(data, model):
    true = []
    pred = []
    n = len(data) # num repeats of the experiments
    for i in range(n):
        if data[i]['Model'] == model:
            last_key = "True Labels"

            last_element = (last_key, data[i][last_key])

            second_key = 'Predictions'
            second_element = (second_key, data[i][second_key])
            true.append(last_element[1])
            pred.append(second_element[1])
    flattened_list_pred = [item for sublist in pred for item in sublist]
    flattened_list_true = [item for sublist in true for item in sublist]

    true_labels_1 = flattened_list_true
    predictions_1 = flattened_list_pred

    return true_labels_1, predictions_1


def plot_confusion_matrix(true_labels, predictions, title, vmin, vmax):
    classes = sorted(list(set(true_labels + predictions)))
    
    true_labels_numeric = [classes.index(label) for label in true_labels]
    predictions_numeric = [classes.index(label) for label in predictions]

    cm = confusion_matrix(true_labels_numeric, predictions_numeric, labels=range(len(classes)))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=20)
    ax.set_yticklabels(classes, fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=20)

    ax.set_xlabel('Predicted Population', fontsize=24)
    ax.set_ylabel('True Population', fontsize=24)
    plt.tight_layout()
    plot_path = Path(__file__).parent.parent / "data" / "output_data" / "plots" / "cms" / f"cm_{model}_mmc5_EUR.pdf"
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    #data_path = Path(__file__).parent.parent / "data" / "output_data" / "results_allele.json"
    data_path = Path(__file__).parent.parent / "results_1-s2.0-S1872497323000285-mmc5_EUR.json"
    with open(data_path, 'r') as file:
        data = json.load(file)

    #models = ["AGFeaturesC_TabPFN", "AGFeaturesC_NaiveBayes", #"RandomFeatures_TabPFN", "RandomFeatures_NaiveBayes",
              #"StaticExpertFeatures_TabPFN", "StaticExpertFeatures_NaiveBayes"]#, "TabPFN", "NaiveBayes"]
    models = ["TabPFN", "NaiveBayes"]

    for model in models:
        true_labels_1, predictions_1 = get_labels(data, model)

        mapping = {'09. Russia - Russian': 'RUS', 'British in England and Scotland': 'GBR', 'Finnish in Finland': 'FIN',
                   'France': 'FRA', 'Iberian population in Spain': 'IBS', 'Italy': 'ITA', 'Turkey': 'TUR',
                   'Toscani in Italia': 'TSI', 'Utah Residents (CEPH) with N & W European ancestry': 'CEU',
                   'AFRICAN': 'AFR', 'EUROPEAN': 'EUR', 'SOUTH ASIAN': 'SAS', 'EAST ASIAN': 'EAS', 'ADMIXED AMERICAN': 'AMR'}

        true_labels_1 = [mapping[item] for item in true_labels_1]
        predictions_1 = [mapping[item] for item in predictions_1]
        
        vmin, vmax = 0, 1
        plot_confusion_matrix(true_labels_1, predictions_1, "", vmin, vmax)
