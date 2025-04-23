from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json

with open('results_TabPFN_vcf_random.json', 'r') as file:
    data = json.load(file)

first_key = next(iter(data[0]))
last_key = next(reversed(data[0]))

first_element = (first_key, data[0][first_key])
last_element = (last_key, data[0][last_key])

second_key = list(data[0].keys())[2]
second_element = (second_key, data[0][second_key])

true = []
pred = []
n = 50 # num repeats of the experiments
for i in range(n):
    first_key = next(iter(data[i]))
    last_key = next(reversed(data[i]))
    
    first_element = (first_key, data[i][first_key])
    last_element = (last_key, data[i][last_key])
    
    second_key = list(data[i].keys())[2]
    second_element = (second_key, data[i][second_key])
    true.append(last_element[1])
    pred.append(second_element[1])
flattened_list_pred = [item for sublist in pred for item in sublist]
flattened_list_true = [item for sublist in true for item in sublist]

true_labels_1 = flattened_list_true
predictions_1 = flattened_list_pred

##### for plotting the results from filtered_population_eur #####
"""
mapping = {'09. Russia - Russian': 'RUS', 'British in England and Scotland': 'GBR', 'Finnish in Finland': 'FIN',
           'France': 'FRA', 'Iberian population in Spain': 'IBS', 'Italy': 'ITA', 'Turkey': 'TUR',
           'Utah Residents (CEPH) with N & W European ancestry': 'UtahEU'}
#mapping = {'AFRICAN': 'AFR', 'AMERICAN': 'EUR', 'EAST ASIAN': 'EAS', 'EUROPEAN': 'EUR', 'MIDDLE EAST': 'ME', 'OCEANIAN': 'OCE', 'SOUTH ASIAN':'SAS'}
true_labels_1 = [mapping[item] for item in true_labels_1]
predictions_1 = [mapping[item] for item in predictions_1]
#################################################################
"""

def plot_confusion_matrix(true_labels, predictions, title, vmin, vmax):
    classes = sorted(list(set(true_labels + predictions)))
    
    true_labels_numeric = [classes.index(label) for label in true_labels]
    predictions_numeric = [classes.index(label) for label in predictions]

    cm = confusion_matrix(true_labels_numeric, predictions_numeric, labels=range(len(classes)))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center", color="black")

    ax.set_xlabel('Predicted Population', fontsize=16)
    ax.set_ylabel('True Population', fontsize=16)
    plt.tight_layout()
    plot_path = r'plots\cm_TabPFN_vcf_random.pdf'
    plt.savefig(plot_path)
    plt.show()

vmin, vmax = 0, 1  
plot_confusion_matrix(true_labels_1, predictions_1, "", vmin, vmax)
