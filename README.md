# Feature Selection Study for Tabular Machine Learning on BGA Classification

## Overview
This repository contains:
- `feature_selection_experiments/run_experiment_feature_selection.py` -Code for the calculation of the feature importances for one split using Permutation Importance
- `data/output_data/fe_results/agg.py` -Calculation of the aggregated feature importances over all splits
- `feature_selection_experiments/run_experiments_baselines.py` -Evaluation of the feature selection with the PI Method and the Visage Enhage Tool using Cross-validation for TabPFN and NaiveBayes
- `feature_selection_experiments/run_experiments_allele_method.py` -Code for feature selection with allele frequency method and evaluation
- `data/input_dataf/ull_data.csv` 
- `evaluation/run_plotting.py` -code for plotting the mean of ROC AUC, accuracy and logloss for the results from crossvalidation
- `evaluation/confusion_matrix.py` -code for plotting the confusion matrix
- `streamlit/Streamlit.py` -Local Graphical User Interface for the feature selection with PI Method and Evaluation
  
---

## Usage

### Install
We recommend to use `uv` and Python 3.11 and a Linux OS. The tutorial below already integrates this into the 
installation process.

```bash
pip install uv
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate
pip install uv

# Install AutoGluon (comes with TabPFN)
uv pip install autogluon["tabarena"] 
# For linting and formatting
uv pip install ruff
```


### Slurm 

```bash
source /work/dlclarge2/purucker-tabarena/venvs/fe/bin/activate && cd /work/dlclarge2/purucker-tabarena/code/BGA_Classification/feature_selection_experiments
sbatch --array=0-59%100 submit_gpu.sh 
```


### Run the experiments (Command Line)
1.  `feature_selection_experiments/run_experiment_feature_selection.py`: Calculation of feature importances for one split (we did 60 splits overall)
2.  `data/output_data/fe_results/agg.py`: Calculate the aggregated feature importance from the results of 1.
3.  `feature_selection_experiments/run_experiments_baselines.py`: Evaluation of the selected features with cross-validation

### Run the experiments (Local Graphical User Interface)
```streamlit run streamlit/Streamlit.py```
