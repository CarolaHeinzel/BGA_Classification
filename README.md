# Feature Selection Study for Tabular Machine Learning on BGA Classification

## Overview

This repository contains:

- `feature_selection_experiments/run_experiment_feature_selection.py` - Calculates feature importances for a single data split using Permutation Importance (PI)
- `data/output_data/fe_results/agg.py` - Calculation of the aggregated feature importances across all splits
- `feature_selection_experiments/run_experiments_baselines.py` - Evaluation of the selected features with the PI Method and the Visage Enhage Tool using cross-validation for TabPFN and NaiveBayes
- `feature_selection_experiments/run_experiments_allele_method.py` - Code for feature selection based on the allele frequency method and evaluation of the results
- `data/input_data/full_data.csv` - 
- `data/output_data` - Results from our experiments
- `evaluation/run_plotting.py` - Code for plotting the mean of ROC AUC, accuracy and logloss from cross-validation results
- `evaluation/confusion_matrix.py` - Code for plotting the confusion matrix
- `streamlit/Streamlit.py` - local graphical user interface for the feature selection with PI method and evaluation
  
---

## Usage

### Install
We recommend to use `uv` and Python 3.11 and a Linux OS. The tutorial below already integrates this into the 
installation process.

```bash
#!/bin/bash

# Install uv globally (only needed once)
pip install uv

# Create a new virtual environment with Python 3.11
# You can change the path if you prefer a different location
uv venv --seed --python 3.11 ~/.venvs/feature_selection_BGA

# Activate the environment
source ~/.venvs/feature_selection_BGA/bin/activate

# Ensure uv is available inside the environment
pip install uv

# Install project dependencies from requirements.txt
uv pip install -r requirements.txt

# Install Ruff for linting and formatting
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


## Data
* `1-s2.0-S1872497323000285-mmc5_EUR.csv`: Example test and training data. This is a subset of the data published as a Supplemental (`1-s2.0-S1872497323000285-mmc5.xlsx`) of [Ruiz-Ramirez et al](https://pubmed.ncbi.nlm.nih.gov/36917866/).<br>
* The other data is from the 1000 Genomes Project.


## Funding Acknowledgement

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 499552394 – SFB 1597.
