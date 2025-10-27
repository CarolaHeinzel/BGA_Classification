# Feature Selection Study for Tabular Machine Learning on BGA Classification


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
1.  `feature_selection_experiments/run_experiment_feature_selection.py`: Calculation of the feature importances for one split (we did 60 splits overall)
2.  `data/output_data/fe_results/agg.py`: Aggregierte feature importance berechnen aus den Ergebnisse aus 1. für alle splits
3.  `feature_selection_experiments/run_experiments_baselines.py`: Auswertung der ausgewählten features mit Crossvalidation für TabPFN und NaiveBayes

### (Local Graphical User Interface)
`streamlit/streamlit.py`: All of the steps above at once in a browser (just need to upload your own data)
Usage: streamlit run streamlit/streamlit.py

---
The repository contains also:
- `feature_selection_experiments/run_experiments_allele_method.py`: 
- `evaluation/run_plotting.py`: code for plotting the mean of ROC AUC, accuracy and logloss for the results from crossvalidation
- `evaluation/confusion_matrix.py`: code for plotting the confusion matrix
