# Feature Selection Study for Tabular Machine Learning on BGA Classification


## Install
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


## Slurm 

```bash
source /work/dlclarge2/purucker-tabarena/venvs/fe/bin/activate && cd /work/dlclarge2/purucker-tabarena/code/BGA_Classification/feature_selection_experiments
sbatch --array=0-59%100 submit_gpu.sh 
```
