#!/bin/bash
#SBATCH --partition=alldlc2_gpu-l40s
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ag_fe_job
#SBATCH --export=ALL,PYTHONPATH:/work/dlclarge2/purucker-tabarena/code/BGA_Classification
#SBATCH --gres=gpu:1,localtmp:100
#SBATCH --propagate=NONE
#SBATCH -o /work/dlclarge2/purucker-tabarena/slurm_out/new_models/%A/slurm-%A_%a.out
#SBATCH -e /work/dlclarge2/purucker-tabarena/slurm_out/new_models/%A/slurm-%A_%a.out

set -e
set -u
set -o pipefail
set -x

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it with 'sudo apt install jq' or 'brew install jq'."
    exit 1
fi
# Read defaults
PYTHON_PATH=/work/dlclarge2/purucker-tabarena/venvs/fe/bin/python
RUNSCRIPT=/work/dlclarge2/purucker-tabarena/code/BGA_Classification/feature_selection_experiments/run_experiment_feature_selection.py
TIME_LIMIT=36000
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}

echo "Python Path: $PYTHON_PATH"
echo "Run Script: $RUNSCRIPT"
echo "Time Limit: $TIME_LIMIT"
echo "Split Index: $SPLIT_INDEX"


$PYTHON_PATH $RUNSCRIPT -s $SPLIT_INDEX -t $TIME_LIMIT