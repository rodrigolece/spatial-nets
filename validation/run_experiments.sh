#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


DATA_DIR="data"
OUT_DIR="output/benchmark_expert"

MODEL='gravity-doubly'
NB_REPEATS=5
NB_NET_REPEATS=10

# Used as default instead
# N=20
# M=20

python -u experiments_expert.py $MODEL $NB_REPEATS $NB_NET_REPEATS; echo
python -u experiments_expert.py $MODEL $NB_REPEATS $NB_NET_REPEATS -B; echo 

