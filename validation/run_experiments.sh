#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt2"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


# OUT_DIR="output_expert"

MODEL='gravity'
NB_REPEATS=5
NB_NET_REPEATS=10

# Used as default instead
# N=20
# M=20

python -u experiments_expert.py "$MODEL-production" $NB_REPEATS $NB_NET_REPEATS --directed &> expert_grav_prod.log & 
python -u experiments_expert.py "$MODEL-attraction" $NB_REPEATS $NB_NET_REPEATS --directed &> expert_grav_attrac.log & 
python -u experiments_expert.py "$MODEL-doubly" $NB_REPEATS $NB_NET_REPEATS --directed &> expert_grav_doubly.log & 

