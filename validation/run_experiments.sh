#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt2"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


# MODEL='gravity'
MODEL='radiation'
NB_REPEATS=10
NB_NET_REPEATS=10

# Used as default instead
# N=20
# M=20


for CT in {production,attraction,doubly}; do
    # python -u expert_line.py "$MODEL-$CT" $NB_REPEATS $NB_NET_REPEATS &> "$MODEL-$CT.log" & 
    # python -u expert_line.py "$MODEL-$CT" $NB_REPEATS $NB_NET_REPEATS --gamma 1.0 &> "$MODEL-$CT.log" & 
    python -u cerina_line.py "$MODEL-$CT" $NB_REPEATS $NB_NET_REPEATS &> "cerina-$MODEL-$CT.log" & 
    # python -u cerina_line.py "$MODEL-$CT" $NB_REPEATS $NB_NET_REPEATS --epsilon 0.5 &> "cerina-$MODEL-$CT.log" & 
done

