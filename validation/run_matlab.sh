#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt-matlab"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


NB_REPEATS=5
NB_NET_REPEATS=10

# Used as default instead
# N=20
# M=20


# 0.5
# 10,20
for BINSIZE in {1,2,5}; do
    python -u modularity.py $NB_REPEATS $NB_NET_REPEATS --binsize $BINSIZE &> "modularity-$BINSIZE.log" & 
done

