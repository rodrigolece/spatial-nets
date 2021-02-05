#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


DATA_DIR="data"
OUT_DIR="output"

python -u score_grav.py "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_here_dmat.npz"; echo
python -u score_grav.py "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_geodesic_dmat.mat"

