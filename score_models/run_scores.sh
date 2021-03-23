#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


DATA_DIR="../data"
# OUT_DIR="output"

# python -u score_grav.py -e "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_geodesic_dmat.mat" -o 'pvalues_grav.npz'
# python -u score_rad.py -e "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_geodesic_dmat.mat" -o 'pvalues_rad_geodesic.npz'

# python -u score_grav.py -e "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_here_dmat.npz" -o 'pvalues_grav_here.npz'
# python -u score_rad.py -e "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_here_dmat.npz" -o 'pvalues_rad_here.npz'
python -u score_grav.py "$DATA_DIR/UK_commute2011.npz" "$DATA_DIR/UK_geodesic_dmat.mat"  #--latex
