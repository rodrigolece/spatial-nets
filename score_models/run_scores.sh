#! /bin/bash


# Activate conda environment; raises unbound error
CONDA_BASE="$(conda info --base)"
CONDA_ENV="gt2"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate  "$CONDA_ENV"

# After conda trick set errors 
set -euo pipefail


DATA_DIR="../data"
FLOW_FILE="$DATA_DIR/US_air_traffic2006.npz"
DMAT_FILE="$DATA_DIR/US_airports2006_dmat.npz"
# OUT_DIR="output_usair"  selected inside the python scripts

{ time python -u score_grav.py -e $FLOW_FILE $DMAT_FILE -o 'pvalues_grav.npz' --latex  > usair_grav.log & } 2> time_grav.txt
{ time  python -u score_rad.py -e $FLOW_FILE $DMAT_FILE -o 'pvalues_rad.npz' --latex > usair_rad.log & } 2> time_rad.txt

