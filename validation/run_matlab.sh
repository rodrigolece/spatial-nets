#! /bin/bash

# source $HOME/.bashrc  # doesn't work for non-interactive shell

BINSIZE=2 

matlab -nodesktop -nosplash -r "run_experiment($BINSIZE); quit" 2>&1 > /dev/null
