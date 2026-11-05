#!/bin/bash
#SBATCH --partition=u22 --account=research --qos=medium -c 4 --mem=8000 --time=06:00:00
# usage: sbatch -J <name> -o slurm_<name>_%j.out experiments/exp.sbatch experiments/<script>.py [args]
cd "$SLURM_SUBMIT_DIR"
source ../ngspice_bundle/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg DVD_N=10000
../.venv/bin/python "$@"
