#!/bin/bash
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out
cd "$SLURM_SUBMIT_DIR"
DVD_TAG= DVD_N=10000 ./run_all.sh
