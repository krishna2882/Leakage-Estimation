#!/bin/bash
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out
cd "$SLURM_SUBMIT_DIR"
# run from a private copy so that syncing code while the job runs cannot break it (stale NFS handle)
cp run_all.sh ".run_all_${SLURM_JOB_ID}.sh"
DVD_TAG=${DVD_TAG-} DVD_N=${DVD_N:-10000} bash ".run_all_${SLURM_JOB_ID}.sh"; rc=$?
rm -f ".run_all_${SLURM_JOB_ID}.sh"; exit $rc
