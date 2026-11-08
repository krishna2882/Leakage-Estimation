#!/bin/bash
# Run ON THE ADA GATEWAY inside ~/projects/dvd/project.  Submits the tiers as a dependency chain:
# each tier starts only if the previous one finished OK.  Optional arg: first tier to start from (1-5).
cd "$(dirname "$0")/.."
FROM=${1:-1}
TO=${2:-5}            # optional last tier; DVD_TAG / DVD_N from the environment select a variant run
# tier 6 (main run only): dataset-size study over all finished runs, then the requirement check (exit 1 = a gap remains)
T=( "" "00_setup_check 01_gen_data 02_data_check" "03_train_models" "04_gate_spice 05_gate_ml" "06_circuit_spice 07_circuit_ml" "08_error_analysis 09_report" "11_learning_curve 10_verify_requirements" )
N=( "" t1_data t2_train t3_gates t4_circuit t5_report t6_verify )
# dataset-size runs reuse the main run's circuit SPICE reference (same corners) and need no report of their own
[ -n "$DVD_TAG" ] && T[4]="07_circuit_ml" && T[5]="08_error_analysis"
dep=""
for i in $(seq $FROM $TO); do
  a="--parsable -J ${N[$i]}${DVD_TAG} -c 4 --mem=8000"
  [ -n "$dep" ] && a="$a --dependency=afterok:$dep"
  dep=$(sbatch $a --export=ALL,STAGES="${T[$i]}" job_tier.sbatch)
  echo "${N[$i]} $dep"
done
