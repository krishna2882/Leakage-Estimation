#!/bin/bash
# Run ON THE ADA GATEWAY inside ~/projects/dvd/project.  Submits the tiers as a dependency chain:
# each tier starts only if the previous one finished OK.  Optional arg: first tier to start from (1-5).
cd "$(dirname "$0")/.."
FROM=${1:-1}
T=( "" "00_setup_check 01_gen_data 02_data_check" "03_train_models" "04_gate_spice 05_gate_ml" "06_circuit_spice 07_circuit_ml" "08_error_analysis 09_report" )
N=( "" t1_data t2_train t3_gates t4_circuit t5_report )
dep=""
for i in $(seq $FROM 5); do
  a="--parsable -J ${N[$i]} -c 4 --mem=8000"
  [ -n "$dep" ] && a="$a --dependency=afterok:$dep"
  dep=$(sbatch $a --export=ALL,STAGES="${T[$i]}" job_tier.sbatch)
  echo "${N[$i]} $dep"
done
