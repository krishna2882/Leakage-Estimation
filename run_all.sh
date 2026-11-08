#!/bin/bash
# Runs every tier in order on an Ada compute node.  env: DVD_TAG, DVD_N, DVD_NCORNERS, DVD_NVEC
cd "$(dirname "$0")"
source ../ngspice_bundle/env.sh
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p logs
ALL="00_setup_check 01_gen_data 02_data_check 03_train_models 04_gate_spice 05_gate_ml 06_circuit_spice 07_circuit_ml 08_error_analysis 09_report 10_verify_requirements"
for s in ${STAGES:-$ALL}; do
  echo "=== $s $(date)"
  $PY $s.py > logs/${s}${DVD_TAG}.log 2>&1 || { echo "FAILED $s"; tail -25 logs/${s}${DVD_TAG}.log; exit 1; }
  grep -v -iE "warn|Convergence" logs/${s}${DVD_TAG}.log | tail -12
done
echo "=== ALL DONE $(date)"
