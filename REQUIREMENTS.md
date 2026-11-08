# Requirement traceability (DVD_ML_PROJECT.pdf)

Every row is checked automatically by `10_verify_requirements.py`; the current PASS/FAIL list is in
`results/10_requirements_check.txt`.

| Spec ref | Requirement | Produced by | Evidence |
|---|---|---|---|
| §2 | Assigned model 32nm_MGK, VDD 0.9 V, 8-bit parity checker, L = 32 nm, W = 1-8 Lmin | `common.py` | `results/00_setup.txt` |
| A.1 | 10,000 random samples per case (NMOS/PMOS ON/OFF), every parameter drawn independently, simulated in SPICE | `01_gen_data.py` | `data/*.csv` |
| A.1 | Smallest dataset that keeps accuracy | `11_learning_curve.py` (runs at 2.5k, 5k, 10k, 20k, 40k) | `results/11_learning_curve.csv`, `plots/learning_curve.png` |
| A.2 | 70/15/15 split with fixed seed; compare model types; select on validation; report test | `03_train_models.py` | `results/03_model_metrics_long.csv` |
| B.3 | Every gate x input combination x 200 random corners in SPICE; never used for training | `04_gate_spice.py` | `results/04_gate_spice.csv` |
| B.4 | ML: terminal voltages from the SPICE-recorded internal/output nodes -> transistor models -> gate leakage | `05_gate_ml.py` | `results/05_gate_results_raw.csv` |
| B.5 | Accuracy per gate and input combination | `05_gate_ml.py` | `results/05_gate_table.csv` |
| C.6 | Full transistor-level test circuit, every input vector, same 200 corners | `06_circuit_spice.py` | `results/06_circuit_spice.csv` |
| C.7 | Hierarchical ML: logic value at every gate input -> gate model -> sum | `07_circuit_ml.py` | `results/07_circuit_results_raw.csv` |
| C.8 | Compare hierarchical ML with SPICE | `07_circuit_ml.py` | `results/07_circuit_overall.csv` |
| §4 | VDD drawn first, nominal +-10 %; Vd, Vs in 0..VDD; W 1-8 Lmin; T -40..125 C; vth0, toxe, ndep, u0, xj +-10 %, separate N/P; Vg, Vb fixed per case | `01_gen_data.py` | range checks in `results/10_requirements_check.txt` |
| §4 | `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` in every netlist; own source per terminal in single-FET netlists | `common.py` | `netlists/*.net` |
| §5 | Static CMOS INV/NAND2/NOR2, AND2 = NAND2+INV, OR2 = NOR2+INV, simulated as complete gates; internal nodes x, y; 18 cases x 200 corners; one NMOS draw and one PMOS draw per corner; inputs exactly 0 or VDD | `common.py`, `04_gate_spice.py` | `netlists/<GATE>_<AB>.net`, `results/corners.json` |
| §5 | Record SPICE leakage, ML leakage, internal node voltages from both, SPICE output voltage | `05_gate_ml.py` | `results/05_gate_results_raw.csv`, `results/05_node_voltages.csv` |
| §5 | Test circuit as one netlist, same corners, all inputs; ML = sum of gate leakages | `06_circuit_spice.py`, `07_circuit_ml.py` | `netlists/parity8_example.net` |
| D1 | Four CSVs, 10,000 rows, columns vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib | `01_gen_data.py` | `data/*.csv` |
| D2 | Example netlists: one per transistor case, one per gate, the test circuit | `01`, `04`, `06` | `netlists/` |
| D3 | Kirchhoff check per case; three trend plots | `02_data_check.py` | `results/02_kirchhoff.txt`, `plots/trend_*.png` |
| D4 | Model comparison table: 4 types x 4 cases, R2, median and p95 error on validation | `03_train_models.py` | `results/03_model_comparison_val.csv` |
| D5 | Chosen models: test median and p95 error, predicted-vs-SPICE plot per case, four saved model files | `03_train_models.py` | `results/03_chosen_models_test.csv`, `plots/pred_vs_spice_*.png`, `saved_models/*.joblib` |
| D6 | Gate results table: 18 cases with SPICE median, ML median, bias, median error, p95 error | `05_gate_ml.py` | `results/05_gate_table.csv` |
| D7 | Gate parity plot on log axes | `05_gate_ml.py` | `plots/gate_parity.png` |
| D8 | Test-circuit metrics and parity plot (log axes) | `07_circuit_ml.py` | `results/07_circuit_overall.csv`, `plots/circuit_parity.png` |
| D9 | Scripts that regenerate everything in order, fixed seeds | `run_all.sh`, `ada/submit_chain.sh` | `README.md` |
| D10 | Report: setup, data generation, models, gate results, test-circuit results, error analysis, limitations | `08_error_analysis.py`, `09_report.py`, `12_report_pdf.py` | `report.md`, `report.pdf` |
| §1 | Target: up to 99 % mean accuracy | all | `results/05_gate_overall.csv`, `results/07_circuit_overall.csv` |
