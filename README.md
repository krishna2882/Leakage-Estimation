# ML-driven PVT-variation-aware leakage estimation (DVD course project)

Assignment: **32nm_MGK** model file, nominal **VDD = 0.9 V**, test circuit **8-bit parity checker**.

## Run order (fixed seeds; everything regenerates from the model file)

| Script | Tier | Produces |
|---|---|---|
| `00_setup_check.py` | 0 | nominal process values read from the model file, parametrised-model fidelity, simulator version |
| `01_gen_data.py` | 1 | `data/{nmos,pmos}_{on,off}.csv` (10,000 rows each), `netlists/single_*.net` |
| `02_data_check.py` | 1 | Kirchhoff check, trend plots |
| `03_train_models.py` | 2 | 4 model types x 4 cases comparison, chosen models in `saved_models/`, pred-vs-SPICE plots |
| `04_gate_spice.py` | 3 | SPICE reference for 18 gate/input cases x 200 corners, gate netlists |
| `05_gate_ml.py` | 3 | ML gate leakage, gate results table, gate parity plot |
| `06_circuit_spice.py` | 4 | full parity-checker SPICE, 256 vectors x 200 corners |
| `07_circuit_ml.py` | 4 | hierarchical ML estimate, circuit metrics, parity plot |
| `08_error_analysis.py` | 5 | where the errors are and correlations with T / VDD / process |
| `09_report.py` | 5 | `report.md` assembled from the result files |

All heavy work runs on the Ada cluster (`ada/`): `run_all.sh` runs any subset of stages (`STAGES="..."`),
`ada/submit_chain.sh` submits the tiers as a SLURM dependency chain.

## Conventions
- Terminal currents (`id, ig, is, ib`) are currents **into** the device terminal (ngspice `I(V*)` negated).
- Gate / circuit leakage = current delivered by every ideal source held at VDD (supply + inputs at logic 1)
  = static power / VDD. Same definition in SPICE and ML.
- Process parameters (vth0, toxe, ndep, u0, xj) are passed as `.param` into `models/32nm_MGK_param.pm`,
  a copy of `models/32nm_MGK.pm` with those five values replaced by parameters (verified identical at nominal).
- Every netlist uses `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`; single-FET netlists have one source per terminal.
- Sizes: INV N=2Lmin/P=4Lmin, NAND2 N=4Lmin (series)/P=4Lmin, NOR2 N=2Lmin/P=8Lmin (series); AND2/OR2 = NAND2/NOR2 + INV.
- Parity checker: XOR(a,b) = AND2(OR2(a,b), NAND2(a,b)); 7 XORs in a balanced tree = 21 gates, 112 transistors.
