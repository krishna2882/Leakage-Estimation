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
| `11_learning_curve.py` | 6 | dataset-size study: device, gate and circuit accuracy at 2.5k / 5k / 10k / 20k / 40k samples per case |
| `09_report.py` | 6 | `report.md` assembled from the result files (needs the learning curve and the 40k run) |
| `10_verify_requirements.py` | 6 | automated check of every spec requirement, `results/10_requirements_check.txt` |
| `12_report_pdf.py` | local | `report.pdf` rendered from `report.md` (after syncing down: needs Chrome/Chromium and `pip install markdown`) |

All heavy work runs on the Ada cluster (`ada/`): `run_all.sh` runs any subset of stages (`STAGES="..."`),
`ada/submit_chain.sh FROM TO` submits tiers FROM..TO as a SLURM dependency chain (each tier starts only if the
previous one succeeded). One-time environment (ngspice bundle and Python venv next to the project, outside this repo):
`ada/setup_env.md`. Dataset-size runs (`DVD_TAG`, `DVD_N`) write to `data_<tag>/`, `results_<tag>/`, ... and leave the main run
untouched; their tier 4 runs only the circuit estimate, against the main run's circuit SPICE reference (same corners).

**Rebuilding everything, in order** (on the Ada gateway; each line after the main run's tiers 1-5 have finished):
```bash
ada/submit_chain.sh 1 5                                   # main run, the specified 10,000 samples per case
DVD_TAG=_2k5 DVD_N=2500 ada/submit_chain.sh 1 3           # dataset-size study (these four can run at the same time)
DVD_TAG=_5k  DVD_N=5000 ada/submit_chain.sh 1 3
DVD_TAG=_20k DVD_N=20000 ada/submit_chain.sh 1 3
DVD_TAG=_40k DVD_N=40000 ada/submit_chain.sh 1 5          # final models (report headline)
ada/submit_chain.sh 6 6                                   # once all runs above finished: learning curve, report, requirement check
```
then `ada/sync_down.sh` and `python3 12_report_pdf.py` on the laptop.

**Final models / report headline:** the 40,000-samples-per-case run (`common.HEADLINE = "_40k"`: `data_40k/`, `saved_models_40k/`,
`results_40k/`, `plots_40k/`), the first size of the study at which both the gate-level and the parity-checker mean accuracy
reach 99 %. The main run keeps the specified 10,000 samples per case (`data/`, deliverable D1); `report.md` gives both, and
`10_verify_requirements.py` checks the 99 % targets on the headline run. `experiments/` holds the studies behind the
design choices (power-based leakage proof, feature/target choices, rail weighting, pmos_off diagnosis); they are not part of the pipeline.

## Conventions
- Terminal currents (`id, ig, is, ib`) are currents **into** the device terminal (ngspice `I(V*)` negated).
- Gate / circuit leakage = current delivered by every ideal source held at VDD (supply + inputs at logic 1)
  = static power / VDD. Same definition in SPICE and ML.
- Process parameters (vth0, toxe, ndep, u0, xj) are passed as `.param` into `models/32nm_MGK_param.pm`,
  a copy of `models/32nm_MGK.pm` with those five values replaced by parameters (verified identical at nominal).
- Every netlist uses `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`; single-FET netlists have one source per terminal.
- Sizes: INV N=2Lmin/P=4Lmin, NAND2 N=4Lmin (series)/P=4Lmin, NOR2 N=2Lmin/P=8Lmin (series); AND2/OR2 = NAND2/NOR2 + INV.
- Parity checker: XOR(a,b) = AND2(OR2(a,b), NAND2(a,b)); 7 XORs in a balanced tree = 21 gates, 112 transistors.
