# ML-driven PVT-variation-aware leakage estimation - DVD course project

**Summary - final models (40,000 samples per case).** Gate level (18 gate/input cases x 200 PVT corners): mean accuracy
**99.19 %**, median error 0.53 %, 95th percentile 2.55 %, bias +0.28 %.
8-bit parity checker (256 vectors x 200 corners): mean accuracy **99.38 %**, median error 0.45 %,
95th percentile 1.96 %. Transistor models (test sets, pooled median relative error):
nmos_on 0.07 %, nmos_off 0.08 %, pmos_on 0.07 %, pmos_off 0.39 %.

**With the specified 10,000 samples per case** (deliverable datasets, `data/`): gate level mean accuracy
**98.36 %**, median error 1.09 %, 95th percentile 5.24 %;
8-bit parity checker (256 vectors x 200 corners): mean accuracy **98.59 %**, median error 1.09 %, 95th percentile 4.03 %.
Transistor models (test sets, pooled over Id, Ig, Is, Ib, median relative error): nmos_on 0.24 %, nmos_off 0.24 %, pmos_on 0.19 %, pmos_off 1.06 %.

Sections 2-5 walk through the pipeline on the specified 10,000-sample datasets. Section 6 gives the final models: the same
pipeline (same sampling, features, model selection and the same 200 gate corners and circuit SPICE reference) with 40,000
samples per case. Section 7 shows why: the remaining error is a data (variance) limit, and accuracy grows with the dataset size.

## 1. Setup
- Assigned model file: **32nm_MGK** (PTM 32 nm metal-gate / high-k, BSIM4); nominal VDD **0.9 V** (VDD drawn in 0.81-0.99 V);
  L = Lmin = 32 nm, W = 1-8 Lmin; test circuit: **8-bit parity checker**.
- Simulator: ngspice-36 : Circuit level simulation program The U. C. Berkeley CAD Group, run on the Ada cluster (Ubuntu 22.04). ML: PyTorch (MLP) and scikit-learn (other model types); fixed seed 2024.
- Conventions: terminal currents are currents INTO the terminal (Id, Ig, Is, Ib = -I(V_x) of the source on that terminal).
  Gate leakage = current delivered by every ideal source held at VDD (supply and inputs at logic 1) = static power / VDD;
  the same definition is used for SPICE and ML.
- Transistor netlists: one voltage source per terminal; `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`;
  the five process parameters (vth0, toxe, ndep, u0, xj) are passed with `.param` into a parametrised copy of the model file,
  verified identical to the original at nominal (`results/00_setup.txt`). AD = AS = W*2Lmin, PD = PS = 2(W + 2Lmin).
- Note on toxe: in the model file toxe = toxp + dtox exactly (NMOS 0.75 = 0.50 + 0.25 nm, PMOS 0.77 = 0.50 + 0.27 nm).
  Varying toxe by +-10 % as specified, with toxp and dtox kept from the file, breaks that identity, so ngspice reports
  "toxe != toxp + dtox; dtox ignored" and keeps toxp from the file: the toxe variation acts on the toxe-based terms
  (e.g. gate tunnelling, Coxe) only. The SPICE references and the training data use the same model, so this does not bias the comparison.
- Gates (static CMOS, width unit WU = 2 Lmin): INV N = 1 WU, P = 2 WU; NAND2 N = 2 WU (series), P = 2 WU; NOR2 N = 1 WU,
  P = 4 WU (series); AND2 = NAND2 + INV, OR2 = NOR2 + INV with the same sizes. Internal series nodes: x (NAND2), y (NOR2).
- Parity checker: XOR(a, b) = AND2(OR2(a, b), NAND2(a, b)); a balanced tree of 7 XORs = 21 gates, 112 transistors, one netlist.

## 2. Data generation
- 10000 samples per case (nmos_on: 10000, nmos_off: 10000, pmos_on: 10000, pmos_off: 10000). Every parameter is drawn independently and
  uniformly: VDD first (nominal +-10 %), then Vd, Vs in 0..VDD, T in -40..125 C, W in 1-8 Lmin, vth0, toxe, ndep, u0, xj
  +-10 % around the file's nominal values (separate NMOS and PMOS draws). Vg and Vb are fixed per case: NMOS ON Vg = VDD,
  NMOS OFF Vg = 0, PMOS ON Vg = 0, PMOS OFF Vg = VDD; NMOS body at 0, PMOS body at VDD. Non-converged samples are redrawn.
- Split: 70 / 15 / 15 (train / validation / test), fixed seed (random_state = 2024).
- The gate and parity-checker SPICE results (Parts B and C) are never read by a training or model-selection script (checked in
  `results/10_requirements_check.txt`); model types are chosen on the device validation sets. Two decisions did look at
  gate-level results: the dataset size of the final models (section 6), and the bounded drain-bias feature (section 3), which
  diagnostics at the gate bias points motivated and which was then chosen on an independent single-FET rail-bias set.
- The final models (section 6) use the same procedure with 40,000 samples per case (`data_40k/`); the dataset-size
  study (section 7) repeats it at 2,500 / 5,000 / 20,000 samples per case.

### Data check (Kirchhoff residual and physical trends)
```
Kirchhoff check: KCL = id+ig+is+ib ; rel = |KCL| / (|id|+|ig|+|is|+|ib|)
nmos_on: n=10000  max|KCL|=1.375e-15 A  median|KCL|=7.198e-17 A  max rel=4.031e-05  median rel=6.868e-13  frac(rel<1e-3)=1.0000
nmos_off: n=10000  max|KCL|=5.438e-22 A  median|KCL|=3.423e-24 A  max rel=4.414e-15  median rel=3.297e-16  frac(rel<1e-3)=1.0000
pmos_on: n=10000  max|KCL|=1.057e-16 A  median|KCL|=7.772e-18 A  max rel=1.460e-06  median rel=1.675e-13  frac(rel<1e-3)=1.0000
pmos_off: n=10000  max|KCL|=7.622e-15 A  median|KCL|=3.105e-16 A  max rel=5.845e-05  median rel=2.847e-08  frac(rel<1e-3)=1.0000
nmos_off: Id(T) increases monotonic: True; Id vs |vth0| decreases monotonic: True
pmos_off: Id(T) increases monotonic: True; Id vs |vth0| decreases monotonic: True
nmos_on: |Ig| vs toxe decreases monotonic: True
pmos_on: |Ig| vs toxe decreases monotonic: True
```
Controlled sweeps (one parameter varied, all others nominal):
![OFF current vs temperature](plots/trend_off_current_vs_temperature.png)
![OFF current vs vth0](plots/trend_off_current_vs_vth0.png)
![Gate current vs toxe](plots/trend_gate_current_vs_toxe.png)

## 3. Models
- Inputs: the 12 sample parameters, turned into physics features (same for every model type): thermal voltage vT, terminal-voltage
  differences (PMOS voltages mirrored to NMOS polarity), the same divided by vT, gate overdrive (Vg* - |vth0|)/vT,
  the subthreshold drain-bias factor 1 - exp(-|Vds|/vT), oxide-field terms Vgs/toxe and Vgd/toxe, ln W, ln toxe, ln ndep, ln u0, xj.
- Why the bounded drain-bias factor: inside gates the ON transistors sit at |Vds| ~ 0, beyond the 0.1 % quantile of the uniform training data, where ln(1 - exp(-|Vds|/vT)) runs far outside its training range. On an independent rail-bias validation set (Vd, Vs snapped to 0 / VDD, own seed, single-FET SPICE) the median Ig error at Vds = 0 was 41.2-133.8 % with the logarithm and 3.7-10.1 % with the bounded form, while the ordinary validation median error changed by at most 0.03 percentage points (`experiments/exp7_bounded_vds_feature.py`). Re-weighting the training loss toward the rails instead gave a smaller and less stable gain (`experiments/exp5_rail_weighting.py`).
- Target per terminal: y = asinh(I / I0_t), I0_t = 1 % of the training median |I_t|. This is logarithmic (so the loss tracks the
  relative error) for |I| >> I0_t and stays linear and smooth where a current changes sign (Vds near 0).
- Four model types: polynomial ridge regression (degree 2 and 3, RidgeCV), random forest (500 trees), histogram gradient boosting
  (one per terminal, early stopping), and an MLP (4 hidden layers x 128, SiLU, Adam with cosine learning rate for 2000 epochs,
  then L-BFGS, ensemble of 5 seeds, float64).
- Selection: poly_ridge: degree 2 or degree 3, the better on validation per case; grad_boost, mlp, random_forest: one fixed setting each (`results/03_hyperparams.json`); per case the model type with the smallest
  validation (median + 95th percentile) of the pooled relative error. Relative error = |dI| / max(|I_SPICE|, 1 fA).

### Model comparison (validation set, pooled over Id, Ig, Is, Ib)
| model | nmos_on R2 | nmos_on median % | nmos_on p95 % | nmos_off R2 | nmos_off median % | nmos_off p95 % | pmos_on R2 | pmos_on median % | pmos_on p95 % | pmos_off R2 | pmos_off median % | pmos_off p95 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poly_ridge | 0.991 | 16.48 | 252.9 | 0.9431 | 11.15 | 129.3 | 0.9917 | 13.99 | 189.1 | 0.8921 | 12.08 | 116.7 |
| random_forest | 0.9883 | 19.44 | 179 | 0.9653 | 15.35 | 79.35 | 0.99 | 17.59 | 147.1 | 0.9169 | 14.72 | 70.86 |
| grad_boost | 0.998 | 6.708 | 51.62 | 0.99 | 4.377 | 37.15 | 0.9983 | 6.025 | 42.83 | 0.9319 | 4.285 | 29.47 |
| mlp | 0.9996 | 0.232 | 3.78 | 0.9938 | 0.232 | 2.78 | 0.9999 | 0.182 | 3.04 | 0.9264 | 1.045 | 8.13 |

### Chosen models (test set)
Selection rule: smallest validation (median + p95) pooled error. Per case:
- nmos_on: **mlp**, validation median 0.232 % / p95 3.78 % against the runner-up grad_boost (6.708 % / 51.62 %).
- nmos_off: **mlp**, validation median 0.232 % / p95 2.78 % against the runner-up grad_boost (4.377 % / 37.15 %).
- pmos_on: **mlp**, validation median 0.182 % / p95 3.04 % against the runner-up grad_boost (6.025 % / 42.83 %).
- pmos_off: **mlp**, validation median 1.045 % / p95 8.13 % against the runner-up grad_boost (4.285 % / 29.47 %).

Median / 95th-percentile relative error (%) per terminal; `pooled_|Vds|>5%VDD` excludes the near-zero-Vds samples
where the channel current itself crosses zero:
| case | chosen model | id median % | id p95 % | ig median % | ig p95 % | is median % | is p95 % | ib median % | ib p95 % | pooled median % | pooled p95 % | pooled_\|Vds\|>5%VDD median % | pooled_\|Vds\|>5%VDD p95 % | predicted KCL residual (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmos_on | mlp | 0.253 | 9.79 | 0.155 | 3.19 | 0.266 | 8.35 | 0.27 | 2.27 | 0.236 | 3.97 | 0.213 | 1.6 | 0.0003734 |
| nmos_off | mlp | 0.419 | 3.85 | 0.185 | 0.87 | 0.382 | 3.57 | 0.019 | 2.06 | 0.238 | 2.57 | 0.234 | 2.26 | 0.001133 |
| pmos_on | mlp | 0.193 | 2.6 | 0.147 | 2.93 | 0.204 | 2.52 | 0.238 | 3.78 | 0.192 | 3.11 | 0.173 | 1.59 | 0.0003367 |
| pmos_off | mlp | 1.571 | 12.69 | 0.673 | 3.15 | 1.577 | 14.72 | 0.818 | 5.37 | 1.063 | 8.74 | 1.043 | 7.79 | 0.004653 |

Predicted-vs-SPICE plots (test set, log axes):
![nmos_on](plots/pred_vs_spice_nmos_on.png)
![nmos_off](plots/pred_vs_spice_nmos_off.png)
![pmos_on](plots/pred_vs_spice_pmos_on.png)
![pmos_off](plots/pred_vs_spice_pmos_off.png)

## 4. Gate results (18 cases x 200 corners)
SPICE simulates each complete gate; the ML estimate takes every transistor's terminal voltages from the SPICE-recorded
output and internal nodes, picks the ON or OFF model by the gate voltage, predicts the four terminal currents and forms
the leakage as sum over devices of [(Vd-Vs) Id + (Vg-Vs) Ig + (Vb-Vs) Ib] / VDD. By Tellegen's theorem this equals the
current drawn from the VDD-level sources exactly (checked with 0 V ammeters on every terminal: relative difference <= 2.8e-08, `experiments/exp1_power_formula.py`); it avoids taking
small differences of large ON-device currents.

Metrics in every gate and circuit table, per row over its n simulations: relative error e = (ML - SPICE) / SPICE;
bias = median e; median error = median |e|; p95 error = 95th percentile of |e|; **mean accuracy = 100 % - mean |e|**.

| gate | pattern | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|---|
| INV | 0 | 200 | 8.104e-08 | 8.094e-08 | -0.1841 | 0.924 | 3.013 | 98.86 |
| INV | 1 | 200 | 9.759e-08 | 9.628e-08 | -0.3606 | 1.903 | 8.212 | 97.28 |
| NAND2 | 00 | 200 | 1.117e-07 | 1.127e-07 | -0.01246 | 0.7019 | 2.456 | 99.05 |
| NAND2 | 01 | 200 | 1.952e-07 | 1.943e-07 | -0.1537 | 0.7731 | 2.81 | 98.9 |
| NAND2 | 10 | 200 | 8.999e-08 | 8.913e-08 | 0.1352 | 0.7491 | 2.813 | 99.03 |
| NAND2 | 11 | 200 | 2.647e-07 | 2.638e-07 | -0.3261 | 1.48 | 7.014 | 97.79 |
| NOR2 | 00 | 200 | 2.667e-07 | 2.662e-07 | 0.01088 | 0.9886 | 3.191 | 98.8 |
| NOR2 | 01 | 200 | 2.716e-07 | 2.71e-07 | -0.9104 | 1.783 | 7.893 | 97.34 |
| NOR2 | 10 | 200 | 1.288e-07 | 1.25e-07 | -1.345 | 2.34 | 7.961 | 97.08 |
| NOR2 | 11 | 200 | 9.102e-08 | 9.105e-08 | 0.23 | 1.152 | 5.016 | 98.29 |
| AND2 | 00 | 200 | 2.245e-07 | 2.253e-07 | -0.3567 | 0.8597 | 3.886 | 98.68 |
| AND2 | 01 | 200 | 2.938e-07 | 2.942e-07 | -0.2384 | 1.008 | 4.073 | 98.65 |
| AND2 | 10 | 200 | 1.967e-07 | 1.926e-07 | -0.3345 | 1.117 | 3.873 | 98.57 |
| AND2 | 11 | 200 | 3.583e-07 | 3.589e-07 | -0.33 | 1.109 | 4.778 | 98.33 |
| OR2 | 00 | 200 | 3.735e-07 | 3.728e-07 | -0.3804 | 0.8844 | 3.333 | 98.81 |
| OR2 | 01 | 200 | 3.551e-07 | 3.534e-07 | -0.7573 | 1.495 | 6.469 | 97.86 |
| OR2 | 10 | 200 | 2.15e-07 | 2.105e-07 | -1.006 | 1.516 | 5.093 | 98.09 |
| OR2 | 11 | 200 | 1.835e-07 | 1.828e-07 | -0.1519 | 0.7563 | 2.762 | 99 |

Overall: mean accuracy **98.36 %**, median error 1.09 %, 95th percentile 5.24 %,
bias -0.31 % (n = 3600).
Most accurate case: NAND2 00 (median error 0.702 %); least accurate: NOR2 10
(2.34 %).
![gate parity](plots/gate_parity.png)

**Extra (not required): ML-only node voltages.** Instead of reading x, y and the INV input of AND2/OR2 from SPICE, the
internal node voltages are solved from the ML currents by KCL (grid search plus bisection, Gauss-Seidel over coupled nodes).
With these self-consistent voltages the gate leakage mean accuracy is 97.80 % (median error 1.40 %).
Node-voltage errors vs SPICE:

| gate | pattern | node | spice_median_V | ml_median_V | abs_err_median_mV | abs_err_p95_mV | bracketed_frac |
|---|---|---|---|---|---|---|---|
| NAND2 | 0 | x | 0.07888 | 0.07886 | 0.6601 | 2.822 | 1 |
| NAND2 | 1 | x | 0.0001109 | 0.0006431 | 3.079 | 7.052 | 1 |
| NAND2 | 10 | x | 0.731 | 0.7319 | 1.487 | 4.697 | 1 |
| NAND2 | 11 | x | 0.0003213 | 0.0003876 | 0.08945 | 1.46 | 1 |
| NOR2 | 0 | y | 0.8945 | 0.8944 | 0.3301 | 1.034 | 1 |
| NOR2 | 1 | y | 0.8947 | 0.8928 | 2.335 | 4.749 | 1 |
| NOR2 | 10 | y | 0.1362 | 0.1343 | 1.998 | 4.872 | 1 |
| NOR2 | 11 | y | 0.8244 | 0.8244 | 2.236 | 6.501 | 1 |
| AND2 | 0 | x | 0.07888 | 0.07896 | 0.7197 | 3.122 | 1 |
| AND2 | 0 | n | 0.8953 | 0.8957 | 1.338 | 5.552 | 1 |
| AND2 | 1 | x | 0.0001109 | 0.0006407 | 3.079 | 7.054 | 1 |
| AND2 | 1 | n | 0.8942 | 0.8947 | 1.1 | 4.679 | 1 |
| AND2 | 10 | x | 0.731 | 0.732 | 1.545 | 4.65 | 1 |
| AND2 | 10 | n | 0.8945 | 0.895 | 1.28 | 5.1 | 1 |
| AND2 | 11 | x | 0.0004164 | 0.001035 | 2.016 | 5.2 | 1 |
| AND2 | 11 | n | 0.0007149 | 0.002183 | 5.406 | 12.01 | 1 |
| OR2 | 0 | y | 0.8944 | 0.8935 | 1.671 | 3.527 | 1 |
| OR2 | 0 | n | 0.8937 | 0.8914 | 4.17 | 8.334 | 1 |
| OR2 | 1 | y | 0.8947 | 0.8928 | 2.33 | 4.749 | 1 |
| OR2 | 1 | n | 0.00063 | 0.001938 | 3.236 | 7.898 | 1 |
| OR2 | 10 | y | 0.1363 | 0.1343 | 1.67 | 4.703 | 1 |
| OR2 | 10 | n | 0.0005145 | 0.001759 | 3.201 | 7.8 | 1 |
| OR2 | 11 | y | 0.8244 | 0.8242 | 2.144 | 6.529 | 1 |
| OR2 | 11 | n | 0.0001816 | 0.0009673 | 3.054 | 7.505 | 1 |


## 5. Test-circuit results (8-bit parity checker, 256 vectors x 200 corners)
SPICE simulates the full 112-transistor netlist for every input vector at the same 200 corners (output logic checked in every
run). The hierarchical ML estimate evaluates the logic value at each gate's inputs, looks up that gate's ML leakage at the
corner and sums the 21 gate leakages.

| grp | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|
| parity8_ALL | 51200 | 5.351e-06 | 5.324e-06 | -0.5468 | 1.086 | 4.034 | 98.59 |
| SPICE-gate-sum vs SPICE circuit | 51200 | 5.351e-06 | 5.346e-06 | -0.1561 | 0.1605 | 0.6546 | 99.77 |
| ML with ML-solved node voltages (extra) | 51200 | 5.351e-06 | 5.356e-06 | -0.5224 | 1.263 | 4.278 | 98.4 |

Compared with the gate level: circuit mean accuracy 98.59 % vs 98.36 % for the gates,
median error 1.09 % vs 1.09 %, 95th percentile 4.03 % vs 5.24 %.
Summing 21 gate estimates averages out part of the independent per-gate error, while the common (bias) part adds up.

The second row isolates the error of the hierarchical assumption itself: summing **SPICE** gate leakages (no ML) differs from the
full-circuit SPICE result by only 0.16 % median (99.77 % mean accuracy), so gate-to-gate
loading is small and the circuit-level error is almost entirely the gate-level ML error.

By number of ones in the input vector:
| hw | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|
| 0 | 200 | 5.433e-06 | 5.437e-06 | -0.2595 | 0.8109 | 3.243 | 98.92 |
| 1 | 1600 | 5.501e-06 | 5.504e-06 | -0.4805 | 0.9383 | 3.765 | 98.73 |
| 2 | 5600 | 5.42e-06 | 5.397e-06 | -0.5132 | 0.9983 | 3.803 | 98.69 |
| 3 | 11200 | 5.45e-06 | 5.431e-06 | -0.5935 | 1.076 | 4.021 | 98.58 |
| 4 | 14000 | 5.309e-06 | 5.276e-06 | -0.5085 | 1.068 | 3.973 | 98.61 |
| 5 | 11200 | 5.342e-06 | 5.301e-06 | -0.6178 | 1.169 | 4.187 | 98.51 |
| 6 | 5600 | 5.188e-06 | 5.156e-06 | -0.5065 | 1.119 | 4.121 | 98.57 |
| 7 | 1600 | 5.22e-06 | 5.142e-06 | -0.5472 | 1.131 | 4.051 | 98.57 |
| 8 | 200 | 4.982e-06 | 5.007e-06 | -0.3123 | 0.9146 | 3.506 | 98.78 |
![circuit parity](plots/circuit_parity.png)


## 6. Final models: 40,000 samples per case
The pipeline of sections 2-5 unchanged, run with 40,000 samples per case: datasets `data_40k/` (4 x 40,000 rows, the same
Section 4 sampling and ranges, checked in `results/10_requirements_check.txt`), models `saved_models_40k/`, tables
`results_40k/`, plots `plots_40k/`. The 70/15/15 split, the four model types and the selection rule are the same; the
gate corners and the full-circuit SPICE reference are identical to the main run, so every number here is directly comparable with
sections 4 and 5. Chosen model types: nmos_on mlp, nmos_off mlp, pmos_on mlp, pmos_off mlp.

How this size was chosen: it is the first size of the dataset-size study (section 7) at which both the gate-level and the parity-checker mean accuracy reach 99 % (40,000 samples per case; sizes between 20,000 and 40,000 were not tested). The device validation error also falls with every doubling of the data (section 7). So this choice used the gate and circuit SPICE results (as noted in section 2); the
model types, hyper-parameters and trained weights come from the same procedure as in sections 2-5 and never see them. For that
reason the specified 10,000-sample results are reported alongside.

### Chosen models (test set)
| case | chosen model | id median % | id p95 % | ig median % | ig p95 % | is median % | is p95 % | ib median % | ib p95 % | pooled median % | pooled p95 % | pooled_\|Vds\|>5%VDD median % | pooled_\|Vds\|>5%VDD p95 % | predicted KCL residual (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmos_on | mlp | 0.077 | 1.48 | 0.061 | 0.71 | 0.076 | 1.42 | 0.083 | 0.73 | 0.074 | 0.94 | 0.068 | 0.52 | 5.984e-05 |
| nmos_off | mlp | 0.137 | 0.93 | 0.063 | 0.29 | 0.135 | 0.97 | 0.004 | 0.61 | 0.08 | 0.7 | 0.078 | 0.65 | 0.0004057 |
| pmos_on | mlp | 0.067 | 0.65 | 0.053 | 0.63 | 0.066 | 0.63 | 0.078 | 1.15 | 0.065 | 0.76 | 0.06 | 0.41 | 6.044e-05 |
| pmos_off | mlp | 0.435 | 3.11 | 0.169 | 0.86 | 0.428 | 3.27 | 0.688 | 2.56 | 0.386 | 2.43 | 0.378 | 2.28 | 0.001084 |

### Gate results (18 cases x 200 corners)
| gate | pattern | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|---|
| INV | 0 | 200 | 8.104e-08 | 8.156e-08 | 0.4101 | 0.4484 | 3.094 | 99.16 |
| INV | 1 | 200 | 9.759e-08 | 9.763e-08 | 0.5484 | 0.8224 | 3.129 | 98.82 |
| NAND2 | 00 | 200 | 1.117e-07 | 1.122e-07 | 0.3114 | 0.3225 | 0.9369 | 99.61 |
| NAND2 | 01 | 200 | 1.952e-07 | 1.944e-07 | 0.1179 | 0.4046 | 2.78 | 99.24 |
| NAND2 | 10 | 200 | 8.999e-08 | 9.018e-08 | 0.5143 | 0.515 | 1.267 | 99.42 |
| NAND2 | 11 | 200 | 2.647e-07 | 2.646e-07 | 0.1433 | 0.8028 | 3.604 | 98.92 |
| NOR2 | 00 | 200 | 2.667e-07 | 2.667e-07 | 0.3738 | 0.4625 | 2.597 | 99.16 |
| NOR2 | 01 | 200 | 2.716e-07 | 2.708e-07 | 0.2159 | 0.5832 | 2.99 | 99.08 |
| NOR2 | 10 | 200 | 1.288e-07 | 1.293e-07 | -0.001327 | 0.6919 | 2.638 | 99.06 |
| NOR2 | 11 | 200 | 9.102e-08 | 9.057e-08 | -0.3483 | 0.6762 | 2.56 | 99.06 |
| AND2 | 00 | 200 | 2.245e-07 | 2.248e-07 | 0.2887 | 0.4828 | 1.757 | 99.35 |
| AND2 | 01 | 200 | 2.938e-07 | 2.931e-07 | 0.1707 | 0.5441 | 2.611 | 99.17 |
| AND2 | 10 | 200 | 1.967e-07 | 1.962e-07 | 0.4168 | 0.5247 | 2.043 | 99.27 |
| AND2 | 11 | 200 | 3.583e-07 | 3.58e-07 | 0.2378 | 0.6692 | 2.889 | 99.08 |
| OR2 | 00 | 200 | 3.735e-07 | 3.75e-07 | 0.3331 | 0.4351 | 2.467 | 99.22 |
| OR2 | 01 | 200 | 3.551e-07 | 3.572e-07 | 0.2257 | 0.5301 | 2.58 | 99.18 |
| OR2 | 10 | 200 | 2.15e-07 | 2.146e-07 | 0.2317 | 0.4631 | 2.155 | 99.27 |
| OR2 | 11 | 200 | 1.835e-07 | 1.824e-07 | 0.05977 | 0.437 | 1.881 | 99.36 |

Overall: mean accuracy **99.19 %**, median error 0.53 %, 95th percentile 2.55 %,
bias +0.28 % (n = 3600); with the specified 10,000 samples per case: 98.36 %, 1.09 %,
5.24 %. Most accurate case: NAND2 00 (median error 0.323 %); least accurate:
INV 1 (0.822 %).
With ML-solved internal node voltages (extra, as in section 4): 99.13 % mean accuracy.
![gate parity, 40,000 samples](plots_40k/gate_parity.png)

### Test-circuit results (8-bit parity checker, 256 vectors x 200 corners)
| grp | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|
| parity8_ALL | 51200 | 5.351e-06 | 5.34e-06 | 0.09936 | 0.4468 | 1.961 | 99.38 |
| SPICE-gate-sum vs SPICE circuit | 51200 | 5.351e-06 | 5.346e-06 | -0.1561 | 0.1605 | 0.6546 | 99.77 |
| ML with ML-solved node voltages (extra) | 51200 | 5.351e-06 | 5.344e-06 | 0.1084 | 0.4922 | 2.052 | 99.34 |

Mean accuracy **99.38 %** (specified 10,000 samples per case: 98.59 %). By number of ones in the input vector:
| hw | n | spice_median | ml_median | bias_pct | median_err_pct | p95_err_pct | mean_accuracy_pct |
|---|---|---|---|---|---|---|---|
| 0 | 200 | 5.433e-06 | 5.432e-06 | 0.2576 | 0.4434 | 1.887 | 99.35 |
| 1 | 1600 | 5.501e-06 | 5.515e-06 | 0.1481 | 0.4141 | 1.846 | 99.39 |
| 2 | 5600 | 5.42e-06 | 5.414e-06 | 0.1408 | 0.4326 | 1.907 | 99.39 |
| 3 | 11200 | 5.45e-06 | 5.425e-06 | 0.07955 | 0.4471 | 1.931 | 99.39 |
| 4 | 14000 | 5.309e-06 | 5.3e-06 | 0.1119 | 0.4421 | 1.982 | 99.38 |
| 5 | 11200 | 5.342e-06 | 5.317e-06 | 0.06805 | 0.4568 | 1.971 | 99.38 |
| 6 | 5600 | 5.188e-06 | 5.176e-06 | 0.09773 | 0.4572 | 2.001 | 99.37 |
| 7 | 1600 | 5.22e-06 | 5.209e-06 | 0.1144 | 0.4537 | 2.046 | 99.37 |
| 8 | 200 | 4.982e-06 | 4.954e-06 | 0.2548 | 0.444 | 2.122 | 99.33 |
![circuit parity, 40,000 samples](plots_40k/circuit_parity.png)

### Remaining error by device case
Same attribution as in section 7 (one device case replaced by exact single-FET SPICE currents at the gate bias points):

| exact_case | median_err_pct | p95_err_pct |
|---|---|---|
| nmos_on | 0.4406 | 2.373 |
| nmos_off | 0.42 | 2.122 |
| pmos_on | 0.433 | 2.434 |
| pmos_off | 0.3128 | 1.743 |

With ML everywhere: gate median error 0.53 %.


## 7. Error analysis
The analysis below is for the specified 10,000-sample run (sections 2-5); the dataset-size study covers all runs.

### Which device model limits the gate accuracy
Every transistor bias point that occurs inside the 18 gates x 200 corners was re-simulated as a single FET in SPICE. Replacing the
ML currents of one device case by the exact SPICE currents (all other cases stay ML) gives the remaining gate-level error:

| exact_case | median_err_pct | p95_err_pct |
|---|---|---|
| nmos_on | 1.003 | 5.198 |
| nmos_off | 0.8841 | 5.045 |
| pmos_on | 0.9686 | 5.301 |
| pmos_off | 0.4941 | 2.118 |

With ML everywhere the same computation gives median 1.09 %. The case whose replacement removes the most error is
the one that limits the gate accuracy. Inside gates, Vd and Vs sit at the rails (0 or VDD), i.e. at the corners of the uniform
Vd-Vs training square, where the training data is thinnest; device errors at these bias points are larger than on the random test set.

### Why pmos_off is the weakest device model
Inside the gates every OFF PMOS has Vs = Vg = Vb = VDD, so only (Vd - Vs) Id enters the leakage; that corner (Vs > 0.95 VDD) holds
about 5 % of the uniform training data (`experiments/exp8_pmos_off_diag.py`). Variants trained on the same 10k data and split, scored
on validation and on the independent rail-bias set (`corner` = Vs > 0.95 VDD; the gate test corners are not used); errors in %:

| variant | train_med | val_med | val_power_med | val_corner_power_med | rail_corner_power_med |
|---|---|---|---|---|---|
| base (exp9) | 0.149 | 1.045 | 0.767 | 1.981 | 3.375 |
| wide (exp9) | 0.006 | 1.222 | 0.917 | 2.230 | 3.615 |
| deep (exp9) | 0.070 | 1.135 | 0.893 | 1.494 | 2.420 |
| long (exp9) | 0.037 | 0.903 | 0.676 | 1.988 | 3.111 |
| wd (exp12) | 0.152 | 1.032 | 0.780 | 2.413 | 3.466 |
| ens15 (exp12) | 0.095 | 0.758 | 0.524 | 1.794 | 2.852 |
| small (exp12) | 0.758 | 1.181 | 0.882 | 2.111 | 3.827 |
| tunnel (exp12) | 0.132 | 1.089 | 0.824 | 1.793 | 4.191 |

- The training error is 7-193x below the validation error (except for the deliberately
  smaller network): the model fits its training set far better than new data, i.e. it is variance (data) limited.
- Against the base network (validation median 1.045 %, rail-corner power median 3.375 %), lower validation error:
  ens15 (exp12) 0.758 %, long (exp9) 0.903 %, wd (exp12) 1.032 %; lower rail-corner error: deep (exp9) 2.420 %, ens15 (exp12) 2.852 %, long (exp9) 3.111 %. The best variant on validation
  (ens15 (exp12)) lowers the validation error by 27 % and the rail-corner error by
  15 %; doubling the data to 20,000 samples per case
  with the base settings lowers the validation error by 45 %
  (learning curve below).
- The SPICE data itself is smooth: re-simulating a training row reproduces it to
  2e-16 relative, and narrow one-parameter
  sweeps deviate from a smooth polynomial by at most 0.0039 % in device power
  (`experiments/exp10_spice_smoothness.py`).
- Learning BSIM4's gate-overlap tunnelling currents (Igd, Igs) as separate targets and rebuilding the terminal currents was worse
  for every case (validation median, terminal vs separated targets: nmos_on 0.23 / 0.26 %, nmos_off 0.23 / 0.30 %, pmos_on 0.18 / 0.24 %, pmos_off 1.04 / 1.92 %):
  in OFF devices Igd is practically the whole drain current (median |Igd / Id| = 1.0000 in nmos_off, median |Igd / Id| = 1.0000 in pmos_off), so the split adds nothing (`experiments/exp11_component_targets.py`).
- The ens15 (exp12) variant costs 3.8x the training time of the base.

### Full analysis output
```
== Device level (test sets): relative error % by region of |Vd-Vs|/VDD (pooled over id,ig,is,ib)
nmos_on   |Vds|/VDD in [0.00,0.05): n= 144 median    1.21%  p95   192.59%  (Id,Is only: median 2.92%, p95 830.37%)
nmos_on   |Vds|/VDD in [0.05,0.30): n= 609 median    0.28%  p95     3.84%  (Id,Is only: median 0.31%, p95 4.31%)
nmos_on   |Vds|/VDD in [0.30,1.01): n= 747 median    0.17%  p95     0.80%  (Id,Is only: median 0.19%, p95 0.70%)
nmos_on   Spearman(median err over id,ig,is ; temp) = -0.051
nmos_on   Spearman(median err over id,ig,is ; |vth0|) = -0.004
nmos_off  |Vds|/VDD in [0.00,0.05): n= 158 median    0.27%  p95    13.92%  (Id,Is only: median 0.47%, p95 64.47%)
nmos_off  |Vds|/VDD in [0.05,0.30): n= 603 median    0.20%  p95     1.66%  (Id,Is only: median 0.33%, p95 2.21%)
nmos_off  |Vds|/VDD in [0.30,1.01): n= 739 median    0.27%  p95     2.74%  (Id,Is only: median 0.47%, p95 4.15%)
nmos_off  Spearman(median err over id,ig,is ; temp) = +0.132
nmos_off  Spearman(median err over id,ig,is ; |vth0|) = -0.048
pmos_on   |Vds|/VDD in [0.00,0.05): n= 148 median    0.82%  p95    88.84%  (Id,Is only: median 1.18%, p95 169.77%)
pmos_on   |Vds|/VDD in [0.05,0.30): n= 623 median    0.24%  p95     4.10%  (Id,Is only: median 0.25%, p95 2.26%)
pmos_on   |Vds|/VDD in [0.30,1.01): n= 729 median    0.14%  p95     0.64%  (Id,Is only: median 0.14%, p95 0.57%)
pmos_on   Spearman(median err over id,ig,is ; temp) = -0.081
pmos_on   Spearman(median err over id,ig,is ; |vth0|) = -0.008
pmos_off  |Vds|/VDD in [0.00,0.05): n= 144 median    1.28%  p95    37.51%  (Id,Is only: median 1.80%, p95 76.30%)
pmos_off  |Vds|/VDD in [0.05,0.30): n= 612 median    0.95%  p95     6.50%  (Id,Is only: median 1.34%, p95 9.36%)
pmos_off  |Vds|/VDD in [0.30,1.01): n= 744 median    1.13%  p95     9.11%  (Id,Is only: median 1.83%, p95 14.36%)
pmos_off  Spearman(median err over id,ig,is ; temp) = -0.242
pmos_off  Spearman(median err over id,ig,is ; |vth0|) = +0.020

== Gate level: Spearman(|err|, parameter)
  temp       rho = -0.045   (signed err: -0.026)
  vdd        rho = +0.175   (signed err: +0.079)
  vth0_n     rho = -0.009   (signed err: -0.084)
  vth0_p_abs rho = -0.071   (signed err: +0.086)
  toxe_n     rho = +0.148   (signed err: -0.174)
  toxe_p     rho = +0.065   (signed err: -0.227)
  u0_n       rho = -0.014   (signed err: -0.010)
  u0_p       rho = -0.016   (signed err: +0.078)
  ndep_n     rho = +0.018   (signed err: -0.001)
  ndep_p     rho = -0.052   (signed err: +0.017)
  xj_n       rho = -0.005   (signed err: +0.025)
  xj_p       rho = -0.008   (signed err: +0.064)
== Gate level: |err| % by temperature quintile
                   median    p95
tbin                            
(-37.57, -1.236]    1.202  5.016
(-1.236, 22.555]    0.942  5.057
(22.555, 54.076]    1.222  5.808
(54.076, 90.532]    1.102  5.596
(90.532, 124.827]   0.973  4.200
== Gate level: worst 8 rows
      gate pattern  corner        temp       vdd         spice            ml        err
1571  NOR2      01     171   70.704368  0.818614  2.319354e-07  2.016213e-07 -13.070063
1501  NOR2      01     101  114.474119  0.974839  4.457799e-07  5.012883e-07  12.451972
1818  NOR2      11      18   92.393490  0.988256  9.740689e-08  1.095070e-07  12.422255
204    INV       1       4   42.034284  0.986133  6.670521e-08  5.843664e-08 -12.395702
1901  NOR2      11     101  114.474119  0.974839  9.933387e-08  1.115027e-07  12.250446
1734  NOR2      10     134  -36.187723  0.846844  6.480399e-08  7.261544e-08  12.053959
1771  NOR2      10     171   70.704368  0.818614  1.578956e-07  1.394072e-07 -11.709256
341    INV       1     141   64.442890  0.947318  7.507192e-08  6.651494e-08 -11.398374

== Circuit level: Spearman(|err|, parameter)
  temp       rho = +0.053   (signed err: -0.201)
  vdd        rho = +0.178   (signed err: +0.043)
  vth0_n     rho = +0.005   (signed err: -0.108)
  vth0_p_abs rho = -0.149   (signed err: +0.106)
  toxe_n     rho = +0.233   (signed err: -0.313)
  toxe_p     rho = +0.221   (signed err: -0.386)
  u0_n       rho = +0.003   (signed err: -0.005)
  u0_p       rho = -0.010   (signed err: +0.076)
  ndep_n     rho = -0.042   (signed err: -0.007)
  ndep_p     rho = -0.030   (signed err: +0.068)
  xj_n       rho = +0.002   (signed err: +0.051)
  xj_p       rho = -0.058   (signed err: +0.095)
== Circuit level: |err| % by temperature quintile
                   median    p95
tbin                            
(-37.57, -1.236]    0.906  2.875
(-1.236, 22.555]    0.875  4.043
(22.555, 54.076]    1.526  4.967
(54.076, 90.532]    1.386  5.130
(90.532, 124.827]   0.838  2.939
== Circuit level: worst 8 rows
       vec  corner       temp       vdd     spice        ml       err
34804  174       4  42.034284  0.986133  0.000003  0.000002 -8.202014
22004  110       4  42.034284  0.986133  0.000003  0.000003 -8.193822
34604  173       4  42.034284  0.986133  0.000003  0.000003 -8.193822
31604  158       4  42.034284  0.986133  0.000003  0.000003 -8.193821
34204  171       4  42.034284  0.986133  0.000003  0.000002 -8.191647
46804  234       4  42.034284  0.986133  0.000003  0.000002 -8.191647
21804  109       4  42.034284  0.986133  0.000003  0.000003 -8.186321
18804   94       4  42.034284  0.986133  0.000003  0.000003 -8.186321

== Device models at the bias points used inside the gates (SPICE single-FET reruns)
nmos_off  n= 4200  power-term error: median 1.335%  p95 5.35%  bias -0.195%   | Ig error median 0.701%
nmos_on   n= 4200  power-term error: median 1.110%  p95 7.14%  bias +0.020%   | Ig error median 1.050%
pmos_off  n= 4200  power-term error: median 3.203%  p95 14.19%  bias -0.838%   | Ig error median 2.008%
pmos_on   n= 4200  power-term error: median 0.971%  p95 3.89%  bias +0.192%   | Ig error median 1.004%

== Attribution (gate level): ML everywhere -> median |err| 1.089%, p95 5.24%
   with exact SPICE currents for nmos_on  : median |err| 1.003%, p95 5.20%
   with exact SPICE currents for nmos_off : median |err| 0.884%, p95 5.04%
   with exact SPICE currents for pmos_on  : median |err| 0.969%, p95 5.30%
   with exact SPICE currents for pmos_off : median |err| 0.494%, p95 2.12%

```
![error vs temperature](plots/error_vs_temp.png)

### Dataset size (learning curve)
Each row is an independent full run (new data, retraining, the same 200 gate corners). The parity-checker columns are the
hierarchical sum of that run's gate estimates against the same full-circuit SPICE reference (the SPICE side does not depend on
the models):

| samples_per_case | gate_mean_accuracy_pct | gate_median_err_pct | gate_p95_err_pct | circuit_mean_accuracy_pct | circuit_median_err_pct | nmos_on_test_median_pct | nmos_off_test_median_pct | pmos_on_test_median_pct | pmos_off_test_median_pct |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 95.892 | 3.093 | 11.337 | 96.781 | 2.500 | 0.947 | 0.746 | 0.772 | 1.720 |
| 5000 | 97.175 | 1.885 | 8.694 | 97.486 | 1.816 | 0.492 | 0.425 | 0.428 | 1.388 |
| 10000 | 98.355 | 1.089 | 5.244 | 98.589 | 1.086 | 0.236 | 0.238 | 0.192 | 1.063 |
| 20000 | 98.850 | 0.793 | 3.394 | 99.008 | 0.712 | 0.120 | 0.128 | 0.097 | 0.572 |
| 40000 | 99.191 | 0.528 | 2.550 | 99.380 | 0.447 | 0.074 | 0.080 | 0.065 | 0.386 |

Factor by which the device test median error changes per doubling of the data: nmos_on 0.48-0.62; nmos_off 0.54-0.63; pmos_on 0.45-0.67; pmos_off 0.54-0.81. Every device error falls with every doubling and the gate and circuit errors follow, so the remaining error is mainly a data (variance) limit, not a model-form limit. Of the sizes tested (2,500, 5,000, 10,000, 20,000, 40,000 samples per case), the smallest that reaches 99 % mean accuracy is 40,000 at gate level and 20,000 for the parity checker.

Best device validation median error (%) per size:

| samples_per_case | nmos_on_val_median_pct | nmos_off_val_median_pct | pmos_on_val_median_pct | pmos_off_val_median_pct |
|---|---|---|---|---|
| 2500 | 1.024 | 0.801 | 0.762 | 2.031 |
| 5000 | 0.492 | 0.426 | 0.437 | 1.341 |
| 10000 | 0.232 | 0.232 | 0.182 | 1.045 |
| 20000 | 0.123 | 0.130 | 0.099 | 0.571 |
| 40000 | 0.076 | 0.081 | 0.065 | 0.382 |

![learning curve](plots/learning_curve.png)


## 8. Limitations and improvements
- The device models are trained on Vg fixed at exactly 0 or VDD; inside AND2/OR2 the output INV sees a near-rail (not exact)
  input voltage, a small extrapolation.
- Relative device error is largest where the true current is tiny or crosses zero (|Vds| near 0); such points rarely carry
  gate leakage, but they dominate the device-level 95th percentile.
- The device bias points inside gates lie at the rails, where uniform Vd/Vs sampling places few samples; sampling more densely
  near the rails (in addition to the uniform data, which the specification requires) would target the gate-level error directly.
- At the specified 10,000 samples per case the pmos_off model is variance limited; the dataset-size study shows the 99 % level is
  reached with more samples per case, at a proportionally longer data-generation and training time.
- The hierarchical circuit estimate ignores gate-to-gate loading by construction; the SPICE gate-sum row in section 5 measures that
  part separately.
- The pmos_off model has the largest pooled test error (1.06 % median); making the pmos_off currents exact lowers the gate median error from 1.09 % to 0.49 %, the largest single gain.
