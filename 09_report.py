"""Tier 5b: assemble report.md (all numbers read from result files; optional studies are included when present)."""
import json
import re
import pandas as pd
from common import *


def md(df, fmt="{:.4g}"):
    cols = list(df.columns)
    # a "|" inside a column name (e.g. pooled_|Vds|>5%VDD) would split the cell
    out = ["| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):        # keeps integer columns integer (iterrows would upcast to float)
        out.append("| " + " | ".join(fmt.format(x) if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


def res(name):
    p = os.path.join(RES, name)
    return p if os.path.exists(p) else None


def csv(name, **kw):
    p = res(name)
    return pd.read_csv(p, **kw) if p else None


ver = open(os.path.join(RES, "00_setup.txt")).read().split("\n")[0].removeprefix("simulator: ")
lm = pd.read_csv(os.path.join(RES, "03_model_metrics_long.csv"))
ch = json.load(open(os.path.join(SAVED, "chosen.json")))
n_rows = {c: len(pd.read_csv(os.path.join(DATA, f"{c}.csv"))) for c in CASES}
cmp_val = pd.read_csv(os.path.join(RES, "03_model_comparison_val.csv"))
chosen_test = pd.read_csv(os.path.join(RES, "03_chosen_models_test.csv"))
gt = pd.read_csv(os.path.join(RES, "05_gate_table.csv"), dtype={"pattern": str})
go = pd.read_csv(os.path.join(RES, "05_gate_overall.csv")).iloc[0]
co = pd.read_csv(os.path.join(RES, "07_circuit_overall.csv"))
chw = pd.read_csv(os.path.join(RES, "07_circuit_by_hamming_weight.csv"))
kir = open(os.path.join(RES, "02_kirchhoff.txt")).read(); trend = open(os.path.join(RES, "02_trends.txt")).read()
ea = open(os.path.join(RES, "08_error_analysis.txt")).read()
sc = csv("05_gate_overall_selfconsistent.csv"); nv = csv("05_node_voltages.csv")
att = csv("08_error_attribution.csv"); lc = pd.read_csv(os.path.join(ROOT, "results", "11_learning_curve.csv")) \
    if os.path.exists(os.path.join(ROOT, "results", "11_learning_curve.csv")) else None
best = gt.loc[gt.median_err_pct.idxmin()]; worst = gt.loc[gt.median_err_pct.idxmax()]
c_ml = co.iloc[0]; c_ref = co.iloc[1] if len(co) > 1 else None
pooled = chosen_test.set_index("case")
dev_med = ", ".join(f"{c} {pooled.loc[c, 'pooled median %']:.2f} %" for c in CASES)
R = []                                      # reason for each final choice: validation score of the winner vs the runner-up
for c in CASES:
    s = cmp_val.set_index("model"); sc_ = (s[f"{c} median %"] + s[f"{c} p95 %"]).sort_values()
    w, r2 = sc_.index[0], sc_.index[1]
    R.append(f"- {c}: **{ch[c]}**, validation median {s.loc[w, f'{c} median %']:.3f} % / p95 {s.loc[w, f'{c} p95 %']:.2f} % "
             f"against the runner-up {r2} ({s.loc[r2, f'{c} median %']:.3f} % / {s.loc[r2, f'{c} p95 %']:.2f} %).")
reasons = "Selection rule: smallest validation (median + p95) pooled error. Per case:\n" + "\n".join(R)
# which model types actually had more than one hyper-parameter setting to choose from (03_hyperparams.json)
hpj = json.load(open(os.path.join(RES, "03_hyperparams.json")))
nvar = {k.split("'")[3]: len(v) for k, v in hpj.items() if not k.endswith("'best')")}
tuned = sorted(t for t, n in nvar.items() if n > 1); fixed = sorted(t for t, n in nvar.items() if n == 1)
hp_txt = lambda t: " or ".join(sorted({re.sub(r"[{}']", "", h["hp"]).replace(":", "") for k, v in hpj.items()
                                       if not k.endswith("'best')") and f"'{t}'" in k for h in v}))
tuned_note = "; ".join([f"{t}: {hp_txt(t)}, the better on validation per case" for t in tuned] +
                       [f"{', '.join(fixed)}: one fixed setting each (`results/03_hyperparams.json`)"])
e7 = [os.path.join(ROOT, "experiments", "out", f"exp7_{c}.csv") for c in CASES]
feat_note = ""
if all(os.path.exists(f) for f in e7):
    x = pd.concat(pd.read_csv(f) for f in e7).set_index(["case", "feature"])
    rng_ = lambda f, k: f"{x.xs(f, level='feature')[k].min():.1f}-{x.xs(f, level='feature')[k].max():.1f} %"
    dv = (x.xs("bounded", level="feature").val_med - x.xs("log", level="feature").val_med).abs().max()
    feat_note = (f"- Why the bounded drain-bias factor: inside gates the ON transistors sit at |Vds| ~ 0, beyond the 0.1 % quantile of the "
                 f"uniform training data, where ln(1 - exp(-|Vds|/vT)) runs far outside its training range. On an independent rail-bias "
                 f"validation set (Vd, Vs snapped to 0 / VDD, own seed, single-FET SPICE) the median Ig error at Vds = 0 was "
                 f"{rng_('log', 'rail_ig_med_vds0')} with the logarithm and {rng_('bounded', 'rail_ig_med_vds0')} with the bounded form, "
                 f"while the ordinary validation median error changed by at most {dv:.2f} percentage points "
                 f"(`experiments/exp7_bounded_vds_feature.py`). Re-weighting the "
                 f"training loss toward the rails instead gave a smaller and less stable gain (`experiments/exp5_rail_weighting.py`).")

X = os.path.join(ROOT, "experiments", "out")
xcsv = lambda n: pd.read_csv(os.path.join(X, n)) if os.path.exists(os.path.join(X, n)) else None
pv = [d.assign(variant=f"{d.variant.iloc[0]} ({e})") for d, e in
      [(xcsv(f"exp9_pmos_off_{v}.csv"), "exp9") for v in ("base", "wide", "deep", "long")] +
      [(xcsv(f"exp12_pmos_off_{v}.csv"), "exp12") for v in ("wd", "ens15", "small", "tunnel")] if d is not None]
pv = pd.concat(pv, ignore_index=True) if pv else None
e10 = xcsv("exp10_spice_smoothness.csv")
e11 = [d for d in (xcsv(f"exp11_{c}.csv") for c in CASES) if d is not None]
e11 = pd.concat(e11, ignore_index=True) if e11 else None
# |Igd / Id| in the OFF datasets, from the BSIM4 component currents cached by exp11 (row-aligned with data/)
igd_share = ", ".join(f"median |Igd / Id| = {(xcsv(f'comp_{c}.csv').igd / pd.read_csv(os.path.join(ROOT, 'data', f'{c}.csv')).id).abs().median():.4f} in {c}"
                      for c in ("nmos_off", "pmos_off") if xcsv(f"comp_{c}.csv") is not None)
f1 = os.path.join(X, "exp1_power_formula.txt")
tel = (f"relative difference <= {float(open(f1).read().split('WORST rel diff:')[1].split()[0]):.1e}"
       if os.path.exists(f1) else "see") + ", `experiments/exp1_power_formula.py`"
# best device validation error per dataset-size run: the size of the final models is chosen on these, not on gate results
SIZES = [("_2k5", 2500), ("_5k", 5000), ("", 10000), ("_20k", 20000), ("_40k", 40000)]
sval = {n: pd.read_csv(os.path.join(ROOT, f"results{t}", "03_model_comparison_val.csv")).set_index("model")
        for t, n in SIZES if os.path.exists(os.path.join(ROOT, f"results{t}", "03_model_comparison_val.csv"))}
sval = pd.DataFrame({c: {n: v[f"{c} median %"].min() for n, v in sval.items()} for c in CASES}).sort_index()
HP = lambda f: os.path.join(ROOT, "results" + HEADLINE, f)
hcsv = lambda f, **kw: pd.read_csv(HP(f), **kw) if os.path.exists(HP(f)) else None
h_dev, h_gt, h_go, h_co = hcsv("03_chosen_models_test.csv"), hcsv("05_gate_table.csv", dtype={"pattern": str}), \
    hcsv("05_gate_overall.csv"), hcsv("07_circuit_overall.csv")
h_hw, h_att, h_sc = hcsv("07_circuit_by_hamming_weight.csv"), hcsv("08_error_attribution.csv"), hcsv("05_gate_overall_selfconsistent.csv")
HEAD = h_dev is not None and h_gt is not None and h_go is not None and h_co is not None and not TAG
# the honest basis for the headline size: the first size of the study at which both accuracy targets are met
size_basis = "see the dataset-size study in section 7."
if lc is not None and "circuit_mean_accuracy_pct" in lc:
    both = lc[(lc.gate_mean_accuracy_pct >= 99) & (lc.circuit_mean_accuracy_pct >= 99)].samples_per_case
    if len(both):
        size_basis = (f"it is the first size of the dataset-size study (section 7) at which both the gate-level and the "
                      f"parity-checker mean accuracy reach 99 % ({int(both.min()):,} samples per case")
        below = lc.samples_per_case[lc.samples_per_case < both.min()]
        size_basis += (f"; sizes between {int(below.max()):,} and {int(both.min()):,} were not tested)." if len(below) else ").")
        if int(both.min()) != HEADLINE_N:
            size_basis += f" **Note: this differs from the headline size {HEADLINE_N:,}.**"
    else:
        size_basis = "no size of the study reaches 99 % at both levels."
    size_basis += (" The device validation error also falls with every doubling of the data (section 7)."
                   if sval.apply(lambda s: s.is_monotonic_decreasing).all() else "")
hn = f"{HEADLINE_N:,}"
spec_line = (f"8-bit parity checker (256 vectors x 200 corners): mean accuracy **{c_ml.mean_accuracy_pct:.2f} %**, median error "
             f"{c_ml.median_err_pct:.2f} %, 95th percentile {c_ml.p95_err_pct:.2f} %.\n"
             f"Transistor models (test sets, pooled over Id, Ig, Is, Ib, median relative error): {dev_med}.")
if HEAD:
    hg, hc = h_go.iloc[0], h_co.iloc[0]; hp = h_dev.set_index("case")
    summary = f"""**Summary - final models ({hn} samples per case).** Gate level (18 gate/input cases x 200 PVT corners): mean accuracy
**{hg.mean_accuracy_pct:.2f} %**, median error {hg.median_err_pct:.2f} %, 95th percentile {hg.p95_err_pct:.2f} %, bias {hg.bias_pct:+.2f} %.
8-bit parity checker (256 vectors x 200 corners): mean accuracy **{hc.mean_accuracy_pct:.2f} %**, median error {hc.median_err_pct:.2f} %,
95th percentile {hc.p95_err_pct:.2f} %. Transistor models (test sets, pooled median relative error):
{", ".join(f"{c} {hp.loc[c, 'pooled median %']:.2f} %" for c in CASES)}.

**With the specified 10,000 samples per case** (deliverable datasets, `data/`): gate level mean accuracy
**{go.mean_accuracy_pct:.2f} %**, median error {go.median_err_pct:.2f} %, 95th percentile {go.p95_err_pct:.2f} %;
{spec_line[0].lower() + spec_line[1:]}

Sections 2-5 walk through the pipeline on the specified 10,000-sample datasets. Section 6 gives the final models: the same
pipeline (same sampling, features, model selection and the same 200 gate corners and circuit SPICE reference) with {hn}
samples per case. Section 7 shows why: the remaining error is a data (variance) limit, and accuracy grows with the dataset size.
"""
else:
    summary = f"""**Summary.** Gate level (18 gate/input cases x 200 PVT corners): mean accuracy **{go.mean_accuracy_pct:.2f} %**, median error
{go.median_err_pct:.2f} %, 95th percentile {go.p95_err_pct:.2f} %, bias {go.bias_pct:+.2f} %.
{spec_line}
"""

A = [f"""# ML-driven PVT-variation-aware leakage estimation - DVD course project

{summary}
## 1. Setup
- Assigned model file: **32nm_MGK** (PTM 32 nm metal-gate / high-k, BSIM4); nominal VDD **0.9 V** (VDD drawn in 0.81-0.99 V);
  L = Lmin = 32 nm, W = 1-8 Lmin; test circuit: **8-bit parity checker**.
- Simulator: {ver}, run on the Ada cluster (Ubuntu 22.04). ML: PyTorch (MLP) and scikit-learn (other model types); fixed seed {SEED}.
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
- {n_rows[CASES[0]]} samples per case ({", ".join(f"{c}: {n_rows[c]}" for c in CASES)}). Every parameter is drawn independently and
  uniformly: VDD first (nominal +-10 %), then Vd, Vs in 0..VDD, T in -40..125 C, W in 1-8 Lmin, vth0, toxe, ndep, u0, xj
  +-10 % around the file's nominal values (separate NMOS and PMOS draws). Vg and Vb are fixed per case: NMOS ON Vg = VDD,
  NMOS OFF Vg = 0, PMOS ON Vg = 0, PMOS OFF Vg = VDD; NMOS body at 0, PMOS body at VDD. Non-converged samples are redrawn.
- Split: 70 / 15 / 15 (train / validation / test), fixed seed (random_state = {SEED}).
- The gate and parity-checker SPICE results (Parts B and C) are never read by a training or model-selection script (checked in
  `results/10_requirements_check.txt`); model types are chosen on the device validation sets. Two decisions did look at
  gate-level results: the dataset size of the final models (section 6), and the bounded drain-bias feature (section 3), which
  diagnostics at the gate bias points motivated and which was then chosen on an independent single-FET rail-bias set.
- The final models (section 6) use the same procedure with {HEADLINE_N:,} samples per case (`data{HEADLINE}/`); the dataset-size
  study (section 7) repeats it at 2,500 / 5,000 / 20,000 samples per case.

### Data check (Kirchhoff residual and physical trends)
```
{kir}{trend}```
Controlled sweeps (one parameter varied, all others nominal):
![OFF current vs temperature](plots/trend_off_current_vs_temperature.png)
![OFF current vs vth0](plots/trend_off_current_vs_vth0.png)
![Gate current vs toxe](plots/trend_gate_current_vs_toxe.png)

## 3. Models
- Inputs: the 12 sample parameters, turned into physics features (same for every model type): thermal voltage vT, terminal-voltage
  differences (PMOS voltages mirrored to NMOS polarity), the same divided by vT, gate overdrive (Vg* - |vth0|)/vT,
  the subthreshold drain-bias factor 1 - exp(-|Vds|/vT), oxide-field terms Vgs/toxe and Vgd/toxe, ln W, ln toxe, ln ndep, ln u0, xj.
{feat_note}
- Target per terminal: y = asinh(I / I0_t), I0_t = 1 % of the training median |I_t|. This is logarithmic (so the loss tracks the
  relative error) for |I| >> I0_t and stays linear and smooth where a current changes sign (Vds near 0).
- Four model types: polynomial ridge regression (degree 2 and 3, RidgeCV), random forest (500 trees), histogram gradient boosting
  (one per terminal, early stopping), and an MLP (4 hidden layers x 128, SiLU, Adam with cosine learning rate for 2000 epochs,
  then L-BFGS, ensemble of 5 seeds, float64).
- Selection: {tuned_note}; per case the model type with the smallest
  validation (median + 95th percentile) of the pooled relative error. Relative error = |dI| / max(|I_SPICE|, 1 fA).

### Model comparison (validation set, pooled over Id, Ig, Is, Ib)
{md(cmp_val)}

### Chosen models (test set)
{reasons}

Median / 95th-percentile relative error (%) per terminal; `pooled_|Vds|>5%VDD` excludes the near-zero-Vds samples
where the channel current itself crosses zero:
{md(chosen_test)}

Predicted-vs-SPICE plots (test set, log axes):
""" + "\n".join(f"![{c}](plots/pred_vs_spice_{c}.png)" for c in CASES) + f"""

## 4. Gate results (18 cases x 200 corners)
SPICE simulates each complete gate; the ML estimate takes every transistor's terminal voltages from the SPICE-recorded
output and internal nodes, picks the ON or OFF model by the gate voltage, predicts the four terminal currents and forms
the leakage as sum over devices of [(Vd-Vs) Id + (Vg-Vs) Ig + (Vb-Vs) Ib] / VDD. By Tellegen's theorem this equals the
current drawn from the VDD-level sources exactly (checked with 0 V ammeters on every terminal: {tel}); it avoids taking
small differences of large ON-device currents.

Metrics in every gate and circuit table, per row over its n simulations: relative error e = (ML - SPICE) / SPICE;
bias = median e; median error = median |e|; p95 error = 95th percentile of |e|; **mean accuracy = 100 % - mean |e|**.

{md(gt)}

Overall: mean accuracy **{go.mean_accuracy_pct:.2f} %**, median error {go.median_err_pct:.2f} %, 95th percentile {go.p95_err_pct:.2f} %,
bias {go.bias_pct:+.2f} % (n = {int(go.n)}).
Most accurate case: {best.gate} {best.pattern} (median error {best.median_err_pct:.3g} %); least accurate: {worst.gate} {worst.pattern}
({worst.median_err_pct:.3g} %).
![gate parity](plots/gate_parity.png)
"""]
if sc is not None:
    s = sc.iloc[0]
    A.append(f"""**Extra (not required): ML-only node voltages.** Instead of reading x, y and the INV input of AND2/OR2 from SPICE, the
internal node voltages are solved from the ML currents by KCL (grid search plus bisection, Gauss-Seidel over coupled nodes).
With these self-consistent voltages the gate leakage mean accuracy is {s.mean_accuracy_pct:.2f} % (median error {s.median_err_pct:.2f} %).
""" + (f"Node-voltage errors vs SPICE:\n\n{md(nv)}\n" if nv is not None else ""))

A.append(f"""
## 5. Test-circuit results (8-bit parity checker, 256 vectors x 200 corners)
SPICE simulates the full 112-transistor netlist for every input vector at the same 200 corners (output logic checked in every
run). The hierarchical ML estimate evaluates the logic value at each gate's inputs, looks up that gate's ML leakage at the
corner and sums the 21 gate leakages.

{md(co)}

Compared with the gate level: circuit mean accuracy {c_ml.mean_accuracy_pct:.2f} % vs {go.mean_accuracy_pct:.2f} % for the gates,
median error {c_ml.median_err_pct:.2f} % vs {go.median_err_pct:.2f} %, 95th percentile {c_ml.p95_err_pct:.2f} % vs {go.p95_err_pct:.2f} %.
Summing 21 gate estimates averages out part of the independent per-gate error, while the common (bias) part adds up.
""" + (f"""
The second row isolates the error of the hierarchical assumption itself: summing **SPICE** gate leakages (no ML) differs from the
full-circuit SPICE result by only {c_ref.median_err_pct:.2f} % median ({c_ref.mean_accuracy_pct:.2f} % mean accuracy), so gate-to-gate
loading is small and the circuit-level error is almost entirely the gate-level ML error.
""" if c_ref is not None else "") + f"""
By number of ones in the input vector:
{md(chw)}
![circuit parity](plots/circuit_parity.png)
""")
if HEAD:
    hch = json.load(open(os.path.join(ROOT, "saved_models" + HEADLINE, "chosen.json")))
    hb, hw_ = h_gt.loc[h_gt.median_err_pct.idxmin()], h_gt.loc[h_gt.median_err_pct.idxmax()]
    A.append(f"""
## 6. Final models: {hn} samples per case
The pipeline of sections 2-5 unchanged, run with {hn} samples per case: datasets `data{HEADLINE}/` (4 x {hn} rows, the same
Section 4 sampling and ranges, checked in `results/10_requirements_check.txt`), models `saved_models{HEADLINE}/`, tables
`results{HEADLINE}/`, plots `plots{HEADLINE}/`. The 70/15/15 split, the four model types and the selection rule are the same; the
gate corners and the full-circuit SPICE reference are identical to the main run, so every number here is directly comparable with
sections 4 and 5. Chosen model types: {", ".join(f"{c} {k}" for c, k in hch.items())}.

How this size was chosen: {size_basis} So this choice used the gate and circuit SPICE results (as noted in section 2); the
model types, hyper-parameters and trained weights come from the same procedure as in sections 2-5 and never see them. For that
reason the specified 10,000-sample results are reported alongside.

### Chosen models (test set)
{md(h_dev)}

### Gate results (18 cases x 200 corners)
{md(h_gt)}

Overall: mean accuracy **{hg.mean_accuracy_pct:.2f} %**, median error {hg.median_err_pct:.2f} %, 95th percentile {hg.p95_err_pct:.2f} %,
bias {hg.bias_pct:+.2f} % (n = {int(hg.n)}); with the specified 10,000 samples per case: {go.mean_accuracy_pct:.2f} %, {go.median_err_pct:.2f} %,
{go.p95_err_pct:.2f} %. Most accurate case: {hb.gate} {hb.pattern} (median error {hb.median_err_pct:.3g} %); least accurate:
{hw_.gate} {hw_.pattern} ({hw_.median_err_pct:.3g} %).""" + (f"""
With ML-solved internal node voltages (extra, as in section 4): {h_sc.iloc[0].mean_accuracy_pct:.2f} % mean accuracy.""" if h_sc is not None else "") + f"""
![gate parity, {hn} samples](plots{HEADLINE}/gate_parity.png)

### Test-circuit results (8-bit parity checker, 256 vectors x 200 corners)
{md(h_co)}

Mean accuracy **{hc.mean_accuracy_pct:.2f} %** (specified 10,000 samples per case: {c_ml.mean_accuracy_pct:.2f} %). By number of ones in the input vector:
""" + (md(h_hw) if h_hw is not None else "") + f"""
![circuit parity, {hn} samples](plots{HEADLINE}/circuit_parity.png)
""" + (f"""
### Remaining error by device case
Same attribution as in section 7 (one device case replaced by exact single-FET SPICE currents at the gate bias points):

{md(h_att)}

With ML everywhere: gate median error {hg.median_err_pct:.2f} %.
""" if h_att is not None else ""))

A.append("""
## 7. Error analysis
The analysis below is for the specified 10,000-sample run (sections 2-5); the dataset-size study covers all runs.
""")
if att is not None:
    A.append(f"""### Which device model limits the gate accuracy
Every transistor bias point that occurs inside the 18 gates x 200 corners was re-simulated as a single FET in SPICE. Replacing the
ML currents of one device case by the exact SPICE currents (all other cases stay ML) gives the remaining gate-level error:

{md(att)}

With ML everywhere the same computation gives median {go.median_err_pct:.2f} %. The case whose replacement removes the most error is
the one that limits the gate accuracy. Inside gates, Vd and Vs sit at the rails (0 or VDD), i.e. at the corners of the uniform
Vd-Vs training square, where the training data is thinnest; device errors at these bias points are larger than on the random test set.
""")
if pv is not None:
    keep = ["variant", "train_med", "val_med", "val_power_med", "val_corner_power_med", "rail_corner_power_med"]
    b = pv.set_index("variant")
    gap = (b.val_med / b.train_med)[~b.index.str.startswith("small")]
    base = b.loc[b.index.str.startswith("base")].iloc[0]; vb, rb = base.val_med, base.rail_corner_power_med

    def better(k):
        w = b[b[k] < base[k]][k].sort_values()
        return ", ".join(f"{i} {v:.3f} %" for i, v in w.items()) or "none"
    A.append(f"""### Why pmos_off is the weakest device model
Inside the gates every OFF PMOS has Vs = Vg = Vb = VDD, so only (Vd - Vs) Id enters the leakage; that corner (Vs > 0.95 VDD) holds
about 5 % of the uniform training data (`experiments/exp8_pmos_off_diag.py`). Variants trained on the same 10k data and split, scored
on validation and on the independent rail-bias set (`corner` = Vs > 0.95 VDD; the gate test corners are not used); errors in %:

{md(pv[[k for k in keep if k in pv]], "{:.3f}")}

- The training error is {gap.min():.0f}-{gap.max():.0f}x below the validation error (except for the deliberately
  smaller network): the model fits its training set far better than new data, i.e. it is variance (data) limited.
- Against the base network (validation median {vb:.3f} %, rail-corner power median {rb:.3f} %), lower validation error:
  {better('val_med')}; lower rail-corner error: {better('rail_corner_power_med')}. The best variant on validation
  ({b.val_med.idxmin()}) lowers the validation error by {100 * (1 - b.val_med.min() / vb):.0f} % and the rail-corner error by
  {100 * (1 - b.rail_corner_power_med[b.val_med.idxmin()] / rb):.0f} %""" + (f"""; doubling the data to 20,000 samples per case
  with the base settings lowers the validation error by {100 * (1 - sval.loc[20000, 'pmos_off'] / sval.loc[10000, 'pmos_off']):.0f} %
  (learning curve below).""" if {10000, 20000} <= set(sval.index) else ".") + """
""" + (f"""- The SPICE data itself is smooth: re-simulating a training row reproduces it to
  {e10[e10['var'] == 'repeat'].max_res_pct.max() / 100:.0e} relative, and narrow one-parameter
  sweeps deviate from a smooth polynomial by at most {e10[e10['var'].str.endswith('power')].max_res_pct.max():.4f} % in device power
  (`experiments/exp10_spice_smoothness.py`).
""" if e10 is not None else "") + (f"""- Learning BSIM4's gate-overlap tunnelling currents (Igd, Igs) as separate targets and rebuilding the terminal currents was worse
  for every case (validation median, terminal vs separated targets: {", ".join(f"{c} {e11[(e11.case == c) & (e11.targets == 'terminal')].val_med.iloc[0]:.2f} / {e11[(e11.case == c) & (e11.targets == 'components')].val_med.iloc[0]:.2f} %" for c in CASES if (e11.case == c).any())}):
  in OFF devices Igd is practically the whole drain current ({igd_share}), so the split adds nothing (`experiments/exp11_component_targets.py`).
""" if e11 is not None else "") + f"""- The {b.val_med.idxmin()} variant costs {b.secs[b.val_med.idxmin()] / base.secs:.1f}x the training time of the base.
""")
A.append(f"""### Full analysis output
```
{ea}
```
![error vs temperature](plots/error_vs_temp.png)
""")
if lc is not None:
    cols = ["samples_per_case", "gate_mean_accuracy_pct", "gate_median_err_pct", "gate_p95_err_pct",
            "circuit_mean_accuracy_pct", "circuit_median_err_pct"] + [f"{c}_test_median_pct" for c in CASES]
    hit = lc[lc.gate_mean_accuracy_pct >= 99]
    hc = lc[lc.circuit_mean_accuracy_pct >= 99] if "circuit_mean_accuracy_pct" in lc else lc.iloc[:0]
    sizes = ", ".join(f"{int(n):,}" for n in lc.samples_per_case)
    reach = (f"Of the sizes tested ({sizes} samples per case), the smallest that reaches 99 % mean accuracy is "
             f"{int(hit.samples_per_case.min()):,} at gate level" if len(hit) else "None of the runs reaches 99 % mean accuracy at gate level") + \
            (f" and {int(hc.samples_per_case.min()):,} for the parity checker." if len(hc) else ".")
    # error ratio per doubling of the data, per device case (test median)
    l2 = lc.sort_values("samples_per_case")
    ratio = {c: (l2[f"{c}_test_median_pct"].values[1:] / l2[f"{c}_test_median_pct"].values[:-1]) for c in CASES}
    rtxt = "; ".join(f"{c} {r.min():.2f}-{r.max():.2f}" for c, r in ratio.items())
    lc = lc.assign(samples_per_case=lc.samples_per_case.astype(int))
    sv = sval.reset_index().rename(columns={"index": "samples_per_case"} | {c: f"{c}_val_median_pct" for c in CASES})
    A.append(f"""### Dataset size (learning curve)
Each row is an independent full run (new data, retraining, the same 200 gate corners). The parity-checker columns are the
hierarchical sum of that run's gate estimates against the same full-circuit SPICE reference (the SPICE side does not depend on
the models):

{md(lc[[c for c in cols if c in lc]], "{:.3f}")}

Factor by which the device test median error changes per doubling of the data: {rtxt}. """ + (
        "Every device error falls with every doubling and the gate and circuit errors follow, so the remaining error is mainly a "
        "data (variance) limit, not a model-form limit." if all((r < 1).all() for r in ratio.values()) else
        "Not every device error falls with every doubling: see the table.") + f""" {reach}

Best device validation median error (%) per size:

{md(sv, "{:.3f}")}

![learning curve](plots/learning_curve.png)
""")
worst_dev = pooled["pooled median %"].idxmax()
limit = f"- The {worst_dev} model has the largest pooled test error ({pooled.loc[worst_dev, 'pooled median %']:.2f} % median)"
if att is not None:
    a = att.set_index("exact_case").median_err_pct
    limit += (f"; making the {a.idxmin()} currents exact lowers the gate median error from {go.median_err_pct:.2f} % to "
              f"{a.min():.2f} %, the largest single gain.")
A.append("""
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
""" + limit + "\n")
open(os.path.join(ROOT, f"report{TAG}.md"), "w").write("\n".join(A))
print("wrote report", TAG)
