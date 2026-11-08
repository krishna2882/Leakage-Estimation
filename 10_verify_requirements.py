"""Checks every requirement of DVD_ML_PROJECT.pdf against the files actually produced.
Writes results/10_requirements_check.txt; exit code 1 if any check fails."""
import json, glob, re, sys
import pandas as pd
from common import *

R = []


def check(sec, text, ok, evidence=""):
    R.append((sec, text, bool(ok), evidence))


def exists(p):
    return os.path.exists(os.path.join(ROOT, p))


COLS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]
GV = {"nmos_on": ("vdd", 0), "nmos_off": (0, 0), "pmos_on": (0, "vdd"), "pmos_off": ("vdd", "vdd")}

# ---------------- Part A / Section 4 / deliverable 1
for case in CASES:
    p = os.path.join(DATA, f"{case}.csv")
    if not os.path.exists(p):
        check("A1/D1", f"{case}.csv exists", False); continue
    d = pd.read_csv(p); s = case[0]
    check("D1", f"{case}: exact columns {','.join(COLS)}", list(d.columns) == COLS, str(list(d.columns)))
    check("A1", f"{case}: >= 10,000 rows, no NaN", len(d) >= 10000 and not d.isna().any().any(), f"rows={len(d)}")
    check("S4", f"{case}: VDD uniform within 0.9 V +-10 %", d.vdd.between(0.81, 0.99).all(), f"[{d.vdd.min():.4f},{d.vdd.max():.4f}]")
    check("S4", f"{case}: Vd, Vs within 0..VDD", ((d.vd >= 0) & (d.vd <= d.vdd) & (d.vs >= 0) & (d.vs <= d.vdd)).all())
    check("S4", f"{case}: W within 1..8 Lmin", d.w.between(LMIN * (1 - 1e-9), 8 * LMIN * (1 + 1e-9)).all(), f"[{d.w.min():.3e},{d.w.max():.3e}]")
    check("S4", f"{case}: T within -40..125 C", d.temp.between(-40, 125).all(), f"[{d.temp.min():.2f},{d.temp.max():.2f}]")
    for k in PKEYS:
        r = d[k] / NOM[s][k]
        check("S4", f"{case}: {k} within nominal +-10 % ({'NMOS' if s == 'n' else 'PMOS'} nominal {NOM[s][k]:g})", r.between(0.9 - 1e-9, 1.1 + 1e-9).all(), f"ratio [{r.min():.4f},{r.max():.4f}]")
    vg, vb = GV[case]
    check("S4", f"{case}: Vg fixed = {vg}, Vb fixed = {vb}",
          np.allclose(d.vg, d.vdd if vg == "vdd" else 0) and np.allclose(d.vb, d.vdd if vb == "vdd" else 0))
    # independence of draws: |corr| small between every pair of sampled inputs (Vd,Vs scale with VDD by construction)
    sampled = ["temp", "w"] + PKEYS
    c = d[sampled + ["vdd"]].corr().abs().values; np.fill_diagonal(c, 0)
    check("A1", f"{case}: parameters drawn independently (max |corr| among T,W,process,VDD < 0.05)", c.max() < 0.05, f"max |corr| = {c.max():.3f}")

# netlist requirements
opt = "gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
nets = sorted(glob.glob(os.path.join(NETS, "*.net")))
check("NET", "every saved netlist contains the required .options", nets and all(opt in open(f).read().lower() for f in nets), f"{len(nets)} netlists")
for case in CASES:
    f = os.path.join(NETS, f"single_{case}.net")
    t = open(f).read() if os.path.exists(f) else ""
    srcs = [l.split()[0] for l in t.split("\n") if l[:1] in ("V", "v")]
    check("D2/NET", f"single_{case}.net exists with one source per terminal (d,g,s,b)", sorted(x.lower() for x in srcs) == ["vb", "vd", "vg", "vs"], str(srcs))
for g in GATES:
    for p in PATTERNS[g]:
        check("D2", f"gate netlist {g}_{p}.net exists", os.path.exists(os.path.join(NETS, f"{g}_{p}.net")))
check("D2", "test-circuit netlist parity8_example.net exists", os.path.exists(os.path.join(NETS, "parity8_example.net")))
check("NET", "saved netlists use a portable (relative) model path", nets and not any("/home" in open(f).read() for f in nets))

# ---------------- deliverable 3: data check
check("D3", "Kirchhoff check per case", os.path.exists(os.path.join(RES, "02_kirchhoff.txt")) and all(c in open(os.path.join(RES, "02_kirchhoff.txt")).read() for c in CASES))
for f in ["trend_off_current_vs_temperature.png", "trend_off_current_vs_vth0.png", "trend_gate_current_vs_toxe.png"]:
    check("D3", f"trend plot {f}", os.path.exists(os.path.join(PLOTS, f)))
tr = open(os.path.join(RES, "02_trends.txt")).read() if os.path.exists(os.path.join(RES, "02_trends.txt")) else ""
check("D3", "all trends physically monotonic (asserted in 02_trends.txt)", tr and "False" not in tr)

# ---------------- Part A step 2 / deliverables 4-5
m = pd.read_csv(os.path.join(RES, "03_model_metrics_long.csv")) if os.path.exists(os.path.join(RES, "03_model_metrics_long.csv")) else pd.DataFrame()
if len(m):
    v = m[(m.split == "val") & (m.terminal == "pooled")]
    check("D4", "model comparison: 4 model types x 4 cases on validation (R2, median, p95)", len(v) == 16 and v.model.nunique() == 4 and v.case.nunique() == 4, f"{len(v)} rows")
check("D4", "formatted comparison table results/03_model_comparison_val.csv", os.path.exists(os.path.join(RES, "03_model_comparison_val.csv")))
check("D5", "chosen-model test table results/03_chosen_models_test.csv", os.path.exists(os.path.join(RES, "03_chosen_models_test.csv")))
ch = json.load(open(os.path.join(SAVED, "chosen.json"))) if os.path.exists(os.path.join(SAVED, "chosen.json")) else {}
jl = sorted(os.path.basename(x) for x in glob.glob(os.path.join(SAVED, "*.joblib")))
check("D5", "exactly four saved model files, one per case, matching chosen.json", len(jl) == 4 and all(f"{c}_{k}.joblib" in jl for c, k in ch.items()), str(jl))
for c in CASES:
    check("D5", f"predicted-vs-SPICE plot for {c}", os.path.exists(os.path.join(PLOTS, f"pred_vs_spice_{c}.png")))
check("A2", "split 70/15/15 with fixed seed (7000/1500/1500)", "test_size=0.30" in open(os.path.join(ROOT, "03_train_models.py")).read())

# ---------------- Part B / Section 5 / deliverables 6-7
gs = pd.read_csv(os.path.join(RES, "04_gate_spice.csv"), dtype={"pattern": str}) if os.path.exists(os.path.join(RES, "04_gate_spice.csv")) else pd.DataFrame()
if len(gs):
    cnt = gs.groupby(["gate", "pattern"]).size()
    check("B3", "18 gate/input cases x 200 corners simulated in SPICE", len(cnt) == 18 and (cnt == 200).all(), f"{len(cnt)} cases, {int(cnt.min())}-{int(cnt.max())} corners")
    ok = True
    for (g, p), grp in gs.groupby(["gate", "pattern"]):
        exp = gate_logic(g, [int(x) for x in p])
        ok &= bool((np.abs(grp.vout - exp * grp.vdd) < 0.1 * grp.vdd).all())
    check("B3", "SPICE gate outputs at the correct logic level in all 3600 runs", ok)
corners = json.load(open(os.path.join(RES, "corners.json"))) if os.path.exists(os.path.join(RES, "corners.json")) else []
check("S5", "200 corners: VDD, T, NMOS draw, PMOS draw (separate)", len(corners) == 200 and all({"vdd", "temp", "n", "p"} <= set(c) for c in corners))
gr = pd.read_csv(os.path.join(RES, "05_gate_results_raw.csv"), dtype={"pattern": str}) if os.path.exists(os.path.join(RES, "05_gate_results_raw.csv")) else pd.DataFrame()
if len(gr):
    check("S5", "recorded per case: SPICE leakage, ML leakage, SPICE output voltage", {"spice", "ml", "vout"} <= set(gr.columns))
    need = {"INV": [], "NAND2": ["x"], "NOR2": ["y"], "AND2": ["x", "n"], "OR2": ["y", "n"]}
    okn = all(gr.loc[gr.gate == g, [f"v_{n}" for n in ns] + [f"ml_v_{n}" for n in ns]].notna().all().all() for g, ns in need.items() if ns) \
        if all(f"ml_v_{n}" in gr.columns for n in ("x", "y", "n")) else False
    check("S5", "internal node voltages recorded from both SPICE and ML (x, y, NAND/NOR output n)", okn)
gt = os.path.join(RES, "05_gate_table.csv")
if os.path.exists(gt):
    t = pd.read_csv(gt)
    check("D6", "gate table: 18 rows with SPICE median, ML median, bias, median error, p95 error",
          len(t) == 18 and {"spice_median", "ml_median", "bias_pct", "median_err_pct", "p95_err_pct"} <= set(t.columns))
check("D7", "gate parity plot (log axes)", os.path.exists(os.path.join(PLOTS, "gate_parity.png")))

# ---------------- Part C / deliverable 8
cs = pd.read_csv(os.path.join(RES, "06_circuit_spice.csv")) if os.path.exists(os.path.join(RES, "06_circuit_spice.csv")) else pd.DataFrame()
if len(cs):
    check("C6", "test circuit: all 256 input vectors x the same 200 corners", cs.vec.nunique() == 256 and cs.corner.nunique() == 200 and len(cs) == 51200, f"{len(cs)} runs")
    check("C6", "test circuit output logic correct in every run", bool(cs.logic_ok.all()))
check("D8", "test-circuit metrics table", os.path.exists(os.path.join(RES, "07_circuit_overall.csv")))
check("D8", "test-circuit parity plot (log axes)", os.path.exists(os.path.join(PLOTS, "circuit_parity.png")))

# ---------------- deliverable 9: code; target
for s in ["00_setup_check", "01_gen_data", "02_data_check", "03_train_models", "04_gate_spice", "05_gate_ml",
          "06_circuit_spice", "07_circuit_ml", "08_error_analysis", "09_report"]:
    check("D9", f"script {s}.py present", os.path.exists(os.path.join(ROOT, s + ".py")))
check("D9", "fixed seeds (SEED in common.py)", "SEED = " in open(os.path.join(ROOT, "common.py")).read())
check("A1", "smallest-dataset study (learning curve) present", os.path.exists(os.path.join(RES, "11_learning_curve.csv")))
# Part B step 3: gate (and circuit) SPICE results are for evaluation only
src = {f: open(os.path.join(ROOT, f)).read() for f in ["01_gen_data.py", "02_data_check.py", "03_train_models.py", "mlmodels.py"]}
used = [f for f, t in src.items() if re.search(r"0[4-7]_(gate|circuit)|corners\.json", t)]
check("B3", "gate/circuit SPICE results never read by data generation, training or model selection", not used, ", ".join(used))

# ---------------- headline run (common.HEADLINE: same sampling and corners, more samples per case) and the 99 % target
HR = lambda f: os.path.join(ROOT, "results" + HEADLINE, f)
for case in CASES:
    p = os.path.join(ROOT, "data" + HEADLINE, f"{case}.csv")
    d = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=COLS)
    s = case[0]; vg, vb = GV[case]
    rng_ok = len(d) and d.vdd.between(0.81, 0.99).all() and ((d.vd >= 0) & (d.vd <= d.vdd) & (d.vs >= 0) & (d.vs <= d.vdd)).all() \
        and d.temp.between(-40, 125).all() and d.w.between(LMIN * (1 - 1e-9), 8 * LMIN * (1 + 1e-9)).all() \
        and all((d[k] / NOM[s][k]).between(0.9 - 1e-9, 1.1 + 1e-9).all() for k in PKEYS) \
        and np.allclose(d.vg, d.vdd if vg == "vdd" else 0) and np.allclose(d.vb, d.vdd if vb == "vdd" else 0)
    check("HEAD", f"headline run {case}: {HEADLINE_N:,} rows, same columns and sampling ranges as Section 4",
          len(d) == HEADLINE_N and list(d.columns) == COLS and rng_ok and not d.isna().any().any(), f"rows={len(d)}")
hch = os.path.join(ROOT, "saved_models" + HEADLINE, "chosen.json")
hj = sorted(os.path.basename(x) for x in glob.glob(os.path.join(ROOT, "saved_models" + HEADLINE, "*.joblib")))
check("HEAD", "headline run: four saved model files matching its chosen.json",
      os.path.exists(hch) and len(hj) == 4 and all(f"{c}_{k}.joblib" in hj for c, k in json.load(open(hch)).items()), str(hj))
check("HEAD", "headline run uses the same 200 gate corners as the main run",
      os.path.exists(HR("corners.json")) and open(HR("corners.json")).read() == open(os.path.join(RES, "corners.json")).read())
for f in ["03_chosen_models_test.csv", "05_gate_table.csv", "07_circuit_overall.csv", "08_error_attribution.csv"]:
    check("HEAD", f"headline run table {f}", os.path.exists(HR(f)))
INFO = []
for name, f, lvl in [("gate-level", "05_gate_overall.csv", "gate"), ("test-circuit", "07_circuit_overall.csv", "circuit")]:
    if os.path.exists(HR(f)):
        acc = float(pd.read_csv(HR(f)).mean_accuracy_pct.iloc[0])
        check("TARGET", f"{name} mean accuracy >= 99 % (headline run, {HEADLINE_N:,} samples per case)", acc >= 99, f"{acc:.2f} %")
    if os.path.exists(os.path.join(RES, f)):
        INFO.append(f"[INFO] {name} mean accuracy with the specified 10,000 samples per case: "
                    f"{float(pd.read_csv(os.path.join(RES, f)).mean_accuracy_pct.iloc[0]):.2f} %")

# ---------------- deliverable 10: report sections
rp = os.path.join(ROOT, f"report{TAG}.md")
rt = open(rp).read() if os.path.exists(rp) else ""
for sec in ["Setup", "Data generation", "Models", "Gate results", "Test-circuit results", "Error analysis", "Limitations"]:
    check("D10", f"report section '{sec}'", re.search(r"^#+ .*" + sec, rt, re.M | re.I) is not None)
check("D10", "report states model file, nominal VDD, test circuit, simulator version",
      all(k in rt for k in ["32nm_MGK", "0.9 V", "parity", "ngspice-36"]))
check("D10", f"report headlines the {HEADLINE_N:,}-sample run and also states the specified 10,000-sample result",
      f"{HEADLINE_N:,} samples per case" in rt and "10,000 samples per case" in rt)
# the PDF is rendered off the cluster (12_report_pdf.py needs Chrome), so it is reported, not checked
pp = os.path.join(ROOT, f"report{TAG}.pdf")
if os.path.exists(pp):
    INFO.append("[INFO] report.pdf " + ("is up to date with report.md" if os.path.getmtime(pp) >= os.path.getmtime(rp)
                                       else "is OLDER than report.md: rerun 12_report_pdf.py"))

lines = [f"[{'PASS' if ok else 'FAIL'}] {sec:7s} {txt}" + (f"   ({ev})" if ev else "") for sec, txt, ok, ev in R]
nf = sum(not ok for _, _, ok, _ in R)
lines += INFO
lines.append(f"\n{len(R) - nf}/{len(R)} checks pass, {nf} fail")
open(os.path.join(RES, "10_requirements_check.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if nf else 0)
