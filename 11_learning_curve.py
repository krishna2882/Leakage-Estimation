"""Learning curve: device-, gate- and circuit-level accuracy vs samples per case.
Each point is an independent full run (tiers 1-3) with DVD_TAG = '' (10k) or '_2k5', '_5k', '_20k', '_40k'.
The circuit SPICE reference does not depend on the models and every run uses the same 200 corners, so the circuit
accuracy of each run is its hierarchical gate sum (07_circuit_ml.py) against the main run's 06_circuit_spice.csv."""
import json, importlib
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, CASES, os, table_metrics
C7 = importlib.import_module("07_circuit_ml")

RUNS = [("_2k5", 2500), ("_5k", 5000), ("", 10000), ("_20k", 20000), ("_40k", 40000)]
sp0 = pd.read_csv(os.path.join(ROOT, "results", "06_circuit_spice.csv"))
corners0 = open(os.path.join(ROOT, "results", "corners.json")).read()
rows = []
for tag, n in RUNS:
    res = os.path.join(ROOT, "results" + tag)
    f3, f5 = os.path.join(res, "03_model_metrics_long.csv"), os.path.join(res, "05_gate_overall.csv")
    if not (os.path.exists(f3) and os.path.exists(f5)):
        print("missing run", tag or "10k"); continue
    m = pd.read_csv(f3); t = m[(m.split == "test") & (m.terminal == "pooled")]
    g = pd.read_csv(f5).iloc[0]
    r = dict(samples_per_case=n, chosen=json.dumps(dict(zip(t.case, t.model))),
             gate_mean_accuracy_pct=g.mean_accuracy_pct, gate_median_err_pct=g.median_err_pct, gate_p95_err_pct=g.p95_err_pct)
    if open(os.path.join(res, "corners.json")).read() == corners0:
        gr = pd.read_csv(os.path.join(res, "05_gate_results_raw.csv"), dtype={"pattern": str})
        c = table_metrics(C7.circuit_columns(gr, sp0.copy(), [("ml", "ml")]).assign(grp="all"), ["grp"]).iloc[0]
        r.update(circuit_mean_accuracy_pct=c.mean_accuracy_pct, circuit_median_err_pct=c.median_err_pct, circuit_p95_err_pct=c.p95_err_pct)
    for _, x in t.iterrows():
        r[f"{x.case}_test_median_pct"] = x.med; r[f"{x.case}_test_p95_pct"] = x.p95
    rows.append(r)
df = pd.DataFrame(rows).sort_values("samples_per_case")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "results", "11_learning_curve.csv"), index=False)
print(df.round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for c in CASES:
    if f"{c}_test_median_pct" in df:
        ax[0].loglog(df.samples_per_case, df[f"{c}_test_median_pct"], "o-", label=c)
ax[0].set(xlabel="samples per case", ylabel="test median relative error (%)", title="Device models (chosen per case)")
ax[1].semilogx(df.samples_per_case, df.gate_median_err_pct, "o-", label="median error")
ax[1].semilogx(df.samples_per_case, df.gate_p95_err_pct, "s-", label="95th percentile error")
if "circuit_median_err_pct" in df:
    ax[1].semilogx(df.samples_per_case, df.circuit_median_err_pct, "o--", label="parity checker median error")
    ax[1].semilogx(df.samples_per_case, df.circuit_p95_err_pct, "s--", label="parity checker 95th percentile")
ax[1].set(xlabel="samples per case", ylabel="leakage error (%)", title="Gates (18 cases x 200 corners) and parity checker")
for a in ax: a.grid(alpha=.3, which="both"); a.legend()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "plots", "learning_curve.png"), dpi=140); plt.close()
