"""Tier 5a: where are the errors? device level (test sets), gate level, circuit level."""
import json, importlib
import pandas as pd, joblib
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
T3 = importlib.import_module("03_train_models")

L = []
def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s)

# ---- device level, test split, chosen models
ch = json.load(open(os.path.join(SAVED, "chosen.json")))
P("== Device level (test sets): relative error % by region of |Vd-Vs|/VDD (pooled over id,ig,is,ib)")
for case, kind in ch.items():
    m = joblib.load(os.path.join(SAVED, f"{case}_{kind}.joblib"))
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); _, _, te = T3.split(df)
    e = T3.relerr(m.predict(te), te[TARGETS].values)
    vds = (te.vd - te.vs).abs() / te.vdd
    for lo, hi in [(0, .05), (.05, .3), (.3, 1.01)]:
        k = ((vds >= lo) & (vds < hi)).values
        if k.sum():
            P(f"{case:9s} |Vds|/VDD in [{lo:.2f},{hi:.2f}): n={k.sum():4d} median {np.median(e[k]):7.2f}%  p95 {np.percentile(e[k],95):8.2f}%  "
              f"(Id,Is only: median {np.median(e[k][:, [0,2]]):.2f}%, p95 {np.percentile(e[k][:, [0,2]],95):.2f}%)")
    for q, nm in [(te.temp, "temp"), (te.vth0.abs(), "|vth0|")]:
        rho = spearmanr(q, np.median(e[:, :3], axis=1))[0]
        P(f"{case:9s} Spearman(median err over id,ig,is ; {nm}) = {rho:+.3f}")

corners = json.load(open(os.path.join(RES, "corners.json")))
cdf = pd.DataFrame([dict(corner=i, vdd=c["vdd"], temp=c["temp"], **{f"{k}_n": c["n"][k] for k in PKEYS}, **{f"{k}_p": c["p"][k] for k in PKEYS}) for i, c in enumerate(corners)])
cdf["vth0_n_abs"] = cdf.vth0_n.abs(); cdf["vth0_p_abs"] = cdf.vth0_p.abs()
cols = ["temp", "vdd", "vth0_n", "vth0_p_abs", "toxe_n", "toxe_p", "u0_n", "u0_p", "ndep_n", "ndep_p", "xj_n", "xj_p"]

def analyse(df, name, keys):
    d = df.merge(cdf.drop(columns=[c for c in ("vdd", "temp") if c in df.columns]), on="corner")
    d["err"] = (d.ml - d.spice) / d.spice * 100
    P(f"\n== {name}: Spearman(|err|, parameter)")
    for c in cols:
        P(f"  {c:10s} rho = {spearmanr(d[c], d.err.abs())[0]:+.3f}   (signed err: {spearmanr(d[c], d.err)[0]:+.3f})")
    d["tbin"] = pd.qcut(d.temp, 5)
    P(f"== {name}: |err| % by temperature quintile")
    P(d.groupby("tbin", observed=True).err.apply(lambda x: pd.Series(dict(median=x.abs().median(), p95=x.abs().quantile(.95)))).unstack().round(3).to_string())
    P(f"== {name}: worst 8 rows")
    P(d.reindex(d.err.abs().sort_values(ascending=False).index).head(8)[keys + ["corner", "temp", "vdd", "spice", "ml", "err"]].to_string())
    return d

g = pd.read_csv(os.path.join(RES, "05_gate_results_raw.csv"), dtype={"pattern": str})
gd = analyse(g, "Gate level", ["gate", "pattern"])
c = pd.read_csv(os.path.join(RES, "07_circuit_results_raw.csv"))
cd = analyse(c, "Circuit level", ["vec"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].scatter(gd.temp, gd.err, s=2, alpha=.3); ax[0].set(xlabel="Temperature (C)", ylabel="signed error (%)", title="Gate level: error vs temperature")
ax[1].scatter(cd.temp, cd.err, s=2, alpha=.3); ax[1].set(xlabel="Temperature (C)", ylabel="signed error (%)", title="Circuit level: error vs temperature")
for a in ax: a.grid(alpha=.3); a.axhline(0, c="k", lw=.6)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "error_vs_temp.png"), dpi=130); plt.close()
open(os.path.join(RES, "08_error_analysis.txt"), "w").write("\n".join(L) + "\n")

# ---- device models evaluated exactly at the bias points they see inside the gates (SPICE single-FET reruns)
# and attribution: replace the ML currents of one device case by the SPICE currents -> remaining gate error
G5 = importlib.import_module("05_gate_ml")
models = G5.load_models()
gr = g.sort_values(["gate", "pattern", "corner"])
jobs, meta = [], []
for (kind, pat), grp in gr.groupby(["gate", "pattern"], sort=False):
    volts = G5.node_volts(kind, pat, corners, grp.sort_values("corner"))
    for dev in gate_devices(kind, ["A", "B"][:len(pat)], "out"):
        name, typ, d, gg, s, b, w = dev
        for ci, c in enumerate(corners):
            vg = volts[gg][ci]; vdd = c["vdd"]
            on = (vg > vdd / 2) if typ == "nmos" else (vg < vdd / 2)
            case = f"{typ}_{'on' if on else 'off'}"
            row = dict(vdd=vdd, temp=c["temp"], w=w, vd=volts[d][ci], vg=vg, vs=volts[s][ci], vb=volts[b][ci], **c[typ[0]])
            jobs.append((single_netlist(case, row), ["i(vd)", "i(vg)", "i(vs)", "i(vb)"], None))
            meta.append(dict(gate=kind, pattern=pat, device=name, corner=ci, case=case, **row))
res = run_many(jobs)
assert all(r is not None for r in res)
dv = pd.DataFrame(meta)
for t, v in zip(TARGETS, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]):
    dv[f"{t}_spice"] = [-r[v] for r in res]
for case in CASES:
    k = dv.case == case
    if k.any():
        pred = models[case].predict(dv.loc[k, FEATS])
        for j, t in enumerate(TARGETS):
            dv.loc[k, f"{t}_ml"] = pred[:, j]
dv["p_spice"] = (dv.vd - dv.vs) * dv.id_spice + (dv.vg - dv.vs) * dv.ig_spice + (dv.vb - dv.vs) * dv.ib_spice
dv["p_ml"] = (dv.vd - dv.vs) * dv.id_ml + (dv.vg - dv.vs) * dv.ig_ml + (dv.vb - dv.vs) * dv.ib_ml
dv.to_csv(os.path.join(RES, "08_device_error_at_gate_bias.csv"), index=False)
P("\n== Device models at the bias points used inside the gates (SPICE single-FET reruns)")
for case, x in dv.groupby("case"):
    sig = x.p_spice.abs() > 1e-3 * x.p_spice.abs().max()      # devices that carry a non-negligible share of the power
    ep = ((x.p_ml - x.p_spice) / x.p_spice.abs())[sig] * 100
    eg = ((x.ig_ml - x.ig_spice) / x.ig_spice.abs()) * 100
    P(f"{case:9s} n={len(x):5d}  power-term error: median {ep.abs().median():.3f}%  p95 {ep.abs().quantile(.95):.2f}%  bias {ep.median():+.3f}%"
      f"   | Ig error median {eg.abs().median():.3f}%")
# attribution
tot_sp = dv.groupby(["gate", "pattern", "corner"]).apply(lambda x: x.p_spice.sum() / x.vdd.iloc[0])
base = dv.groupby(["gate", "pattern", "corner"]).apply(lambda x: x.p_ml.sum() / x.vdd.iloc[0])
e0 = ((base - tot_sp) / tot_sp * 100).abs()
P(f"\n== Attribution (gate level): ML everywhere -> median |err| {e0.median():.3f}%, p95 {e0.quantile(.95):.2f}%")
att = []
for case in CASES:
    pw = np.where(dv.case == case, dv.p_spice, dv.p_ml)
    alt = dv.assign(pw=pw).groupby(["gate", "pattern", "corner"]).apply(lambda x: x.pw.sum() / x.vdd.iloc[0])
    e = ((alt - tot_sp) / tot_sp * 100).abs()
    att.append(dict(exact_case=case, median_err_pct=e.median(), p95_err_pct=e.quantile(.95)))
    P(f"   with exact SPICE currents for {case:9s}: median |err| {e.median():.3f}%, p95 {e.quantile(.95):.2f}%")
pd.DataFrame(att).to_csv(os.path.join(RES, "08_error_attribution.csv"), index=False)
open(os.path.join(RES, "08_error_analysis.txt"), "w").write("\n".join(L) + "\n")
