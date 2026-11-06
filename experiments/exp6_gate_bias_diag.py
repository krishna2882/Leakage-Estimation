"""Why does gate accuracy not improve with dataset size while device test error halves per doubling?
Evaluates the 10k / 20k / 40k device models at every transistor bias point that occurs inside the gates
(SPICE single-FET currents from results/08_device_error_at_gate_bias.csv) and breaks the error down by where
Vd and Vs sit (at a rail, inside, or outside the training range 0..VDD)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, joblib
from common import *

dv = pd.read_csv(os.path.join(ROOT, "results", "08_device_error_at_gate_bias.csv"), dtype={"pattern": str})


def where(v, vdd, tol=0.01):
    r = v / vdd
    return np.select([r < -1e-6, r > 1 + 1e-6, r < tol, r > 1 - tol], ["below0", "aboveVDD", "rail0", "railV"], "inside")


dv["d_at"] = where(dv.vd.values, dv.vdd.values); dv["s_at"] = where(dv.vs.values, dv.vdd.values)
print("== bias-point locations (count per case)")
print(dv.groupby(["case", "d_at", "s_at"]).size().unstack(["d_at", "s_at"]).fillna(0).astype(int).to_string())
print("\n== range: min/max of Vd/VDD and Vs/VDD per case")
print(dv.assign(rd=dv.vd / dv.vdd, rs=dv.vs / dv.vdd).groupby("case")[["rd", "rs"]].agg(["min", "max"]).to_string())

sp = dv.p_spice.values
key = ["gate", "pattern", "corner"]
tot_sp = dv.assign(p=sp).groupby(key).apply(lambda x: x.p.sum() / x.vdd.iloc[0])
for tag in ["", "_20k", "_40k"]:
    sv = os.path.join(ROOT, "saved_models" + tag)
    if not os.path.exists(os.path.join(sv, "chosen.json")):
        continue
    ch = json.load(open(os.path.join(sv, "chosen.json")))
    pm = np.zeros(len(dv)); ig = np.zeros(len(dv))
    for case, kind in ch.items():
        k = (dv.case == case).values
        pr = joblib.load(os.path.join(sv, f"{case}_{kind}.joblib")).predict(dv.loc[k, FEATS])
        x = dv[k]
        pm[k] = (x.vd - x.vs) * pr[:, 0] + (x.vg - x.vs) * pr[:, 1] + (x.vb - x.vs) * pr[:, 3]    # columns id, ig, is, ib
        ig[k] = pr[:, 1]
    dv["pm"] = pm; dv["ig_m"] = ig
    tot = dv.groupby(key).apply(lambda x: x.pm.sum() / x.vdd.iloc[0])
    e = (tot - tot_sp) / tot_sp * 100
    print(f"\n######## models{tag or '_10k'}: gate total (SPICE device currents as reference) median |err| {e.abs().median():.3f}%  "
          f"p95 {e.abs().quantile(.95):.2f}%  bias {e.median():+.3f}%")
    for case, x in dv.groupby("case"):
        sig = x.p_spice.abs() > 1e-3 * x.p_spice.abs().max()
        ep = ((x.pm - x.p_spice) / x.p_spice.abs() * 100)[sig]
        eg = (x.ig_m - x.ig_spice) / x.ig_spice.abs() * 100
        print(f"  {case:9s} power-term |err| median {ep.abs().median():.3f}% p95 {ep.abs().quantile(.95):.2f}% bias {ep.median():+.3f}%"
              f" | Ig |err| median {eg.abs().median():.3f}% bias {eg.median():+.3f}%")
        for (da, sa), y in x[sig].groupby(["d_at", "s_at"]):
            ey = (y.pm - y.p_spice) / y.p_spice.abs() * 100
            share = y.p_spice.abs().sum() / x.p_spice.abs().sum() * 100
            print(f"      d={da:8s} s={sa:8s} n={len(y):5d} power share {share:5.1f}%  |err| median {ey.abs().median():.3f}%  "
                  f"p95 {ey.abs().quantile(.95):.2f}%  bias {ey.median():+.3f}%")
    # contribution of each case to the gate-total error (signed, in % of SPICE gate leakage)
    c = dv.assign(d=(dv.pm - dv.p_spice) / dv.vdd).groupby(key + ["case"]).d.sum().unstack("case")
    rel = c.div(tot_sp, axis=0) * 100
    print("  signed contribution to gate error, median / p95 of |.| (%):",
          {k: (round(rel[k].median(), 3), round(rel[k].abs().quantile(.95), 2)) for k in rel})
