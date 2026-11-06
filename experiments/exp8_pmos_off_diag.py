"""Why is pmos_off the weakest device model (test median ~1 % vs ~0.2 % for the other three cases, and the largest
share of the gate error in results/08_error_attribution.csv)?
Splits the device power error  P = (Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib  into its three terms, at the transistor bias
points that occur inside the gates (results/08_device_error_at_gate_bias.csv) and on the pmos_off test split, and
shows which terminal current carries the error and where in the bias / process space it sits."""
import sys, os, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, joblib
from common import *
T3 = importlib.import_module("03_train_models")
CASE = "pmos_off"


def terms(x, i_d, i_g, i_b):
    return np.column_stack([(x.vd - x.vs) * i_d, (x.vg - x.vs) * i_g, (x.vb - x.vs) * i_b])


def report(x, sp, ml, label):
    tot = sp.sum(1)
    share = np.abs(sp) / np.abs(sp).sum(1, keepdims=True)
    err = (ml - sp) / np.abs(tot)[:, None] * 100           # contribution of each term to the relative power error
    e = err.sum(1)
    print(f"\n== {label}: n={len(x)}  |power err| median {np.median(np.abs(e)):.3f}%  p95 {np.percentile(np.abs(e), 95):.3f}%")
    print("   term            median |share|   median |err contribution| %   p95 %    median signed %")
    for j, n in enumerate(["(Vd-Vs)Id", "(Vg-Vs)Ig", "(Vb-Vs)Ib"]):
        print(f"   {n:12s}    {np.median(share[:, j]):8.3f}        {np.median(np.abs(err[:, j])):8.3f}           "
              f"{np.percentile(np.abs(err[:, j]), 95):7.3f}   {np.median(err[:, j]):+8.3f}")
    return e


if __name__ == "__main__":
    ch = json.load(open(os.path.join(SAVED, "chosen.json")))
    m = joblib.load(os.path.join(SAVED, f"{CASE}_{ch[CASE]}.joblib"))

    dv = pd.read_csv(os.path.join(RES, "08_device_error_at_gate_bias.csv"), dtype={"pattern": str})
    dv = dv[dv.case == CASE].reset_index(drop=True)
    sp = terms(dv, dv.id_spice, dv.ig_spice, dv.ib_spice); ml = terms(dv, dv.id_ml, dv.ig_ml, dv.ib_ml)
    e = report(dv, sp, ml, "pmos_off at gate bias points")
    dv["abs_err"] = np.abs(e)
    dv["bias"] = np.where(np.abs(dv.vd - dv.vs) / dv.vdd < 0.01, "Vds~0", "Vds>0")
    dv["where"] = dv.gate + " " + dv.pattern + " " + dv.device
    print("\n   by device position (median / p95 |err| %, n):")
    print(dv.groupby(["where", "bias"]).abs_err.agg(["median", lambda s: s.quantile(.95), "size"]).round(3).to_string())

    df = pd.read_csv(os.path.join(DATA, f"{CASE}.csv")); tr, va, te = T3.split(df)
    te = te.reset_index(drop=True); pr = m.predict(te)
    sp = terms(te, te.id, te.ig, te.ib); ml = terms(te, pr[:, 0], pr[:, 1], pr[:, 3])
    e = report(te, sp, ml, "pmos_off test split")
    te["abs_err"] = np.abs(e); te["signed"] = e
    te["vs_r"] = pd.cut(te.vs / te.vdd, [-.01, .05, .3, .7, .95, 1.01]); te["vd_r"] = pd.cut(te.vd / te.vdd, [-.01, .05, .3, .7, .95, 1.01])
    print("\n   median |err| % by Vs/VDD (rows) and Vd/VDD (columns):")
    print(te.pivot_table(index="vs_r", columns="vd_r", values="abs_err", aggfunc="median", observed=False).round(3).to_string())
    for c in ["temp", "vdd", "vth0", "toxe", "ndep", "u0", "xj", "w"]:
        q = pd.qcut(te[c], 5)
        g = te.groupby(q, observed=True).abs_err.median().round(3).values
        print(f"   {c:5s} quintiles: median |err| % {g}   Spearman(|err|) {te[[c, 'abs_err']].corr('spearman').iloc[0, 1]:+.3f}")
    # share of the current in each terminal and how often Id changes sign against the subthreshold direction
    print("\n   fraction of test rows where |Ig| > |Id|:", round(float((np.abs(te.ig) > np.abs(te.id)).mean()), 3),
          "  |Ib| > |Id|:", round(float((np.abs(te.ib) > np.abs(te.id)).mean()), 3))
    print("   Id magnitude decades (log10 |Id|) 1/50/99 %:", np.percentile(np.log10(np.abs(te.id) + 1e-30), [1, 50, 99]).round(2))
    worst = te.sort_values("abs_err", ascending=False).head(10)
    print("\n   worst 10 test rows:")
    print(worst[["vdd", "temp", "vd", "vs", "vth0", "toxe", "id", "ig", "ib", "signed"]].to_string())
