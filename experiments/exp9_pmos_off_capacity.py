"""Is the pmos_off device model (test median ~1 %, 4-5x the other cases) limited by capacity / optimisation or by data?
Same 10k data and split as the pipeline; reports train vs validation error (a large gap = data-limited, no gap =
underfit) for a few network sizes, and the power error in the corner gates actually use (Vs = Vg = Vb = VDD, i.e.
Vs/VDD > 0.95 for PMOS, < 0.05 for NMOS) on the validation split and on the independent rail-bias set of exp5 (gate test corners not used).
usage: python experiments/exp9_pmos_off_capacity.py <variant> [case]"""
import sys, os, time, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *
from mlmodels import CurrentModel
from exp5_rail_weighting import rail_set, OUT
T3 = importlib.import_module("03_train_models")

VARIANTS = {"base": dict(width=128, depth=4, epochs=2000), "wide": dict(width=256, depth=4, epochs=2000),
            "deep": dict(width=128, depth=6, epochs=2000), "long": dict(width=128, depth=4, epochs=6000)}


def power_err(df, pred):
    """relative error % of (Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib for every row, and the rows where that power is significant"""
    dv = lambda a: (df[a] - df.vs).values
    ps = dv("vd") * df.id.values + dv("vg") * df.ig.values + dv("vb") * df.ib.values
    pm = dv("vd") * pred[:, 0] + dv("vg") * pred[:, 1] + dv("vb") * pred[:, 3]      # columns id, ig, is, ib
    sig = np.abs(ps) > 1e-3 * np.abs(ps).max()
    return np.abs(pm - ps) / np.maximum(np.abs(ps), 1e-30) * 100, sig


def corner(df):                                   # source terminal at the rail the gate ties it (and the body) to
    r = df.vs.values / df.vdd.values
    return r > 0.95 if df.vb.iloc[0] > 0 else r < 0.05


if __name__ == "__main__":
    var = sys.argv[1]; case = sys.argv[2] if len(sys.argv) > 2 else "pmos_off"
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, te = T3.split(df)
    rv = rail_set(case)
    t0 = time.time()
    m = CurrentModel("mlp", case[0], seed=SEED, n_jobs=NPROC, ens=5, **VARIANTS[var]).fit(tr, tr[TARGETS].values)
    r = dict(case=case, variant=var, **VARIANTS[var])
    for name, d in [("train", tr), ("val", va)]:
        pr = m.predict(d); mt = T3.metrics(pr, d[TARGETS].values); ep, sig = power_err(d, pr); k = corner(d) & sig
        r.update({f"{name}_med": mt["pooled"]["med"], f"{name}_p95": mt["pooled"]["p95"],
                  f"{name}_id_med": mt["id"]["med"], f"{name}_ig_med": mt["ig"]["med"],
                  f"{name}_power_med": float(np.median(ep[sig])), f"{name}_corner_power_med": float(np.median(ep[k])),
                  f"{name}_corner_power_p95": float(np.percentile(ep[k], 95)), f"{name}_corner_n": int(k.sum())})
    ep, sig = power_err(rv, m.predict(rv)); k = corner(rv) & sig
    r.update(rail_power_med=float(np.median(ep[sig])), rail_corner_power_med=float(np.median(ep[k])),
             rail_corner_power_p95=float(np.percentile(ep[k], 95)), secs=time.time() - t0)
    print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, indent=1), flush=True)
    pd.DataFrame([r]).to_csv(os.path.join(OUT, f"exp9_{case}_{var}.csv"), index=False)
