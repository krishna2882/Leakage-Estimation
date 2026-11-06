"""Physics-separated training targets.  In the gate corner the pmos_off drain current is the sum of two mechanisms of
similar size with different physics: subthreshold channel current (set by Vgs, vth0, T) and gate-to-drain overlap
tunnelling Igd (set by Vgd, toxe); elsewhere the gate tunnelling terms dominate.  The plain model learns that mixture
in asinh(I) space and overfits (exp9: train 0.15 %, validation 1.0 %).
Here the same 10k rows are re-simulated to read BSIM4's Igd and Igs (@m1[igd], @m1[igs]) and the network learns six
separately smooth targets, from which the four terminal currents are rebuilt exactly:
    ch_d = id - Igd, ch_s = is - Igs (channel + small junction terms), Igd, Igs, g_rest = ig + s*(Igd + Igs), ib
    id = ch_d + Igd,  is = ch_s + Igs,  ig = g_rest - s*(Igd + Igs)      (s = sign convention, fixed from the data)
Same split, same network and training as the pipeline; evaluated on the validation split and the exp5 rail-bias set
(gate test corners are not used).  usage: python experiments/exp11_component_targets.py <case>"""
import sys, os, time, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *
from mlmodels import CurrentModel
from exp5_rail_weighting import rail_set, OUT
from exp9_pmos_off_capacity import power_err, corner
T3 = importlib.import_module("03_train_models")
CV = ["i(vd)", "i(vg)", "i(vs)", "i(vb)", "@m1[igd]", "@m1[igs]"]


def components(case, df):
    f = os.path.join(OUT, f"comp_{case}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    os.makedirs(WORK, exist_ok=True)
    res = run_many([(single_netlist(case, r), CV, None) for r in df.to_dict("records")])
    assert all(r is not None for r in res)
    c = pd.DataFrame({"igd": [r["@m1[igd]"] for r in res], "igs": [r["@m1[igs]"] for r in res]})
    rep = np.max([np.abs(-np.array([r[v] for r in res]) - df[t].values) / np.maximum(np.abs(df[t].values), 1e-30)
                  for v, t in zip(CV[:4], TARGETS)])
    print(case, "re-simulated terminal currents reproduce the dataset to", rep, flush=True)
    c.to_csv(f, index=False)
    return c


class ComponentModel:
    def __init__(self, typ, s, **kw):
        self.typ, self.s, self.kw = typ, s, kw

    def fit(self, df, igd, igs):
        s = self.s
        T = np.column_stack([df.id - igd, df["is"] - igs, igd, igs, df.ig + s * (igd + igs), df.ib])
        self.m = CurrentModel("mlp", self.typ, **self.kw).fit(df, T)
        return self

    def predict(self, df):
        c = self.m.predict(df); s = self.s
        return np.column_stack([c[:, 0] + c[:, 2], c[:, 4] - s * (c[:, 2] + c[:, 3]), c[:, 1] + c[:, 3], c[:, 5]])


if __name__ == "__main__":
    case = sys.argv[1]
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv"))
    c = components(case, df); df["igd_c"] = c.igd.values; df["igs_c"] = c.igs.values
    # sign of the reported Igd/Igs relative to the into-terminal gate current: ig = -s*(Igd + Igs) + small terms
    s = min((+1, -1), key=lambda s: np.median(np.abs(df.ig + s * (df.igd_c + df.igs_c)) / np.abs(df.ig)))
    print(case, "sign", s, "median |g_rest|/|ig|:", float(np.median(np.abs(df.ig + s * (df.igd_c + df.igs_c)) / np.abs(df.ig))),
          " median |Igd|/|id|:", float(np.median(np.abs(df.igd_c / df.id))), flush=True)
    tr, va, te = T3.split(df); rv = rail_set(case)
    rows = []
    for name in ["terminal", "components"]:
        t0 = time.time()
        kw = dict(seed=SEED, n_jobs=NPROC, width=128, depth=4, epochs=2000, ens=5)
        m = (CurrentModel("mlp", case[0], **kw).fit(tr, tr[TARGETS].values) if name == "terminal"
             else ComponentModel(case[0], s, **kw).fit(tr, tr.igd_c.values, tr.igs_c.values))
        r = dict(case=case, targets=name)
        for nm, d in [("train", tr), ("val", va)]:
            pr = m.predict(d); mt = T3.metrics(pr, d[TARGETS].values); ep, sig = power_err(d, pr); k = corner(d) & sig
            r.update({f"{nm}_med": mt["pooled"]["med"], f"{nm}_p95": mt["pooled"]["p95"],
                      **{f"{nm}_{t}_med": mt[t]["med"] for t in TARGETS},
                      f"{nm}_power_med": float(np.median(ep[sig])), f"{nm}_power_p95": float(np.percentile(ep[sig], 95)),
                      f"{nm}_corner_power_med": float(np.median(ep[k])), f"{nm}_corner_power_p95": float(np.percentile(ep[k], 95))})
        ep, sig = power_err(rv, m.predict(rv)); k = corner(rv) & sig
        r.update(rail_power_med=float(np.median(ep[sig])), rail_power_p95=float(np.percentile(ep[sig], 95)),
                 rail_corner_power_med=float(np.median(ep[k])), rail_corner_power_p95=float(np.percentile(ep[k], 95)),
                 secs=time.time() - t0)
        rows.append(r)
        print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"exp11_{case}.csv"), index=False)
