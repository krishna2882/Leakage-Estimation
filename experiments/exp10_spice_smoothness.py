"""Is the single-FET SPICE output smooth enough to learn to < 1 %?  pmos_off overfits (exp9: train 0.15 %, validation
1.0 %, nmos_off validation 0.24 %), which would also happen if its currents carried small numerical noise.
For training rows of nmos_off and pmos_off (half of them in the corner the gates use: source at the gate's rail),
re-simulates the row (reproducibility) and sweeps one input at a time over a narrow window; the residual of each
current about a smooth polynomial in that input measures non-smoothness.  usage: python experiments/exp10_spice_smoothness.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from common import *

OUT = os.path.join(ROOT, "experiments", "out")
NPTS = 41
SWEEPS = {"vs": ("abs", 5e-3), "vd": ("abs", 5e-3), "temp": ("abs", 1.0), "toxe": ("rel", 5e-3), "vth0": ("rel", 5e-3)}
VECS = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]


def rows_for(case, rng, n=6):
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv"))
    r = df.vs / df.vdd
    corner = df[(r > 0.95) if case.startswith("pmos") else (r < 0.05)]
    return pd.concat([corner.sample(n, random_state=rng), df.sample(n, random_state=rng)]).assign(
        kind=["corner"] * n + ["random"] * n).reset_index(drop=True)


def rough(x, y):
    """median and max |residual| % of y about a degree-4 polynomial in x (sign-safe: relative to max |y| on the window)"""
    t = (x - x.mean()) / (x.max() - x.min())
    res = y - np.polyval(np.polyfit(t, y, 4), t)
    s = np.abs(y).max()
    return np.median(np.abs(res)) / s * 100, np.abs(res).max() / s * 100


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    out = []
    for case in ["nmos_off", "pmos_off"]:
        rows = rows_for(case, 7)
        jobs = [(single_netlist(case, r), VECS, None) for r in rows.to_dict("records")]
        rep = run_many(jobs)
        for i, r in rows.iterrows():
            d = max(abs(-rep[i][v] - r[t]) / max(abs(r[t]), 1e-30) for v, t in zip(VECS, TARGETS))
            out.append(dict(case=case, row=i, kind=r.kind, var="repeat", med_res_pct=d * 100, max_res_pct=d * 100))
        for var, (mode, h) in SWEEPS.items():
            jobs, idx = [], []
            for i, r in rows.iterrows():
                xs = r[var] + np.linspace(-h, h, NPTS) * (abs(r[var]) if mode == "rel" else 1)
                for x in xs:
                    jobs.append((single_netlist(case, {**r.to_dict(), var: x}), VECS, None)); idx.append((i, x))
            res = run_many(jobs)
            for i, r in rows.iterrows():
                sel = [k for k, (j, _) in enumerate(idx) if j == i]
                x = np.array([idx[k][1] for k in sel])
                cur = np.array([[-res[k][v] for v in VECS] for k in sel])
                vd = x if var == "vd" else r.vd; vs = x if var == "vs" else r.vs
                p = (vd - vs) * cur[:, 0] + (r.vg - vs) * cur[:, 1] + (r.vb - vs) * cur[:, 3]   # device power
                for name, y in [("id", cur[:, 0]), ("ig", cur[:, 1]), ("power", p)]:
                    m, mx = rough(x, y)
                    out.append(dict(case=case, row=i, kind=r.kind, var=f"{var}:{name}", med_res_pct=m, max_res_pct=mx))
        print(case, "done", flush=True)
    df = pd.DataFrame(out); df.to_csv(os.path.join(OUT, "exp10_spice_smoothness.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["var", "case", "kind"]).max_res_pct.agg(["median", "max"]).unstack(["case", "kind"]).round(5).to_string())
