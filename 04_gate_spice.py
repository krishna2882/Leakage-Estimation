"""Tier 3a: SPICE reference for the 18 gate/input cases at 200 corners (evaluation only, never used for training)."""
import json, time
import pandas as pd
from common import *

if __name__ == "__main__":
    t0 = time.time()
    corners = draw_corners()
    json.dump(corners, open(os.path.join(RES, "corners.json"), "w"))
    jobs, meta = [], []
    for kind in GATES:
        for pat in PATTERNS[kind]:
            for ci, c in enumerate(corners):
                nl, vecs = gate_netlist(kind, pat, c)
                jobs.append((nl, vecs, None)); meta.append((kind, pat, ci))
    res = run_many(jobs)
    # retry failures once (convergence), then fail loudly
    bad = [i for i, r in enumerate(res) if r is None]
    print("failed first pass:", len(bad), flush=True)
    for i in bad:
        nl, vecs, _ = jobs[i]
        res[i] = run_ngspice((nl + "\n.options itl1=500 itl2=500 gminsteps=50 srcsteps=50", vecs, None))
    bad = [i for i, r in enumerate(res) if r is None]
    assert not bad, f"{len(bad)} gate sims failed"
    rows = []
    for (kind, pat, ci), r in zip(meta, res):
        c = corners[ci]
        rows.append(dict(gate=kind, pattern=pat, corner=ci, vdd=c["vdd"], temp=c["temp"],
                         spice=gate_total_leak(r, pat), i_vdd_only=-r["i(vdd)"], vout=r["v(out)"],
                         v_x=r.get("v(x)", np.nan), v_y=r.get("v(y)", np.nan), v_n=r.get("v(n)", np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "04_gate_spice.csv"), index=False)
    print(len(df), "rows;", f"{time.time()-t0:.0f}s;", "min leak", df.spice.min(), "max", df.spice.max())
    # example gate netlists (corner 0)
    for kind in GATES:
        for pat in PATTERNS[kind]:
            open(os.path.join(NETS, f"{kind}_{pat}.net"), "w").write(gate_netlist(kind, pat, corners[0])[0].replace(MODEL, REL_MODEL) + "\n")
