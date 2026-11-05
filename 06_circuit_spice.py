"""Tier 4a: full transistor-level 8-bit parity checker in SPICE, 256 input vectors x 200 corners."""
import json, time, itertools
import pandas as pd
from common import *

if __name__ == "__main__":
    t0 = time.time()
    corners = json.load(open(os.path.join(RES, "corners.json")))
    gates, out_net = parity_structure(8)
    nv = int(os.environ.get("DVD_NVEC", 256))
    vecs_all = [[(v >> (7 - i)) & 1 for i in range(8)] for v in range(256)]
    sel = np.random.default_rng(SEED).permutation(256)[:nv] if nv < 256 else np.arange(256)
    vecs = [vecs_all[i] for i in sorted(sel)]
    jobs, meta = [], []
    for vi, vec in zip(sorted(sel), vecs):
        for ci, c in enumerate(corners):
            nl, pv = parity_netlist(vec, c, gates, out_net)
            jobs.append((nl, pv, None)); meta.append((int(vi), vec, ci))
    res = run_many(jobs, chunksize=16)
    bad = [i for i, r in enumerate(res) if r is None]
    print("failed first pass:", len(bad), flush=True)
    for i in bad:
        nl, pv, _ = jobs[i]
        res[i] = run_ngspice((nl + "\n.options itl1=500 itl2=500 gminsteps=50 srcsteps=50", pv, None))
    bad = [i for i, r in enumerate(res) if r is None]
    assert not bad, f"{len(bad)} circuit sims failed"
    rows = []
    wrong = 0
    for (vi, vec, ci), r in zip(meta, res):
        exp = sum(vec) % 2
        ok = abs(r["v(out)"] - exp * corners[ci]["vdd"]) < 0.1 * corners[ci]["vdd"]
        wrong += (not ok)
        rows.append(dict(vec=vi, corner=ci, spice=parity_total_leak(r, vec), vout=r["v(out)"], logic_ok=ok))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "06_circuit_spice.csv"), index=False)
    print(len(df), "rows; logic mismatches:", wrong, f"{time.time()-t0:.0f}s")
    gates_lines = parity_netlist(vecs[0], corners[0], gates, out_net)[0]
    open(os.path.join(NETS, "parity8_example.net"), "w").write(gates_lines.replace(MODEL, REL_MODEL) + "\n")
