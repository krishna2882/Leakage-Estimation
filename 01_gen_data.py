"""Tier 1: draw samples (Section 4), simulate single FETs in ngspice, write the 4 CSVs + example netlists."""
import sys, time
import pandas as pd
from common import *

N = int(os.environ.get("DVD_N", sys.argv[1] if len(sys.argv) > 1 else 10000))
GATE = {"nmos_on": ("vdd", 0), "nmos_off": (0, 0), "pmos_on": (0, "vdd"), "pmos_off": ("vdd", "vdd")}  # (vg, vb)


def draw(case, n, rng):
    suf = case[0]
    vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM, n)          # drawn first
    df = pd.DataFrame(dict(vdd=vdd,
                           temp=rng.uniform(-40, 125, n),
                           w=rng.uniform(1 * LMIN, 8 * LMIN, n),
                           vd=rng.uniform(0, 1, n) * vdd,
                           vs=rng.uniform(0, 1, n) * vdd))
    vg, vb = GATE[case]
    df["vg"] = vdd if vg == "vdd" else 0.0
    df["vb"] = vdd if vb == "vdd" else 0.0
    for k in PKEYS:
        df[k] = NOM[suf][k] * rng.uniform(0.9, 1.1, n)
    return df[FEATS]


def simulate(case, df):
    jobs = [(single_netlist(case, r), ["i(vd)", "i(vg)", "i(vs)", "i(vb)"], None)
            for r in df.to_dict("records")]
    res = run_many(jobs)
    return res


if __name__ == "__main__":
    for ci, case in enumerate(CASES):
        t0 = time.time()
        rng = np.random.default_rng(SEED + ci)
        df = draw(case, N, rng)
        res = simulate(case, df)
        bad = [i for i, r in enumerate(res) if r is None]
        tries = 0
        while bad and tries < 3:                      # redraw any non-converged sample
            tries += 1
            new = draw(case, len(bad), rng)
            nr = simulate(case, new)
            for j, i in enumerate(bad):
                if nr[j] is not None:
                    df.iloc[i] = new.iloc[j].values
                    res[i] = nr[j]
            bad = [i for i, r in enumerate(res) if r is None]
        assert not bad, f"{case}: {len(bad)} samples failed"
        for t, v in zip(TARGETS, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]):
            df[t] = [-r[v] for r in res]              # current INTO the terminal
        df.to_csv(os.path.join(DATA, f"{case}.csv"), index=False)
        print(case, len(df), "rows", f"{time.time()-t0:.0f}s", "redrawn:", tries)
        # example netlist = first sample (main run only: netlists/ is shared by the dataset-size runs)
        if not TAG:
            open(os.path.join(NETS, f"single_{case}.net"), "w").write(single_netlist(case, df.iloc[0].to_dict()).replace(MODEL, REL_MODEL) + "\n")
