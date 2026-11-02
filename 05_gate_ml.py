"""Tier 3b: ML gate leakage from SPICE-recorded node voltages (spec step 4), metrics, parity plot.
Extra (clearly separated): ML-solved internal-node voltage by KCL on the series node x / y."""
import json
import pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def load_models():
    ch = json.load(open(os.path.join(SAVED, "chosen.json")))
    return {c: joblib.load(os.path.join(SAVED, f"{c}_{k}.joblib")) for c, k in ch.items()}


def device_currents(models, dev, volts, c, n):
    """currents into (d,g,s,b) of one device at all corners at once (arrays of length n)."""
    name, typ, d, g, s, b, w = dev
    V = lambda nd: volts[nd]
    vdd = np.array([cc["vdd"] for cc in c]); T = np.array([cc["temp"] for cc in c])
    vg = V(g)
    on = (vg > vdd / 2) if typ == "nmos" else (vg < vdd / 2)
    pr = {k: np.array([cc[typ[0]][k] for cc in c]) for k in PKEYS}
    df = pd.DataFrame(dict(vdd=vdd, temp=T, w=w, vd=V(d), vs=V(s), vg=vg, vb=V(b), **pr))
    out = np.zeros((n, 4))
    for flag in (True, False):
        m = on == flag
        if m.any():
            case = f"{typ}_{'on' if flag else 'off'}"
            out[m] = models[case].predict(df[m])
    return out


def node_volts(kind, pat, c, spice):
    n = len(c)
    vdd = np.array([cc["vdd"] for cc in c])
    v = {"vdd": vdd, "0": np.zeros(n), "out": spice["vout"].values}
    for nm, bit in zip(["A", "B"], pat):
        v[nm] = vdd if bit == "1" else np.zeros(n)
    for nd, col in (("x", "v_x"), ("y", "v_y"), ("n", "v_n")):
        if nd in internal_nodes(kind):
            v[nd] = spice[col].values
    return v


def ml_leak(models, kind, pat, c, volts):
    """Leakage = static power / VDD.  Power is summed per device as (Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib
    (Tellegen; verified exact against SPICE terminal currents in experiments/exp1_power_formula.py).
    An ON device with Vds ~ 0 then contributes its channel current times a tiny Vds, so the error of the
    ill-conditioned near-zero channel current no longer dominates.  `term` keeps the naive sum of the
    currents entering the VDD-level terminals, for comparison only."""
    n = len(c)
    devs = gate_devices(kind, ["A", "B"][:len(pat)], "out")
    hi = {"vdd"} | {nm for nm, bit in zip(["A", "B"], pat) if bit == "1"}
    per = {}
    vdd = volts["vdd"]
    pw = np.zeros(n); term = np.zeros(n)
    for dev in devs:
        I = device_currents(models, dev, volts, c, n)
        per[dev[0]] = I
        vd, vg, vs, vb = (volts[x] for x in dev[2:6])
        pw += (vd - vs) * I[:, 0] + (vg - vs) * I[:, 1] + (vb - vs) * I[:, 3]
        for j, nd in enumerate(dev[2:6]):
            if nd in hi:
                term += I[:, j]
    return pw / vdd, term, per, devs


def node_current(models, devs, c, vv, node):
    """net current flowing from the devices INTO `node` = -(sum of currents into the device terminals on that node)."""
    n = len(c); tot = np.zeros(n)
    for dev in devs:
        if node in dev[2:6]:
            I = device_currents(models, dev, vv, c, n)
            for j, nd in enumerate(dev[2:6]):
                if nd == node:
                    tot -= I[:, j]
    return tot


def solve_node(models, devs, c, vv, node, ngrid=41, nbis=30):
    """KCL root of one internal node with the ML models only: grid scan over [-2 %, 102 %] of VDD,
    bracket the sign change with the smallest |f|, then bisection.  No sign change -> grid point with the smallest |f|."""
    vdd = vv["vdd"]; n = len(c)
    grid = np.linspace(-0.02, 1.02, ngrid)[:, None] * vdd[None, :]
    F = np.array([node_current(models, devs, c, {**vv, node: g}, node) for g in grid])
    lo = np.empty(n); hi = np.empty(n); best = grid[np.argmin(np.abs(F), 0), np.arange(n)]
    found = np.zeros(n, bool)
    for k in range(ngrid - 1):
        ch = (np.sign(F[k]) != np.sign(F[k + 1])) & ~found
        lo[ch] = grid[k, ch]; hi[ch] = grid[k + 1, ch]; found |= ch
    lo[~found] = hi[~found] = best[~found]
    flo = node_current(models, devs, c, {**vv, node: lo}, node)
    for _ in range(nbis):
        mid = 0.5 * (lo + hi); fm = node_current(models, devs, c, {**vv, node: mid}, node)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); flo = np.where(same, fm, flo); hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi), found


def solve_internal(models, kind, pat, c, volts, sweeps=3):
    """ML-only internal-node voltages (x / y, and the NAND/NOR output n inside AND2/OR2), Gauss-Seidel over nodes.
    Start: x/y and n from the logic levels; out is kept at the SPICE value (spec: output voltage from SPICE)."""
    devs = gate_devices(kind, ["A", "B"][:len(pat)], "out")
    nodes = internal_nodes(kind)
    vv = dict(volts); vdd = volts["vdd"]
    bits = [int(b) for b in pat]
    if "n" in nodes:
        vv["n"] = gate_logic("NAND2" if kind == "AND2" else "NOR2", bits) * vdd
    for nd in nodes:
        if nd in ("x", "y"):
            vv[nd] = 0.5 * vdd
    found = {}
    for _ in range(sweeps if len(nodes) > 1 else 1):
        for nd in nodes:
            vv[nd], found[nd] = solve_node(models, devs, c, vv, nd)
    return {nd: vv[nd] for nd in nodes}, found, vv


if __name__ == "__main__":
    models = load_models()
    corners = json.load(open(os.path.join(RES, "corners.json")))
    sp = pd.read_csv(os.path.join(RES, "04_gate_spice.csv"), dtype={"pattern": str})
    out = []
    for kind in GATES:
        for pat in PATTERNS[kind]:
            s = sp[(sp.gate == kind) & (sp.pattern == pat)].sort_values("corner")
            assert len(s) == len(corners)
            volts = node_volts(kind, pat, corners, s)
            tot, term, per, devs = ml_leak(models, kind, pat, corners, volts)
            d = s.copy()
            d["ml"] = tot
            d["ml_terminal_sum"] = term
            # spec: internal node voltages "from both": ML-solved values next to the SPICE ones,
            # plus the leakage the ML models give with their own node voltages (fully ML, for comparison)
            if internal_nodes(kind):
                vml, found, vv = solve_internal(models, kind, pat, corners, volts)
                for nd, v in vml.items():
                    d[f"ml_v_{nd}"] = v; d[f"ml_v_{nd}_bracketed"] = found[nd]
                d["ml_selfconsistent"] = ml_leak(models, kind, pat, corners, vv)[0]
            else:
                d["ml_selfconsistent"] = tot
            out.append(d)
    df = pd.concat(out).rename(columns={"spice": "spice"})
    # negative ML leakage (non-physical) is kept and counted, never clipped
    df.to_csv(os.path.join(RES, "05_gate_results_raw.csv"), index=False)
    df["case"] = df.gate + "_" + df.pattern
    tab = table_metrics(df.assign(spice=df.spice), ["gate", "pattern"])
    tab.to_csv(os.path.join(RES, "05_gate_table.csv"), index=False)
    allm = table_metrics(df.assign(g="ALL"), ["g"])
    allm.to_csv(os.path.join(RES, "05_gate_overall.csv"), index=False)
    print(tab.to_string()); print(allm.to_string())
    print("negative ML leakage count:", int((df.ml <= 0).sum()))
    # internal node voltage comparison (bonus)
    rows = []
    for (g, p), grp in df.groupby(["gate", "pattern"], sort=False):
        for nd in internal_nodes(g):
            e = (grp[f"ml_v_{nd}"] - grp[f"v_{nd}"]).abs() * 1e3
            rows.append(dict(gate=g, pattern=p, node=nd, spice_median_V=grp[f"v_{nd}"].median(), ml_median_V=grp[f"ml_v_{nd}"].median(),
                             abs_err_median_mV=e.median(), abs_err_p95_mV=e.quantile(.95), bracketed_frac=grp[f"ml_v_{nd}_bracketed"].mean()))
    nv = pd.DataFrame(rows); nv.to_csv(os.path.join(RES, "05_node_voltages.csv"), index=False)
    print(nv.round(4).to_string())
    sc = table_metrics(df.assign(g="ALL (ML node voltages)", ml=df.ml_selfconsistent), ["g"])
    sc.to_csv(os.path.join(RES, "05_gate_overall_selfconsistent.csv"), index=False)
    print(sc.to_string())
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for g, gg in df.groupby("gate"):
        ax.loglog(gg.spice, gg.ml.clip(lower=1e-15), ".", ms=3, alpha=.5, label=g)
    lo, hi = df.spice.min() * .8, df.spice.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "k-", lw=.8); ax.set(xlabel="SPICE leakage (A)", ylabel="ML leakage (A)", title="Gate-level parity (18 cases x 200 corners)")
    ax.legend(markerscale=4); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "gate_parity.png"), dpi=140); plt.close()
