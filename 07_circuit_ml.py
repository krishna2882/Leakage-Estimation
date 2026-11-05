"""Tier 4b: hierarchical ML estimate: logic value at every gate input -> look up the gate ML leakage -> sum."""
import json
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def gate_sum(g, vecs, col):
    """per input vector: the parity checker leakage over the corners, as the sum of the 21 gate values in column `col`
    of the gate results (05_gate_results_raw.csv) for the logic pattern each gate sees"""
    tab = {(k, p): grp.sort_values("corner")[col].values for (k, p), grp in g.groupby(["gate", "pattern"])}
    gates, out_net = parity_structure(8)
    out = {}
    for v in vecs:
        val = parity_eval(gates, out_net, [(v >> (7 - i)) & 1 for i in range(8)])
        out[v] = sum(tab[(kind, "".join(str(val[i]) for i in ins))] for kind, ins, o, pre in gates)
    return out


def circuit_columns(g, sp, cols):
    """adds the hierarchical sum of each gate-result column to the circuit SPICE table (one row per vector x corner)"""
    for col, name in cols:
        s = gate_sum(g, sorted(sp.vec.unique()), col)
        sp[name] = [s[v][c] for v, c in zip(sp.vec, sp.corner)]
    return sp


if __name__ == "__main__":
    g = pd.read_csv(os.path.join(RES, "05_gate_results_raw.csv"), dtype={"pattern": str})
    # the circuit SPICE reference does not depend on the models: a dataset-size run (DVD_TAG) reuses the main run's
    # 06_circuit_spice.csv when both use the same corners
    f6 = os.path.join(RES, "06_circuit_spice.csv")
    if not os.path.exists(f6) and TAG:
        assert open(os.path.join(RES, "corners.json")).read() == open(os.path.join(ROOT, "results", "corners.json")).read(), \
            "corners differ from the main run: run 06_circuit_spice.py for this run"
        f6 = os.path.join(ROOT, "results", "06_circuit_spice.csv")
    sp = pd.read_csv(f6)
    # spice_gate_sum: same composition with the SPICE gate leakages, isolates the "ideal inputs, no loading" assumption;
    # ml_selfconsistent (extra): gate models that use their own ML-solved internal node voltages (no SPICE information at all)
    sp = circuit_columns(g, sp, [("ml", "ml"), ("spice", "spice_gate_sum"), ("ml_selfconsistent", "ml_selfconsistent")])
    sp["hw"] = [bin(v).count("1") for v in sp.vec]
    sp.to_csv(os.path.join(RES, "07_circuit_results_raw.csv"), index=False)
    tot = table_metrics(sp.assign(grp="parity8_ALL"), ["grp"])
    byhw = table_metrics(sp, ["hw"])
    ref = table_metrics(sp.assign(grp="SPICE-gate-sum vs SPICE circuit", ml=sp.spice_gate_sum), ["grp"])
    scc = table_metrics(sp.assign(grp="ML with ML-solved node voltages (extra)", ml=sp.ml_selfconsistent), ["grp"])
    tot = pd.concat([tot, ref, scc], ignore_index=True)
    tot.to_csv(os.path.join(RES, "07_circuit_overall.csv"), index=False)
    byhw.to_csv(os.path.join(RES, "07_circuit_by_hamming_weight.csv"), index=False)
    print(tot.to_string()); print(byhw.to_string())
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.loglog(sp.spice, sp.ml, ".", ms=2, alpha=.3)
    lo, hi = sp.spice.min() * .8, sp.spice.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "r-", lw=.8)
    ax.set(xlabel="SPICE leakage (A)", ylabel="ML (hierarchical) leakage (A)", title="8-bit parity checker parity plot"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "circuit_parity.png"), dpi=140); plt.close()
