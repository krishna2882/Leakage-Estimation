"""Shared constants, netlist builders and the ngspice runner.

Assignment: 32nm_MGK, nominal VDD = 0.9 V, 8-bit parity checker.
Current convention everywhere: I_x = current flowing INTO terminal x of the
device (x = d, g, s, b).  From ngspice, I_x = -I(V_x) because V_x delivers
current into the device node.
"""
import os, re, subprocess, tempfile, itertools
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "32nm_MGK_param.pm")   # parametrised copy of 32nm_MGK.pm
REL_MODEL = "../models/32nm_MGK_param.pm"                   # path used in the saved example netlists
TAG = os.environ.get("DVD_TAG", "")          # e.g. "_40k" -> separate data/results folders
DATA = os.path.join(ROOT, "data" + TAG)
RES = os.path.join(ROOT, "results" + TAG)
PLOTS = os.path.join(ROOT, "plots" + TAG)
SAVED = os.path.join(ROOT, "saved_models" + TAG)
NETS = os.path.join(ROOT, "netlists")
WORK = os.path.join(ROOT, "work" + TAG)
for _d in (DATA, RES, PLOTS, SAVED, NETS, WORK):
    os.makedirs(_d, exist_ok=True)

NODE = 32e-9
LMIN = NODE
VDD_NOM = 0.9
LDIFF = 2 * LMIN                      # diffusion length for AD/AS/PD/PS
OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
SEED = 2024
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

# nominal process values from the model file (read, not typed from memory: see 00_setup_check.py)
NOM = {
    "n": dict(vth0=0.3558, toxe=7.5e-10, ndep=8.7e18, u0=0.0238, xj=1e-8),
    "p": dict(vth0=-0.24123, toxe=7.7e-10, ndep=3.5e18, u0=0.00306, xj=1.008e-8),
}
PKEYS = ["vth0", "toxe", "ndep", "u0", "xj"]
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
FEATS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb"] + PKEYS
TARGETS = ["id", "ig", "is", "ib"]

# gate sizing (multiples of Lmin) -- identical in SPICE netlists and ML equations
WU = 2 * LMIN
GATE_W = {
    "INV":   dict(n=1 * WU, p=2 * WU),
    "NAND2": dict(n=2 * WU, p=2 * WU),
    "NOR2":  dict(n=1 * WU, p=4 * WU),
}


def params_line(pn, pp):
    s = " ".join(f"{k}_n={pn[k]:.10g}" for k in PKEYS) + " " + " ".join(f"{k}_p={pp[k]:.10g}" for k in PKEYS)
    return ".param " + s


def mos_line(name, typ, d, g, s, b, w):
    ad = w * LDIFF
    pd = 2 * (w + LDIFF)
    return f"{name} {d} {g} {s} {b} {typ} W={w:.6g} L={LMIN:.6g} AD={ad:.6g} AS={ad:.6g} PD={pd:.6g} PS={pd:.6g}"


def single_netlist(case, row, model_path=MODEL):
    """row: dict with vdd,temp,w,vd,vs,vg,vb and process params (suffix _n / _p chosen by case)."""
    typ = "nmos" if case.startswith("nmos") else "pmos"
    suf = typ[0]
    pn = {k: (row[k] if suf == "n" else NOM["n"][k]) for k in PKEYS}
    pp = {k: (row[k] if suf == "p" else NOM["p"][k]) for k in PKEYS}
    lines = [f"single {typ.upper()} {case.split('_')[1].upper()}",
             OPTIONS, params_line(pn, pp), f".include \"{model_path}\"", f".temp {row['temp']:.6g}",
             mos_line("M1", typ, "d", "g", "s", "b", row["w"]),
             f"Vd d 0 dc {row['vd']:.10g}", f"Vg g 0 dc {row['vg']:.10g}",
             f"Vs s 0 dc {row['vs']:.10g}", f"Vb b 0 dc {row['vb']:.10g}",
             ".op"]
    return "\n".join(lines)


def run_ngspice(args):
    """args = (netlist_text, [vectors to print]); returns dict vec->float or None on failure."""
    text, vecs, probes = args
    ctl = ["", ".control", "set numdgt=15", "set noaskquit", "op"]
    for v in vecs:
        ctl.append(f"print {v}")
    ctl += [".endc", ".end"]
    # strip any .op line; the .control block runs it
    body = "\n".join(l for l in text.split("\n") if l.strip().lower() != ".op")
    with tempfile.NamedTemporaryFile("w", suffix=".cir", dir=WORK, delete=False) as f:
        f.write(body + "\n" + "\n".join(ctl) + "\n")
        fn = f.name
    try:
        r = subprocess.run(["ngspice", "-b", fn], capture_output=True, text=True, timeout=300)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ""
    finally:
        os.remove(fn)
    res = {}
    for v in vecs:
        m = re.search(r"^" + re.escape(v) + r"\s*=\s*([-+0-9.eE]+)\s*$", out, re.M | re.I)
        if not m:
            return None
        res[v] = float(m.group(1))
    return res


def run_many(jobs, nproc=NPROC, chunksize=8):
    with Pool(nproc) as p:
        return p.map(run_ngspice, jobs, chunksize=chunksize)


def draw_process(rng, suf):
    return {k: NOM[suf][k] * rng.uniform(0.9, 1.1) for k in PKEYS}


# ---------------------------------------------------------------- gates / circuit
NCORNERS = 200
PATTERNS = {"INV": ["0", "1"], "NAND2": ["00", "01", "10", "11"], "NOR2": ["00", "01", "10", "11"],
            "AND2": ["00", "01", "10", "11"], "OR2": ["00", "01", "10", "11"]}
GATES = ["INV", "NAND2", "NOR2", "AND2", "OR2"]


def gate_devices(kind, ins, out, pre=""):
    """Transistors of a static CMOS gate as (name, type, d, g, s, b, width).  Nodes: 'vdd', '0'."""
    def nand(a, b, o):
        x = pre + "x"; w = GATE_W["NAND2"]
        return [("M" + pre + "P1", "pmos", o, a, "vdd", "vdd", w["p"]), ("M" + pre + "P2", "pmos", o, b, "vdd", "vdd", w["p"]),
                ("M" + pre + "N1", "nmos", o, a, x, "0", w["n"]), ("M" + pre + "N2", "nmos", x, b, "0", "0", w["n"])]

    def nor(a, b, o):
        y = pre + "y"; w = GATE_W["NOR2"]
        return [("M" + pre + "P1", "pmos", y, a, "vdd", "vdd", w["p"]), ("M" + pre + "P2", "pmos", o, b, y, "vdd", w["p"]),
                ("M" + pre + "N1", "nmos", o, a, "0", "0", w["n"]), ("M" + pre + "N2", "nmos", o, b, "0", "0", w["n"])]

    def inv(a, o, p=""):
        w = GATE_W["INV"]
        return [("M" + pre + p + "PI", "pmos", o, a, "vdd", "vdd", w["p"]), ("M" + pre + p + "NI", "nmos", o, a, "0", "0", w["n"])]

    if kind == "INV":
        return inv(ins[0], out)
    if kind == "NAND2":
        return nand(ins[0], ins[1], out)
    if kind == "NOR2":
        return nor(ins[0], ins[1], out)
    n = pre + "n"
    if kind == "AND2":
        return nand(ins[0], ins[1], n) + inv(n, out)
    if kind == "OR2":
        return nor(ins[0], ins[1], n) + inv(n, out)
    raise ValueError(kind)


def gate_logic(kind, bits):
    a = bits[0]; b = bits[1] if len(bits) > 1 else 0
    return {"INV": 1 - a, "NAND2": 1 - (a & b), "NOR2": 1 - (a | b), "AND2": a & b, "OR2": a | b}[kind]


def internal_nodes(kind, pre=""):
    return {"INV": [], "NAND2": [pre + "x"], "NOR2": [pre + "y"], "AND2": [pre + "x", pre + "n"],
            "OR2": [pre + "y", pre + "n"]}[kind]


def draw_corners(n=int(os.environ.get("DVD_NCORNERS", NCORNERS)), seed=SEED + 100):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM)
        temp = rng.uniform(-40, 125)
        out.append(dict(vdd=vdd, temp=temp, n=draw_process(rng, "n"), p=draw_process(rng, "p")))
    return out


def netlist_header(title, c):
    return [title, OPTIONS, params_line(c["n"], c["p"]), f".include \"{MODEL}\"", f".temp {c['temp']:.8g}"]


def devices_to_lines(devs):
    return [mos_line(n, t, d, g, s, b, w) for (n, t, d, g, s, b, w) in devs]


def gate_netlist(kind, pattern, c):
    ins = ["A", "B"][:len(pattern)]
    lines = netlist_header(f"{kind} {pattern}", c)
    lines += devices_to_lines(gate_devices(kind, ins, "out"))
    lines.append(f"Vdd vdd 0 dc {c['vdd']:.10g}")
    for nm, bit in zip(ins, pattern):
        lines.append(f"V{nm} {nm} 0 dc {c['vdd'] if bit == '1' else 0:.10g}")
    bits = [int(x) for x in pattern]
    o = gate_logic(kind, bits)
    lines.append(f".nodeset v(out)={o * c['vdd']:.6g}")
    if kind in ("AND2", "OR2"):
        base = "NAND2" if kind == "AND2" else "NOR2"
        lines.append(f".nodeset v(n)={gate_logic(base, bits) * c['vdd']:.6g}")
    lines.append(".op")
    vecs = ["i(vdd)"] + [f"i(v{x.lower()})" for x in ins] + ["v(out)"] + [f"v({n})" for n in internal_nodes(kind)]
    return "\n".join(lines), vecs


def gate_total_leak(res, pattern):
    """Power-equivalent leakage: current delivered by every source held at VDD (supply + inputs at 1)."""
    tot = -res["i(vdd)"]
    for nm, bit in zip(["a", "b"], pattern):
        if bit == "1":
            tot += -res[f"i(v{nm})"]
    return tot


# parity checker: XOR(a,b) = AND2(OR2(a,b), NAND2(a,b)); 8 inputs -> 7 XORs (balanced tree), 21 gates, 112 transistors
def parity_structure(nbits=8):
    gates = []   # (kind, [in nets], out net, prefix)
    level = [f"in{i}" for i in range(nbits)]
    k = 0
    while len(level) > 1:
        nxt = []
        for a, b in zip(level[0::2], level[1::2]):
            o = f"x{k}_or"; nd = f"x{k}_nd"; xo = f"x{k}_o"
            gates += [("OR2", [a, b], o, f"g{k}o_"), ("NAND2", [a, b], nd, f"g{k}n_"), ("AND2", [o, nd], xo, f"g{k}a_")]
            nxt.append(xo); k += 1
        level = nxt
    return gates, level[0]


def parity_eval(gates, out_net, vec):
    """vec: list of bits.  Returns net->logic dict."""
    val = {f"in{i}": b for i, b in enumerate(vec)}
    for kind, ins, o, pre in gates:
        val[o] = gate_logic(kind, [val[i] for i in ins])
    return val


def parity_netlist(vec, c, gates, out_net):
    lines = netlist_header("parity8 " + "".join(map(str, vec)), c)
    for kind, ins, o, pre in gates:
        lines += devices_to_lines(gate_devices(kind, ins, o if o != out_net else "out", pre))
    lines.append(f"Vdd vdd 0 dc {c['vdd']:.10g}")
    for i, b in enumerate(vec):
        lines.append(f"Vi{i} in{i} 0 dc {c['vdd'] if b else 0:.10g}")
    val = parity_eval(gates, out_net, vec)
    for kind, ins, o, pre in gates:
        nm = "out" if o == out_net else o
        lines.append(f".nodeset v({nm})={val[o] * c['vdd']:.6g}")
        if kind in ("AND2", "OR2"):
            base = "NAND2" if kind == "AND2" else "NOR2"
            lines.append(f".nodeset v({pre}n)={gate_logic(base, [val[i] for i in ins]) * c['vdd']:.6g}")
    lines.append(".op")
    vecs = ["i(vdd)"] + [f"i(vi{i})" for i in range(len(vec))] + ["v(out)"]
    return "\n".join(lines), vecs


def parity_total_leak(res, vec):
    tot = -res["i(vdd)"]
    for i, b in enumerate(vec):
        if b:
            tot += -res[f"i(vi{i})"]
    return tot


def table_metrics(df, keys):
    """df has spice, ml columns. Errors are relative to SPICE, in percent."""
    import pandas as pd
    rows = []
    for k, g in df.groupby(keys, sort=False):
        s, m = g["spice"].values, g["ml"].values
        e = (m - s) / s * 100.0
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(keys, k), n=len(g), spice_median=np.median(s), ml_median=np.median(m),
                         bias_pct=np.median(e), median_err_pct=np.median(np.abs(e)),
                         p95_err_pct=np.percentile(np.abs(e), 95), mean_accuracy_pct=100 - np.mean(np.abs(e))))
    return pd.DataFrame(rows)
