"""Check: gate leakage (current delivered by sources at VDD) ==
sum over devices of [(Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib] / VDD, using SPICE terminal currents
measured with 0 V ammeter sources on every device terminal."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

worst = 0
for kind in GATES:
    for pat in PATTERNS[kind]:
        es = []
        for c in draw_corners(5):
            nl, vecs = gate_netlist(kind, pat, c)
            devs = gate_devices(kind, ["A", "B"][:len(pat)], "out")
            dev_lines = set(devices_to_lines(devs))
            lines = [l for l in nl.split("\n") if l not in dev_lines]
            extra = []
            for (n, t, d, g, s, b, w) in devs:
                pins = []
                for term, node in zip("dgsb", (d, g, s, b)):
                    pin = f"{n}_{term}".lower()
                    lines.insert(-1, f"Vam_{pin} {node} {pin} 0")      # current node -> pin = into terminal
                    extra.append(f"i(vam_{pin})"); pins.append(pin)
                lines.insert(-1, mos_line(n, t, *pins, w))
            nodes = sorted({x for dv in devs for x in dv[2:6]} - {"0"})
            extra += [f"v({x.lower()})" for x in nodes]
            r = run_ngspice(("\n".join(lines), vecs + extra, None))
            assert r is not None, (kind, pat)
            V = lambda x: 0.0 if x == "0" else r[f"v({x.lower()})"]
            P = 0.0
            for (n, t, d, g, s, b, w) in devs:
                I = {term: r[f"i(vam_{n.lower()}_{term})"] for term in "dgsb"}
                P += (V(d) - V(s)) * I["d"] + (V(g) - V(s)) * I["g"] + (V(b) - V(s)) * I["b"]
            sp = gate_total_leak(r, pat)
            es.append(abs(P / c["vdd"] - sp) / sp)
        worst = max(worst, max(es))
        print(f"{kind:5s} {pat}: last corner spice {sp:.4e} A, power-formula {P / c['vdd']:.4e} A, max rel diff over 5 corners {max(es):.2e}", flush=True)
print("WORST rel diff:", worst)
