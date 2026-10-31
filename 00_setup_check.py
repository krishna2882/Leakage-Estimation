"""Tier 0: verify model file, nominal values, parametrised-model fidelity, simulator version."""
import re, subprocess
from common import *
src = open(os.path.join(ROOT, "models", "32nm_MGK.pm")).read()
i = src.index(".model  pmos")
blocks = {"n": src[:i], "p": src[i:]}
log = []
ver = subprocess.run(["ngspice", "-v"], capture_output=True, text=True).stdout.split("\n")[1:3]
log.append("simulator: " + " ".join(s.strip("* ").strip() for s in ver))
for s in "np":
    for k in PKEYS:
        m = re.findall(r"(?<![A-Za-z0-9_])" + k + r"\s*=\s*([-0-9.eE+]+)", blocks[s])
        assert len(m) == 1 and abs(float(m[0]) - NOM[s][k]) <= 1e-12 * abs(NOM[s][k]), (s, k, m)
        log.append(f"nominal {s}.{k} = {m[0]} (matches common.NOM)")
# fidelity: parametrised model at nominal == original model file
V = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]
for case, vd, vg, vs, vb, suf in [("nmos_on", .9, .9, 0, 0, "n"), ("nmos_off", .9, 0, 0, 0, "n"),
                                  ("pmos_on", 0, 0, .9, .9, "p"), ("pmos_off", 0, .9, .9, .9, "p")]:
    row = dict(vdd=.9, temp=85, w=96e-9, vd=vd, vs=vs, vg=vg, vb=vb, **NOM[suf])
    a = run_ngspice((single_netlist(case, row), V, None))
    b = run_ngspice((single_netlist(case, row, model_path=os.path.join(ROOT, "models", "32nm_MGK.pm")), V, None))
    worst = max(abs(a[v] - b[v]) / max(abs(b[v]), 1e-18) for v in V)
    log.append(f"{case}: param-model vs original-model max rel diff = {worst:.2e}")
    assert worst < 1e-6
log.append("Lmin = 32 nm, W in [32,256] nm, VDD nominal 0.9 V (0.81..0.99 V)")
open(os.path.join(RES, "00_setup.txt"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
