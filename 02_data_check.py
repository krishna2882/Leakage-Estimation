"""Tier 1b: Kirchhoff check per case + three trend plots (controlled SPICE sweeps + dataset scatter)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

V = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]
lines = ["Kirchhoff check: KCL = id+ig+is+ib ; rel = |KCL| / (|id|+|ig|+|is|+|ib|)"]
for c in CASES:
    d = pd.read_csv(os.path.join(DATA, f"{c}.csv"))
    kcl = d[TARGETS].sum(axis=1).abs()
    rel = kcl / d[TARGETS].abs().sum(axis=1)
    lines.append(f"{c}: n={len(d)}  max|KCL|={kcl.max():.3e} A  median|KCL|={kcl.median():.3e} A  "
                 f"max rel={rel.max():.3e}  median rel={rel.median():.3e}  frac(rel<1e-3)={(rel<1e-3).mean():.4f}")
open(os.path.join(RES, "02_kirchhoff.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))


def sweep(case, key, vals, base_over, target):
    suf = case[0]
    out = []
    for v in vals:
        row = dict(vdd=0.9, temp=27, w=3 * LMIN, vd=0, vs=0, vg=0, vb=0, **NOM[suf])
        vdd = 0.9
        vg, vb = {"nmos_on": (vdd, 0), "nmos_off": (0, 0), "pmos_on": (0, vdd), "pmos_off": (vdd, vdd)}[case]
        row.update(vg=vg, vb=vb, **base_over)
        row[key] = v
        r = run_ngspice((single_netlist(case, row), V, None))
        out.append(abs(-r[{"id": "i(vd)", "ig": "i(vg)", "is": "i(vs)", "ib": "i(vb)"}[target]]))
    return np.array(out)


# trend 1/2: OFF current (drain current, |Vds| = VDD) vs temperature / vth0 ; trend 3: Ig vs toxe (ON state, Vds=0)
off_over = {"nmos_off": dict(vd=0.9, vs=0.0), "pmos_off": dict(vd=0.0, vs=0.9)}
T = np.linspace(-40, 125, 12)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
res = {}
for c in ["nmos_off", "pmos_off"]:
    y = sweep(c, "temp", T, off_over[c], "id"); ax[0].semilogy(T, y, "o-", label=c); res[f"{c}_T"] = y
ax[0].set(xlabel="Temperature (C)", ylabel="|Id| OFF (A)", title="OFF current vs temperature (SPICE, others nominal)"); ax[0].legend(); ax[0].grid(alpha=.3)
for c, s in [("nmos_off", "n"), ("pmos_off", "p")]:
    vals = NOM[s]["vth0"] * np.linspace(0.9, 1.1, 9)
    y = sweep(c, "vth0", vals, off_over[c], "id"); ax[1].semilogy(abs(vals), y, "o-", label=c); res[f"{c}_vth"] = y
ax[1].set(xlabel="|vth0| (V)", ylabel="|Id| OFF (A)", title="OFF current vs vth0 (SPICE, others nominal)"); ax[1].legend(); ax[1].grid(alpha=.3)
for c, s in [("nmos_on", "n"), ("pmos_on", "p")]:
    vals = NOM[s]["toxe"] * np.linspace(0.9, 1.1, 9)
    on_over = dict(vd=0.0, vs=0.0) if c == "nmos_on" else dict(vd=0.9, vs=0.9)   # |Vgs| = VDD, Vds = 0
    y = sweep(c, "toxe", vals, on_over, "ig"); ax[2].semilogy(vals * 1e9, y, "o-", label=c); res[f"{c}_tox"] = y
ax[2].set(xlabel="toxe (nm)", ylabel="|Ig| ON (A)", title="Gate current vs toxe (SPICE, |Vgs|=VDD, Vds=0)"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "trends_controlled.png"), dpi=140); plt.close()
# the three required trend plots, also as separate files
for key, fname, xl, yl, ttl, cases in [
        ("T", "trend_off_current_vs_temperature.png", "Temperature (C)", "|Id| OFF (A)", "OFF current vs temperature", ["nmos_off", "pmos_off"]),
        ("vth", "trend_off_current_vs_vth0.png", "|vth0| (V)", "|Id| OFF (A)", "OFF current vs vth0", ["nmos_off", "pmos_off"]),
        ("tox", "trend_gate_current_vs_toxe.png", "toxe (nm)", "|Ig| ON (A)", "Gate current vs toxe (|Vgs|=VDD, Vds=0)", ["nmos_on", "pmos_on"])]:
    fig, a = plt.subplots(figsize=(5.6, 4.3))
    for c in cases:
        s = c[0]
        x = T if key == "T" else (np.abs(NOM[s]["vth0"]) * np.linspace(0.9, 1.1, 9) if key == "vth" else NOM[s]["toxe"] * np.linspace(0.9, 1.1, 9) * 1e9)
        a.semilogy(x, res[f"{c}_{key}"], "o-", label=c)
    a.set(xlabel=xl, ylabel=yl, title=ttl + " (SPICE, others nominal)"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, fname), dpi=140); plt.close()

# monotonicity assertions (physical sanity)
txt = []
for c in ["nmos_off", "pmos_off"]:
    txt.append(f"{c}: Id(T) increases monotonic: {bool(np.all(np.diff(res[f'{c}_T'])>0))}; Id vs |vth0| decreases monotonic: {bool(np.all(np.diff(res[f'{c}_vth'])<0))}")
for c in ["nmos_on", "pmos_on"]:
    txt.append(f"{c}: |Ig| vs toxe decreases monotonic: {bool(np.all(np.diff(res[f'{c}_tox'])<0))}")
open(os.path.join(RES, "02_trends.txt"), "w").write("\n".join(txt) + "\n")
print("\n".join(txt))

# dataset-level trends (binned medians, confounded by the other random parameters, shown for completeness)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for c in ["nmos_off", "pmos_off"]:
    d = pd.read_csv(os.path.join(DATA, f"{c}.csv"))
    ax[0].scatter(d.temp, d.id.abs(), s=2, alpha=.3, label=c)
    ax[1].scatter(d.vth0.abs(), d.id.abs(), s=2, alpha=.3, label=c)
for c in ["nmos_on", "pmos_on"]:
    d = pd.read_csv(os.path.join(DATA, f"{c}.csv"))
    ax[2].scatter(d.toxe * 1e9, d.ig.abs(), s=2, alpha=.3, label=c)
for a, (x, y, t) in zip(ax, [("Temperature (C)", "|Id| (A)", "dataset: OFF Id vs T"), ("|vth0| (V)", "|Id| (A)", "dataset: OFF Id vs vth0"), ("toxe (nm)", "|Ig| (A)", "dataset: ON Ig vs toxe")]):
    a.set_yscale("log"); a.set(xlabel=x, ylabel=y, title=t); a.legend(markerscale=4); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "trends_dataset.png"), dpi=140); plt.close()
