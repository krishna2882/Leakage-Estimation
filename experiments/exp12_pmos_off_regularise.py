"""pmos_off is variance-limited (exp9: train 0.15 % vs validation 1.0 %; exp10: the SPICE data is smooth).  In off
devices the drain and gate currents are almost entirely gate-overlap tunnelling (exp11: median |Igd|/|Id| = 0.99999),
whose BSIM4 form is  ln Igd ~ ln W + ln Vaux(Vgd) - 2 ln toxe - B toxe (A - B' |Vgd|)(1 + C |Vgd|),
Vaux = vT ln(1 + exp(|Vgd|/vT)).  Variants (same data, split and selection sets as the pipeline):
  base      pipeline network (128x4, 2000 epochs, ensemble of 5)
  wd        + weight decay 1e-4 (AdamW)
  ens15     ensemble of 15
  small     64x4
  tunnel    + tunnelling features ln(Vaux/vT), toxe |V|, toxe V^2 for Vgs and Vgd
usage: python experiments/exp12_pmos_off_regularise.py <variant> [case]"""
import sys, os, time, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *
import mlmodels
from mlmodels import CurrentModel, TorchMLP
from exp5_rail_weighting import rail_set, OUT
from exp9_pmos_off_capacity import power_err, corner
T3 = importlib.import_module("03_train_models")
BASE = mlmodels.featurize


def featurize_tunnel(df, typ):
    s = -1.0 if typ == "p" else 1.0
    vT = mlmodels.K_Q * (df["temp"].values + 273.15); tox = df["toxe"].values * 1e9
    ex = []
    for v in (s * (df.vg.values - df.vs.values), s * (df.vg.values - df.vd.values)):
        a = np.abs(v)
        ex += [np.log(np.logaddexp(0, a / vT)), tox * a, tox * v * v]
    return np.column_stack([BASE(df, typ)] + ex)


class WDMLP(TorchMLP):
    """TorchMLP with AdamW weight decay (torch.optim.Adam is swapped for the duration of fit)"""
    wd = 1e-4

    def fit(self, X, Y):
        import torch
        adam = torch.optim.Adam
        torch.optim.Adam = lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=self.wd)
        try:
            return super().fit(X, Y)
        finally:
            torch.optim.Adam = adam


VARIANTS = {"base": {}, "wd": {}, "ens15": dict(ens=15), "small": dict(width=64), "tunnel": {}}

if __name__ == "__main__":
    var = sys.argv[1]; case = sys.argv[2] if len(sys.argv) > 2 else "pmos_off"
    if var == "tunnel":
        mlmodels.featurize = featurize_tunnel
    if var == "wd":
        mlmodels.TorchMLP = WDMLP
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, te = T3.split(df); rv = rail_set(case)
    kw = dict(width=128, depth=4, epochs=2000, ens=5); kw.update(VARIANTS[var])
    t0 = time.time()
    m = CurrentModel("mlp", case[0], seed=SEED, n_jobs=NPROC, **kw).fit(tr, tr[TARGETS].values)
    r = dict(case=case, variant=var)
    for nm, d in [("train", tr), ("val", va)]:
        pr = m.predict(d); mt = T3.metrics(pr, d[TARGETS].values); ep, sig = power_err(d, pr); k = corner(d) & sig
        r.update({f"{nm}_med": mt["pooled"]["med"], f"{nm}_p95": mt["pooled"]["p95"],
                  **{f"{nm}_{t}_med": mt[t]["med"] for t in TARGETS},
                  f"{nm}_power_med": float(np.median(ep[sig])), f"{nm}_power_p95": float(np.percentile(ep[sig], 95)),
                  f"{nm}_corner_power_med": float(np.median(ep[k])), f"{nm}_corner_power_p95": float(np.percentile(ep[k], 95))})
    ep, sig = power_err(rv, m.predict(rv)); k = corner(rv) & sig
    r.update(rail_power_med=float(np.median(ep[sig])), rail_power_p95=float(np.percentile(ep[sig], 95)),
             rail_corner_power_med=float(np.median(ep[k])), rail_corner_power_p95=float(np.percentile(ep[k], 95)),
             secs=time.time() - t0)
    print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
    pd.DataFrame([r]).to_csv(os.path.join(OUT, f"exp12_{case}_{var}.csv"), index=False)
