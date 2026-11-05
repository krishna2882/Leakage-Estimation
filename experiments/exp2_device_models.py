"""Device-model experiments: physics features, per-terminal models, asinh scale I0.
Reports relative error (floor 1 fA) overall and for |Vds| > 0.05 VDD, on the validation split."""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from common import *
T3 = importlib.import_module("03_train_models")
K_Q = 8.617333e-5


def feats_v2(df, typ):
    s = -1.0 if typ == "p" else 1.0          # mirror PMOS voltages to NMOS-like polarity
    vT = K_Q * (df.temp.values + 273.15)
    vd, vg, vs, vb = (s * df[c].values for c in ("vd", "vg", "vs", "vb"))
    vds, vgs, vgd, vbs, vbd = vd - vs, vg - vs, vg - vd, vb - vs, vb - vd
    vth = np.abs(df.vth0.values); tox = df.toxe.values
    a = np.abs(vds) / vT
    return np.column_stack([df.vdd, df.temp, vT, np.log(df.w), vds, vgs, vgd, vbs, vbd, vds / vT, vgs / vT, vgd / vT,
                            (np.maximum(vgs, vgd) - vth) / vT, (np.minimum(vgs, vgd) - vth) / vT,
                            np.log(-np.expm1(-np.maximum(a, 1e-9))), vth, np.log(tox), 1e-9 / tox, vgs / tox * 1e-9, vgd / tox * 1e-9,
                            np.log(df.ndep), np.log(df.u0), df.xj * 1e8])


def run(case, kind, i0fac, feat):
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv"))
    tr, va, _ = T3.split(df)
    typ = case[0]
    if feat == "v2":
        Xtr, Xva = feats_v2(tr, typ), feats_v2(va, typ)
    else:
        from mlmodels import featurize
        Xtr, Xva = featurize(tr), featurize(va)
    xs = StandardScaler().fit(Xtr); Xtr, Xva = xs.transform(Xtr), xs.transform(Xva)
    P = np.zeros((len(va), 4))
    for j, t in enumerate(TARGETS):
        I0 = max(1e-18, i0fac * np.median(np.abs(tr[t]))) if i0fac else 1e-18
        y = np.arcsinh(tr[t].values / I0); ys = StandardScaler().fit(y[:, None])
        yy = ys.transform(y[:, None]).ravel()
        if kind == "gb":
            m = HistGradientBoostingRegressor(max_iter=4000, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=100, random_state=SEED).fit(Xtr, yy)
        else:
            m = MLPRegressor((128, 128, 128), activation="tanh", learning_rate_init=1e-3, batch_size=64, max_iter=1500,
                             early_stopping=True, validation_fraction=0.1, n_iter_no_change=60, alpha=0, tol=1e-7,
                             random_state=SEED).fit(Xtr, yy)
        P[:, j] = I0 * np.sinh(ys.inverse_transform(m.predict(Xva)[:, None]).ravel())
    e = T3.relerr(P, va[TARGETS].values)
    far = ((va.vd - va.vs).abs() / va.vdd > 0.05).values
    return e, far


if __name__ == "__main__":
    configs = [("gb", 0, "v1"), ("gb", 0, "v2"), ("gb", 1e-3, "v2"), ("mlp", 0, "v2"), ("mlp", 1e-3, "v2"), ("mlp", 1e-2, "v2")]
    for case in CASES:
        for kind, f, feat in configs:
            t0 = time.time()
            e, far = run(case, kind, f, feat)
            per = " ".join(f"{t}:{np.median(e[:, j]):.2f}/{np.percentile(e[:, j], 95):.1f}" for j, t in enumerate(TARGETS))
            print(f"{case:9s} {kind:3s} I0fac={f:<6} {feat}  ALL med {np.median(e):6.2f}% p95 {np.percentile(e,95):7.1f}% | "
                  f"|Vds|>5%: med {np.median(e[far]):6.2f}% p95 {np.percentile(e[far],95):7.1f}% | {per} [{time.time()-t0:.0f}s]", flush=True)
