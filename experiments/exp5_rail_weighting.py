"""Rail-weighted training: inside gates every transistor sits with Vd, Vs at (or near) 0 or VDD, the corners of the uniform
Vd-Vs training square.  Same 10k datasets and split; only the training loss is re-weighted toward the rails.
Selection data: the usual validation split plus an independent rail-bias validation set (own seed, single-FET SPICE,
Vd and Vs each snapped to 0 / VDD with prob. 0.45 / 0.45, uniform otherwise).  Gate test corners are never used here.
usage: python experiments/exp5_rail_weighting.py <case>"""
import sys, os, time, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from common import *
from mlmodels import CurrentModel, TorchMLP, featurize, I0_FRAC
T3 = importlib.import_module("03_train_models")
G1 = importlib.import_module("01_gen_data")
OUT = os.path.join(ROOT, "experiments", "out"); os.makedirs(OUT, exist_ok=True)


def rail_kernel(v, vdd, s=0.05):
    """1 at a rail, exp(-dist / (s VDD)) away from it"""
    return np.exp(-np.minimum(v, vdd - v) / (s * vdd))


def rail_weight(df, alpha):
    return 1.0 + alpha * rail_kernel(df.vd.values, df.vdd.values) * rail_kernel(df.vs.values, df.vdd.values)


class WTorchMLP(TorchMLP):
    def fit(self, X, Y, w):
        torch.set_num_threads(self.threads); torch.manual_seed(self.seed)
        L = [torch.nn.Linear(X.shape[1], self.width), torch.nn.SiLU()]
        for _ in range(self.depth - 1):
            L += [torch.nn.Linear(self.width, self.width), torch.nn.SiLU()]
        net = torch.nn.Sequential(*L, torch.nn.Linear(self.width, Y.shape[1])).double()
        X, Y, W = torch.tensor(X), torch.tensor(Y), torch.tensor(w / w.mean())[:, None]
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), 512):
                b = perm[i:i + 512]
                opt.zero_grad(); (W[b] * (net(X[b]) - Y[b]) ** 2).mean().backward(); opt.step()
            sch.step()
        opt2 = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=200, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(); l = (W * (net(X) - Y) ** 2).mean(); l.backward(); return l
        opt2.step(closure)
        self.net = net.eval()
        return self


class WCurrentModel(CurrentModel):
    def fit(self, df, I, w):
        X = featurize(df, self.typ)
        self.I0 = np.maximum(I0_FRAC * np.median(np.abs(I), axis=0), 1e-18)
        Y = np.arcsinh(I / self.I0)
        from sklearn.preprocessing import StandardScaler
        self.xs = StandardScaler().fit(X); self.ys = StandardScaler().fit(Y)
        self.ms = [WTorchMLP(self.kw.get("width", 128), self.kw.get("depth", 4), self.kw.get("epochs", 2000), self.seed + e,
                             self.n_jobs).fit(self.xs.transform(X), self.ys.transform(Y), w) for e in range(self.kw.get("ens", 5))]
        return self


def rail_set(case, n=2000):
    f = os.path.join(OUT, f"railval_{case}.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    rng = np.random.default_rng(SEED + 500 + CASES.index(case))
    df = G1.draw(case, n, rng)
    for c in ("vd", "vs"):
        u = rng.uniform(size=n)
        df[c] = np.where(u < 0.45, 0.0, np.where(u < 0.9, df.vdd, df[c]))
    res = G1.simulate(case, df)
    ok = [r is not None for r in res]
    df = df[ok].reset_index(drop=True); res = [r for r in res if r is not None]
    for t, v in zip(TARGETS, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]):
        df[t] = [-r[v] for r in res]
    df.to_csv(f, index=False)
    return df


def power_err(df, pred):
    """error of the device's power term (Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib, the quantity that enters gate leakage"""
    dv = lambda a: (df[a] - df.vs).values
    ps = dv("vd") * df.id.values + dv("vg") * df.ig.values + dv("vb") * df.ib.values
    pm = dv("vd") * pred[:, 0] + dv("vg") * pred[:, 1] + dv("vb") * pred[:, 3]      # columns id, ig, is, ib
    sig = np.abs(ps) > 1e-3 * np.abs(ps).max()
    return (pm - ps)[sig] / np.abs(ps[sig]) * 100


if __name__ == "__main__":
    case = sys.argv[1]
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, te = T3.split(df)
    rv = rail_set(case)
    print(case, "train", len(tr), "val", len(va), "rail-val", len(rv), flush=True)
    VAR = [dict(alpha=0), dict(alpha=10), dict(alpha=30), dict(alpha=100), dict(alpha=30, width=256)]
    rows = []
    for v in VAR:
        t0 = time.time()
        hp = dict(width=v.get("width", 128), depth=4, epochs=2000, ens=5)
        m = WCurrentModel("mlp", case[0], seed=SEED, n_jobs=NPROC, **hp).fit(tr, tr[TARGETS].values, rail_weight(tr, v["alpha"]))
        mv = T3.metrics(m.predict(va), va[TARGETS].values)["pooled"]
        pr = m.predict(rv); mr = T3.metrics(pr, rv[TARGETS].values)
        ep = power_err(rv, pr)
        r = dict(case=case, **v, val_med=mv["med"], val_p95=mv["p95"], rail_med=mr["pooled"]["med"], rail_p95=mr["pooled"]["p95"],
                 rail_ig_med=mr["ig"]["med"], rail_power_med=float(np.median(np.abs(ep))),
                 rail_power_p95=float(np.percentile(np.abs(ep), 95)), rail_power_bias=float(np.median(ep)), secs=time.time() - t0)
        rows.append(r)
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"exp5_{case}.csv"), index=False)
