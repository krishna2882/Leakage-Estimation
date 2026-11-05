"""ML wrappers: predict the four terminal currents (id, ig, is, ib) from the 12 input parameters.

Preprocessing (same for every model type):
  * physics features: thermal voltage vT, terminal-voltage differences (PMOS mirrored to NMOS polarity),
    the same normalised by vT, gate overdrive (V_g* - |vth0|)/vT, the drain-bias factor 1 - exp(-|Vds|/vT),
    oxide-field terms, ln W, ln toxe, ln ndep, ln u0, xj
    (the drain-bias factor is used in its bounded form: inside gates ON devices sit at |Vds| ~ 0, where its logarithm
    runs far outside the training range; experiments/exp7_bounded_vds_feature.py)
  * target per terminal: y = asinh(I / I0_t), I0_t = 1 % of the training median |I_t|
    -> log-like (relative error) for |I| >> I0_t, linear and smooth where the current changes sign (Vds ~ 0)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

FEATS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]
K_Q = 8.617333e-5          # Boltzmann constant / q  (V/K)
I0_FRAC = 1e-2


def featurize(df, typ):
    s = -1.0 if typ == "p" else 1.0
    vT = K_Q * (df["temp"].values + 273.15)
    vd, vg, vs, vb = (s * df[c].values for c in ("vd", "vg", "vs", "vb"))
    vds, vgs, vgd, vbs, vbd = vd - vs, vg - vs, vg - vd, vb - vs, vb - vd
    vth = np.abs(df["vth0"].values); tox = df["toxe"].values
    a = np.abs(vds) / vT
    return np.column_stack([df["vdd"].values, df["temp"].values, vT, np.log(df["w"].values), vds, vgs, vgd, vbs, vbd,
                            vds / vT, vgs / vT, vgd / vT,
                            (np.maximum(vgs, vgd) - vth) / vT, (np.minimum(vgs, vgd) - vth) / vT,
                            -np.expm1(-a), vth, np.log(tox), 1e-9 / tox,
                            vgs / tox * 1e-9, vgd / tox * 1e-9,
                            np.log(df["ndep"].values), np.log(df["u0"].values), df["xj"].values * 1e8])


class TorchMLP:
    """Fully connected SiLU network, Adam + cosine LR, then L-BFGS polish; float64."""

    def __init__(self, width=128, depth=4, epochs=2000, seed=0, threads=4):
        self.width, self.depth, self.epochs, self.seed, self.threads = width, depth, epochs, seed, threads

    def fit(self, X, Y):
        import torch
        torch.set_num_threads(self.threads); torch.manual_seed(self.seed)
        L = [torch.nn.Linear(X.shape[1], self.width), torch.nn.SiLU()]
        for _ in range(self.depth - 1):
            L += [torch.nn.Linear(self.width, self.width), torch.nn.SiLU()]
        net = torch.nn.Sequential(*L, torch.nn.Linear(self.width, Y.shape[1])).double()
        X, Y = torch.tensor(X), torch.tensor(Y)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), 512):
                b = perm[i:i + 512]
                opt.zero_grad(); ((net(X[b]) - Y[b]) ** 2).mean().backward(); opt.step()
            sch.step()
        opt2 = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=200, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(); l = ((net(X) - Y) ** 2).mean(); l.backward(); return l
        opt2.step(closure)
        self.net = net.eval()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.tensor(X)).numpy()


class CurrentModel:
    """kind in {poly_ridge, random_forest, grad_boost, mlp}; mlp = ensemble of `ens` TorchMLPs (mean in y-space)."""

    def __init__(self, kind, typ, seed=0, n_jobs=4, **kw):
        self.kind, self.typ, self.seed, self.n_jobs, self.kw = kind, typ, seed, n_jobs, kw

    def _make(self, seed):
        k = self.kind
        if k == "poly_ridge":
            return make_pipeline(PolynomialFeatures(self.kw.get("degree", 3)), RidgeCV(alphas=np.logspace(-6, 2, 9)))
        if k == "random_forest":
            return RandomForestRegressor(n_estimators=self.kw.get("n", 500), n_jobs=self.n_jobs, random_state=seed)
        if k == "grad_boost":
            return HistGradientBoostingRegressor(max_iter=self.kw.get("iters", 4000), learning_rate=0.05, max_leaf_nodes=63,
                                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=100, random_state=seed)
        if k == "mlp":
            return TorchMLP(self.kw.get("width", 128), self.kw.get("depth", 4), self.kw.get("epochs", 2000), seed, self.n_jobs)
        raise ValueError(k)

    def fit(self, df, I):
        X = featurize(df, self.typ)
        self.I0 = np.maximum(I0_FRAC * np.median(np.abs(I), axis=0), 1e-18)
        Y = np.arcsinh(I / self.I0)
        self.xs = StandardScaler().fit(X); self.ys = StandardScaler().fit(Y)
        Xs, Ys = self.xs.transform(X), self.ys.transform(Y)
        if self.kind == "grad_boost":        # single-output estimator: one per terminal
            self.ms = [[self._make(self.seed).fit(Xs, Ys[:, j]) for j in range(Ys.shape[1])]]
        elif self.kind == "mlp":
            self.ms = [self._make(self.seed + e).fit(Xs, Ys) for e in range(self.kw.get("ens", 5))]
        else:
            self.ms = [self._make(self.seed).fit(Xs, Ys)]
        return self

    def predict_y(self, df):
        Xs = self.xs.transform(featurize(df, self.typ))
        if self.kind == "grad_boost":
            Ys = np.column_stack([m.predict(Xs) for m in self.ms[0]])
        else:
            Ys = np.mean([m.predict(Xs) for m in self.ms], axis=0)
        return self.ys.inverse_transform(Ys)

    def predict(self, df):
        """currents (n,4) in A, columns id, ig, is, ib (into the terminal)"""
        return self.I0 * np.sinh(self.predict_y(df))
