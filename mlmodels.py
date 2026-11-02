"""ML wrappers: predict the four terminal currents (id, ig, is, ib) from the 12 input features.
Target transform: y = arcsinh(I / I0), I0 = 1 aA  (signed log10-like for |I| >> I0, linear near 0)."""
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

I0 = 1e-18
FEATS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]


def fwd(I):
    return np.arcsinh(np.asarray(I) / I0)


def inv(y):
    return I0 * np.sinh(np.asarray(y))


def featurize(df):
    v = df["vdd"].values; T = df["temp"].values
    cols = [v, T, np.log(df["w"].values), df["vd"].values, df["vs"].values, df["vg"].values, df["vb"].values,
            df["vd"].values / v, df["vs"].values / v, (df["vd"].values - df["vs"].values) / v,
            df["vg"].values / v, df["vb"].values / v, 1000.0 / (T + 273.15),
            df["vth0"].values, np.log(df["toxe"].values), np.log(df["ndep"].values), np.log(df["u0"].values), df["xj"].values]
    return np.column_stack(cols)


class CurrentModel:
    def __init__(self, kind, seed=0, n_jobs=4, **kw):
        self.kind, self.seed, self.n_jobs, self.kw = kind, seed, n_jobs, kw

    def _make(self):
        k = self.kind
        if k == "poly_ridge":
            return make_pipeline(StandardScaler(), PolynomialFeatures(self.kw.get("degree", 3)),
                                 RidgeCV(alphas=np.logspace(-4, 2, 7)))
        if k == "random_forest":
            return RandomForestRegressor(n_estimators=self.kw.get("n", 300), min_samples_leaf=1, n_jobs=self.n_jobs, random_state=self.seed)
        if k == "grad_boost":
            return HistGradientBoostingRegressor(max_iter=self.kw.get("iters", 2000), learning_rate=0.06, max_leaf_nodes=31,
                                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=60, random_state=self.seed)
        if k == "mlp":
            return MLPRegressor(hidden_layer_sizes=self.kw.get("hidden", (256, 256, 256)), activation=self.kw.get("act", "tanh"),
                                solver="adam", learning_rate_init=self.kw.get("lr", 2e-3), batch_size=128, max_iter=self.kw.get("epochs", 600),
                                early_stopping=True, validation_fraction=0.1, n_iter_no_change=40, alpha=1e-6, random_state=self.seed)
        raise ValueError(k)

    def fit(self, df, I):
        X = featurize(df); Y = fwd(I)
        self.xs = StandardScaler().fit(X)
        self.ys = StandardScaler().fit(Y)
        Xs, Ys = self.xs.transform(X), self.ys.transform(Y)
        if self.kind == "grad_boost":
            self.ms = [self._make().fit(Xs, Ys[:, j]) for j in range(Ys.shape[1])]
        else:
            self.ms = [self._make().fit(Xs, Ys)]
        return self

    def predict_y(self, df):
        Xs = self.xs.transform(featurize(df))
        if self.kind == "grad_boost":
            Ys = np.column_stack([m.predict(Xs) for m in self.ms])
        else:
            Ys = self.ms[0].predict(Xs)
        return self.ys.inverse_transform(Ys)

    def predict(self, df):
        """currents (n,4) in A: id, ig, is, ib"""
        return inv(self.predict_y(df))
