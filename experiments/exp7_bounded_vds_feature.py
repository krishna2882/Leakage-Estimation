"""Drain-bias feature: ln(1 - exp(-|Vds|/vT)) (current) vs the bounded 1 - exp(-|Vds|/vT).
Inside gates the ON transistors sit at |Vds| ~ 0.3 mV, beyond the 0.1 % quantile of the uniform training data; the log
form stretches that sparse region over a long feature range, the bounded form keeps it next to the many samples with
|Vds| of a few mV (where Ig and Ib barely change).  Same 10k data and split; selection data: the validation split and the
independent rail-bias set of exp5 (gate test corners are not used).  usage: python experiments/exp7_bounded_vds_feature.py <case>"""
import sys, os, time, json, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *
import mlmodels
from mlmodels import CurrentModel
from exp5_rail_weighting import rail_set, power_err, OUT
T3 = importlib.import_module("03_train_models")
BASE = mlmodels.featurize                    # current pipeline features; column 14 = drain-bias factor


def _a(df):
    return np.abs(df["vd"].values - df["vs"].values) / (mlmodels.K_Q * (df["temp"].values + 273.15))


def featurize_bounded(df, typ):
    X = BASE(df, typ); X[:, 14] = -np.expm1(-_a(df)); return X


def featurize_log(df, typ):                  # the form used before this study
    X = BASE(df, typ); X[:, 14] = np.log(-np.expm1(-np.maximum(_a(df), 1e-9))); return X


LOGFEAT = featurize_log


def term_err(pred, true):
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-18) * 100


if __name__ == "__main__":
    case = sys.argv[1]
    df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, te = T3.split(df)
    rv = rail_set(case)
    z = (rv.vd == rv.vs).values                              # both terminals on the same rail: Vds = 0
    print(case, "train", len(tr), "rail-val", len(rv), "of which Vds = 0:", int(z.sum()), flush=True)
    rows = []
    for name, feat in [("log", LOGFEAT), ("bounded", featurize_bounded)]:
        mlmodels.featurize = feat                             # CurrentModel looks the function up at call time
        t0 = time.time()
        m = CurrentModel("mlp", case[0], seed=SEED, n_jobs=NPROC, width=128, depth=4, epochs=2000, ens=5).fit(tr, tr[TARGETS].values)
        mv = T3.metrics(m.predict(va), va[TARGETS].values)
        pr = m.predict(rv)
        ep = power_err(rv, pr)
        eg = term_err(pr[:, 1], rv.ig.values); eb = term_err(pr[:, 3], rv.ib.values)
        r = dict(case=case, feature=name, val_med=mv["pooled"]["med"], val_p95=mv["pooled"]["p95"],
                 val_ig_med=mv["ig"]["med"], rail_power_med=float(np.median(np.abs(ep))), rail_power_p95=float(np.percentile(np.abs(ep), 95)),
                 rail_ig_med_vds0=float(np.median(eg[z])), rail_ig_p95_vds0=float(np.percentile(eg[z], 95)),
                 rail_ig_bias_vds0=float(np.median(((pr[:, 1] - rv.ig.values) / np.abs(rv.ig.values) * 100)[z])),
                 rail_ib_med_vds0=float(np.median(eb[z])), rail_ig_med_other=float(np.median(eg[~z])), secs=time.time() - t0)
        rows.append(r)
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"exp7_{case}.csv"), index=False)
    mlmodels.featurize = BASE
