"""Tier 2: 70/15/15 split (seed fixed), 4 model types x 4 cases, selection on validation, test report."""
import sys, time, json
import pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from common import *
from mlmodels import CurrentModel, fwd

KINDS = ["poly_ridge", "random_forest", "grad_boost", "mlp"]
VARIANTS = {"poly_ridge": [dict(degree=3), dict(degree=4)], "random_forest": [dict(n=500)],
            "grad_boost": [dict(iters=3000)],
            "mlp": [dict(hidden=(256, 256, 256), act="tanh", epochs=800), dict(hidden=(512, 512, 512), act="relu", lr=1e-3, epochs=800)]}
if len(sys.argv) > 1 and sys.argv[1] == "quick":
    VARIANTS["mlp"] = [dict(epochs=20)]; VARIANTS["poly_ridge"] = [dict(degree=3)]
FLOOR = 1e-15      # A; relative error = |dI| / max(|I_true|, FLOOR)
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"


def split(df):
    tr, rest = train_test_split(df, test_size=0.30, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED)
    return tr, va, te


def relerr(pred, true):
    return np.abs(pred - true) / np.maximum(np.abs(true), FLOOR) * 100.0


def metrics(pred, true):
    """per-terminal and pooled metrics. R2 on the transformed target (log-like scale)."""
    out = {}
    e = relerr(pred, true)
    for j, t in enumerate(TARGETS):
        out[t] = dict(r2=r2_score(fwd(true[:, j]), fwd(pred[:, j])), med=float(np.median(e[:, j])), p95=float(np.percentile(e[:, j], 95)))
    out["pooled"] = dict(r2=float(np.mean([out[t]["r2"] for t in TARGETS])), med=float(np.median(e)), p95=float(np.percentile(e, 95)))
    # significant terminals only (id, ig, is): the ones that set gate leakage
    e3 = e[:, :3]
    out["pooled_ids"] = dict(r2=float(np.mean([out[t]["r2"] for t in TARGETS[:3]])), med=float(np.median(e3)), p95=float(np.percentile(e3, 95)))
    return out


if __name__ == "__main__":
    rows, chosen, hpd = [], {}, {}
    for case in CASES:
        df = pd.read_csv(os.path.join(DATA, f"{case}.csv"))
        tr, va, te = split(df)
        print(case, len(tr), len(va), len(te), flush=True)
        best = None
        for kind in KINDS:
            t0 = time.time()
            cand = None
            for hp in VARIANTS[kind]:             # hyper-parameter variants: pick best per type on validation
                mm = CurrentModel(kind, seed=SEED, n_jobs=NPROC, **hp).fit(tr, tr[TARGETS].values)
                mvv = metrics(mm.predict(va), va[TARGETS].values)
                print(f"    variant {kind} {hp}: val pooled med={mvv['pooled']['med']:.3f}% p95={mvv['pooled']['p95']:.2f}%", flush=True)
                hpd[(case, kind)] = hpd.get((case, kind), []) + [dict(hp=str(hp), med=mvv["pooled"]["med"], p95=mvv["pooled"]["p95"])]
                if cand is None or (mvv["pooled"]["med"] + mvv["pooled"]["p95"]) < cand[0]:
                    cand = (mvv["pooled"]["med"] + mvv["pooled"]["p95"], mm, mvv, hp)
            _, m, mv, hp_best = cand
            hpd[(case, kind, "best")] = str(hp_best)
            for t in TARGETS + ["pooled", "pooled_ids"]:
                rows.append(dict(case=case, model=kind, split="val", terminal=t, **mv[t]))
            print(f"  {kind:14s} val pooled r2={mv['pooled']['r2']:.5f} med={mv['pooled']['med']:.3f}% p95={mv['pooled']['p95']:.2f}%  "
                  f"(ids: med={mv['pooled_ids']['med']:.3f}% p95={mv['pooled_ids']['p95']:.2f}%) {time.time()-t0:.0f}s", flush=True)
            score = mv["pooled"]["p95"]
            if best is None or score < best[0]:
                best = (score, kind, m)
        _, kind, m = best
        pt = m.predict(te)
        mt = metrics(pt, te[TARGETS].values)
        for t in TARGETS + ["pooled", "pooled_ids"]:
            rows.append(dict(case=case, model=kind, split="test", terminal=t, **mt[t]))
        chosen[case] = kind
        print(f"  -> chosen {kind}: TEST pooled med={mt['pooled']['med']:.3f}% p95={mt['pooled']['p95']:.2f}%", flush=True)
        joblib.dump(m, os.path.join(SAVED, f"{case}_{kind}.joblib"))
        # predicted-vs-SPICE plot (test set)
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
        for j, t in enumerate(TARGETS):
            a = np.abs(te[t].values); b = np.abs(pt[:, j])
            ax[j].loglog(np.maximum(a, 1e-18), np.maximum(b, 1e-18), ".", ms=2, alpha=.4)
            lo, hi = max(min(a.min(), b.min()), 1e-18), max(a.max(), b.max())
            ax[j].plot([lo, hi], [lo, hi], "r-", lw=.8)
            ax[j].set(xlabel=f"SPICE |{t}| (A)", ylabel=f"ML |{t}| (A)", title=f"{case} {t}: med {mt[t]['med']:.2f}%, p95 {mt[t]['p95']:.1f}%")
            ax[j].grid(alpha=.3)
        plt.suptitle(f"{case} - chosen model: {kind} (test set)"); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"pred_vs_spice_{case}.png"), dpi=130); plt.close()
    pd.DataFrame(rows).to_csv(os.path.join(RES, "03_model_metrics_long.csv"), index=False)
    json.dump({str(k): v for k, v in hpd.items()}, open(os.path.join(RES, "03_hyperparams.json"), "w"), indent=1)
    json.dump(chosen, open(os.path.join(SAVED, "chosen.json"), "w"))
    print("chosen:", chosen)
