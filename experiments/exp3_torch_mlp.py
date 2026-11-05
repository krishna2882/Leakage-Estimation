"""Torch MLP experiment: physics features (v2), targets asinh(I/I0_t) per terminal, Adam + cosine LR, then L-BFGS polish."""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from common import *
T3 = importlib.import_module("03_train_models")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_device_models import feats_v2
torch.manual_seed(SEED); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))


def train(Xtr, Ytr, Xva, Yva, width=128, depth=4, epochs=3000, lbfgs=200):
    net = [torch.nn.Linear(Xtr.shape[1], width), torch.nn.SiLU()]
    for _ in range(depth - 1):
        net += [torch.nn.Linear(width, width), torch.nn.SiLU()]
    net = torch.nn.Sequential(*net, torch.nn.Linear(width, Ytr.shape[1])).double()
    X, Y = torch.tensor(Xtr), torch.tensor(Ytr)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad(); loss = ((net(X[b]) - Y[b]) ** 2).mean(); loss.backward(); opt.step()
        sch.step()
    if lbfgs:
        opt2 = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=lbfgs, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad(); l = ((net(X) - Y) ** 2).mean(); l.backward(); return l
        opt2.step(closure)
    with torch.no_grad():
        return net(torch.tensor(Xva)).numpy()


if __name__ == "__main__":
    for case in CASES:
        df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, _ = T3.split(df)
        Xtr, Xva = feats_v2(tr, case[0]), feats_v2(va, case[0])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-30; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        for fac in (0, 1e-3, 1e-2):
            for (w, dpt, ep) in [(128, 4, 2000), (256, 4, 4000)]:
                t0 = time.time()
                I0 = np.array([max(1e-18, fac * np.median(np.abs(tr[t]))) if fac else 1e-18 for t in TARGETS])
                Y = np.arcsinh(tr[TARGETS].values / I0); ym, ys = Y.mean(0), Y.std(0)
                P = train(Xtr, (Y - ym) / ys, Xva, None, w, dpt, ep) * ys + ym
                P = I0 * np.sinh(P)
                e = T3.relerr(P, va[TARGETS].values); far = ((va.vd - va.vs).abs() / va.vdd > 0.05).values
                per = " ".join(f"{t}:{np.median(e[:, j]):.2f}/{np.percentile(e[:, j], 95):.1f}" for j, t in enumerate(TARGETS))
                print(f"{case:9s} torch w{w}d{dpt}e{ep} I0fac={fac:<6} ALL med {np.median(e):6.2f}% p95 {np.percentile(e,95):7.1f}% | "
                      f"|Vds|>5%: med {np.median(e[far]):6.2f}% p95 {np.percentile(e[far],95):7.1f}% | {per} [{time.time()-t0:.0f}s]", flush=True)
