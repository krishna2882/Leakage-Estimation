"""Physics-structured torch model:
  Ig, Ib : asinh(I/I0) regression (single-signed per case, smooth)
  Id     : exp(a(x)) - exp(b(x))  (forward minus reverse, both positive -> smooth through Vds=0)
  Is     : -(Id + Ig + Ib)        (Kirchhoff holds exactly for every prediction)
Loss: squared error of asinh(I/s_t) on all four terminals, s_t = 1e-3 * median|I_t| (relative-error-like)."""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from common import *
from exp2_device_models import feats_v2
T3 = importlib.import_module("03_train_models")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
LN10 = np.log(10)


class FRNet(torch.nn.Module):
    def __init__(self, nin, width, depth, logscale):
        super().__init__()
        L = [torch.nn.Linear(nin, width), torch.nn.SiLU()]
        for _ in range(depth - 1):
            L += [torch.nn.Linear(width, width), torch.nn.SiLU()]
        self.trunk = torch.nn.Sequential(*L)
        self.head = torch.nn.Linear(width, 4)          # a, b (log Id fwd/rev), log|Ig|, log|Ib| (in units of ln)
        self.ls = torch.tensor(logscale)               # offsets so heads start near the data scale

    def forward(self, x, sg, sb):
        h = self.head(self.trunk(x)) + self.ls
        Id = torch.exp(h[:, 0]) - torch.exp(h[:, 1])
        Ig = sg * torch.exp(h[:, 2]); Ib = sb * torch.exp(h[:, 3])
        return torch.stack([Id, Ig, -(Id + Ig + Ib), Ib], 1)


def fit_predict(tr, va, width=128, depth=4, epochs=2500, seed=0):
    torch.manual_seed(seed)
    typ = tr.case_typ
    Xtr, Xva = feats_v2(tr, typ), feats_v2(va, typ)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-30
    X, Xv = torch.tensor((Xtr - mu) / sd), torch.tensor((Xva - mu) / sd)
    Y = torch.tensor(tr[TARGETS].values)
    s = torch.tensor([1e-3 * np.median(np.abs(tr[t])) for t in TARGETS])
    # Ig and Ib signs: dominant sign in the training data, per sample sign from the data is not available at test time,
    # so use the case-level sign (checked: fraction of samples with that sign is printed)
    sg = float(np.sign(np.median(tr.ig))); sb = float(np.sign(np.median(tr.ib)))
    lm = np.log(np.median(np.abs(tr[TARGETS].values), 0))
    net = FRNet(X.shape[1], width, depth, [lm[0], lm[0] - 2, lm[1], lm[3]]).double()
    tgt = torch.asinh(Y / s)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = ((torch.asinh(net(X[b], sg, sb) / s) - tgt[b]) ** 2).mean()
            loss.backward(); opt.step()
        sch.step()
    opt2 = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=300, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); l = ((torch.asinh(net(X, sg, sb) / s) - tgt) ** 2).mean(); l.backward(); return l
    opt2.step(closure)
    with torch.no_grad():
        return net(Xv, sg, sb).numpy(), (np.mean(np.sign(tr.ig) == sg), np.mean(np.sign(tr.ib) == sb))


if __name__ == "__main__":
    for case in CASES:
        df = pd.read_csv(os.path.join(DATA, f"{case}.csv")); tr, va, _ = T3.split(df)
        tr.case_typ = case[0]
        for (w, d, ep) in [(128, 4, 2500), (256, 4, 4000)]:
            t0 = time.time()
            P, signfrac = fit_predict(tr, va, w, d, ep)
            e = T3.relerr(P, va[TARGETS].values); far = ((va.vd - va.vs).abs() / va.vdd > 0.05).values
            per = " ".join(f"{t}:{np.median(e[:, j]):.2f}/{np.percentile(e[:, j], 95):.1f}" for j, t in enumerate(TARGETS))
            print(f"{case:9s} FR w{w}d{d}e{ep} sign-consistency ig/ib={signfrac[0]:.4f}/{signfrac[1]:.4f} ALL med {np.median(e):6.2f}% p95 {np.percentile(e,95):7.1f}% | "
                  f"|Vds|>5%: med {np.median(e[far]):6.2f}% p95 {np.percentile(e[far],95):7.1f}% | {per} [{time.time()-t0:.0f}s]", flush=True)
