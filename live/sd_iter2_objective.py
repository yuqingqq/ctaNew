"""Signal-diversity loop — iteration 2 (S3): train on the NET-OF-COST TRADING OBJECTIVE instead of MSE.

The incumbent fits RidgeCV to squared error on xs_z and applies cost afterwards. The SOTA stat-arb paper
attributes its net-Sharpe gain to jointly learning factors and the trading policy — "crucial to maximize
profitability after trading costs". Cost is our measured binding constraint, so this is the highest-prior
owned-data lever left after S1 closed.

Design isolates the OBJECTIVE, holding architecture fixed:
  - identical per-symbol linear parameterization, identical features, identical walk-forward folds/preproc
  - theta INITIALIZED at the RidgeCV solution, then gradient-ascended on the portfolio objective
  - so any difference is attributable to the objective, not to a pooled/deep architecture
    (gen_coef_shrink.py already showed pooled coefficients are far worse: +3.46 per-sym vs -0.71 common)

Objective (nothing tuned on the evaluation window):
    s_it = x_it . theta_i          score
    c_t  = s_t - mean(s_t)         cross-sectional demean  -> dollar neutral
    w_t  = c_t / sum|c_t|          L1-normalised            -> gross 1
    n_t  = w_t . a_t - sum_i |w_it - w_i,t-1| * cost_i      net book return, TRUE calibrated per-symbol cost
    J    = Sharpe(n) - rho * ||theta - theta_ridge||^2      shrinkage to the init, fixed coefficient
|.| is smoothed as sqrt(x^2+eps^2) so J is differentiable. Gradients are analytic and numerically verified
before any training run. Steps/learning-rate/shrinkage are fixed a priori (see CFG) — not searched.

Gates (live/SIGNAL_DIVERSITY_LOOP.md): G1 held-out net@10k beats the MSE incumbent on the SAME continuous
construction, paired 7d-block CI>0. G2 also beats the incumbent's best deployed construction (band). G3
rank-IC must not degrade materially in either era. Falsifier: G1 fails -> S3 does not transfer.
Run: python3 -u -m live.sd_iter2_objective
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.cost_loop_harness import (
    CACHE, ERAS, CUTS, block_ci, build_panel, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.v0_feature_ablation import V0
from live.build_alpha_beta_decomp import x6, FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP

EMB = pd.Timedelta(days=1)
HL = 60.0
EPS = 1e-6
CFG = dict(steps=300, lr=0.05, rho=1e-3, seed=0)      # fixed a priori, not searched
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")


# ------------------------------------------------------------------ objective + analytic gradient
def forward(theta, X, M, A, K):
    """X [T,N,F] features, M [T,N] mask, A [T,N] residual returns, K [N] per-unit cost (fraction).
    Returns (net series [T], intermediates) for the dollar-neutral L1-normalised book."""
    s = np.einsum("tnf,nf->tn", X, theta) * M
    cnt = M.sum(1, keepdims=True)
    mu = (s.sum(1, keepdims=True) / np.maximum(cnt, 1))
    c = (s - mu) * M
    absc = np.sqrt(c * c + EPS * EPS) * M
    A1 = absc.sum(1, keepdims=True)
    w = c / np.maximum(A1, EPS)
    d = np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))
    absd = np.sqrt(d * d + EPS * EPS)
    gross = (w * A).sum(1)
    cost = (absd * K[None, :]).sum(1)
    n = gross - cost
    return n, dict(s=s, c=c, absc=absc, A1=A1, w=w, d=d, absd=absd, cnt=cnt)


def objective(theta, X, M, A, K, theta0, rho):
    n, st = forward(theta, X, M, A, K)
    n = n[1:]                                       # first bar has no prior weights
    mu, sd = n.mean(), n.std()
    J = mu / sd if sd > 0 else 0.0
    return J - rho * float(((theta - theta0) ** 2).sum()), st, n


def grad(theta, X, M, A, K, theta0, rho):
    n_full, st = forward(theta, X, M, A, K)
    n = n_full[1:]
    T = len(n); mu, sd = n.mean(), n.std()
    if sd <= 0:
        return np.zeros_like(theta)
    J = mu / sd
    dJdn = (1.0 / (T * sd)) * (1.0 - J * (n - mu) / sd)          # [T-1]
    dJdn_full = np.concatenate([[0.0], dJdn])                     # align to bars

    w, d, absd, c, absc, A1 = st["w"], st["d"], st["absd"], st["c"], st["absc"], st["A1"]
    sgn_d = d / np.maximum(absd, EPS)
    # dn_t/dw_t = A_t - K*sgn(d_t) ; and w_t also enters d_{t+1} = w_{t+1}-w_t  -> +K*sgn(d_{t+1})
    gw = dJdn_full[:, None] * (A - K[None, :] * sgn_d)
    gw[:-1] += dJdn_full[1:, None] * (K[None, :] * sgn_d[1:])

    # w = c/A1 with A1 = sum_j sqrt(c_j^2+eps^2)
    sgn_c = c / np.maximum(absc, EPS)
    gc = gw / np.maximum(A1, EPS) - ((gw * c).sum(1, keepdims=True) / np.maximum(A1 ** 2, EPS)) * sgn_c
    gc = gc * M
    # c = s - mean(s) over present names
    gs = (gc - (gc.sum(1, keepdims=True) / np.maximum(st["cnt"], 1))) * M
    g = np.einsum("tn,tnf->nf", gs, X)
    return g - 2.0 * rho * (theta - theta0)


def check_grad(theta, X, M, A, K, theta0, rho, n=6, seed=0):
    rng = np.random.default_rng(seed)
    g = grad(theta, X, M, A, K, theta0, rho)
    errs = []
    for _ in range(n):
        i = rng.integers(0, theta.shape[0]); j = rng.integers(0, theta.shape[1])
        h = 1e-6 * max(1.0, abs(theta[i, j]))
        tp = theta.copy(); tp[i, j] += h
        tm = theta.copy(); tm[i, j] -= h
        num = (objective(tp, X, M, A, K, theta0, rho)[0] - objective(tm, X, M, A, K, theta0, rho)[0]) / (2 * h)
        den = max(abs(num), abs(g[i, j]), 1e-12)
        errs.append(abs(num - g[i, j]) / den)
    return float(np.max(errs))


def train(theta0, X, M, A, K, steps, lr, rho, verbose=False):
    theta = theta0.copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2 = 0.9, 0.999
    best, best_t = -np.inf, theta.copy()
    for t in range(1, steps + 1):
        g = grad(theta, X, M, A, K, theta0, rho)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8)
        if t % 25 == 0 or t == steps:
            J = objective(theta, X, M, A, K, theta0, rho)[0]
            if J > best:
                best, best_t = J, theta.copy()
            if verbose:
                print(f"      step {t:3d}  train J {J:+.4f}", flush=True)
    return best_t, best


# ------------------------------------------------------------------ fold machinery
def fold_tensors(tr, feats, syms):
    """Per-symbol preproc fit on the fold's train rows (identical to gen()); returns tensors + ridge init."""
    sidx = {s: i for i, s in enumerate(syms)}
    times = np.sort(tr["open_time"].unique())
    tidx = {t: i for i, t in enumerate(times)}
    T, Nn, F = len(times), len(syms), len(feats)
    X = np.zeros((T, Nn, F), np.float64); M = np.zeros((T, Nn)); A = np.zeros((T, Nn))
    theta = np.zeros((Nn, F)); pp = {}
    t_end = tr["open_time"].max()
    for sym, gg in tr.groupby("symbol"):
        if sym not in sidx or len(gg) < 300:
            continue
        try:
            st, ht = x6.fit_preproc(gg, feats)
            Xi = x6.apply_preproc(gg, feats, st, ht)
            w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xi, gg["z_res"].to_numpy(), sample_weight=w)
        except Exception:
            continue
        i = sidx[sym]
        rows = np.array([tidx[t] for t in gg["open_time"].to_numpy()])
        X[rows, i, :] = Xi
        M[rows, i] = 1.0
        A[rows, i] = np.nan_to_num(gg["alpha_A"].to_numpy())
        theta[i] = m.coef_
        pp[sym] = (st, ht)
    return X, M, A, theta, pp, sidx


def predict(te, feats, pp, theta, sidx, name):
    rec = []
    for sym, gg in te.groupby("symbol"):
        if sym not in pp:
            continue
        st, ht = pp[sym]
        Xi = x6.apply_preproc(gg, feats, st, ht)
        rec.append(pd.DataFrame({"symbol": sym, "open_time": gg["open_time"].values,
                                 name: Xi @ theta[sidx[sym]]}))
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def cont_book(d, sig, cost):
    """The same continuous dollar-neutral L1 book the objective optimises, for like-for-like evaluation."""
    x = d.dropna(subset=[sig, "alpha_A"]).copy()
    S = x.pivot_table(index="open_time", columns="symbol", values=sig)
    Aa = x.pivot_table(index="open_time", columns="symbol", values="alpha_A").reindex_like(S)
    Mm = S.notna().astype(float)
    C = (S.sub(S.mean(axis=1), axis=0)).fillna(0.0) * Mm
    W = C.div(C.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    g = (W * Aa.fillna(0.0)).sum(axis=1)
    dW = W.diff().abs()
    c, med = cost
    kv = pd.Series([c.get(s, med) for s in W.columns], index=W.columns) / 1e4
    ch = dW.mul(kv, axis=1).sum(axis=1)
    return (g - ch).iloc[1:], g.iloc[1:], (0.5 * dW.sum(axis=1)).iloc[1:]


def main():
    rng = np.random.default_rng(CFG["seed"])
    CT = cost_tiers(); cost10, cmed = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    PAN = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"}).merge(RP, on=["symbol", "open_time"],
                                                                        how="left")
    feats = list(V0)
    syms = sorted(PAN["symbol"].unique())
    K = np.array([float(cost10.get(s, cmed)) for s in syms]) / 1e4

    out = []
    for era in ERAS:
        cuts = CUTS[era]
        print(f"\n===== {era}: {len(cuts)-1} folds =====", flush=True)
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna() & PAN["alpha_A"].notna()]
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
            if tr.empty or te.empty:
                continue
            X, M, A, th0, pp, sidx = fold_tensors(tr, feats, syms)
            if M.sum() == 0:
                continue
            if i == 0 and era == ERAS[0]:
                err = check_grad(th0 + 1e-3 * rng.standard_normal(th0.shape), X, M, A, K, th0, CFG["rho"])
                print(f"  gradient check: max rel err {err:.2e} "
                      f"{'OK' if err < 1e-4 else 'FAIL — aborting'}", flush=True)
                if err >= 1e-4:
                    raise SystemExit("analytic gradient does not match numeric")
            J0 = objective(th0, X, M, A, K, th0, CFG["rho"])[0]
            th, J1 = train(th0, X, M, A, K, CFG["steps"], CFG["lr"], CFG["rho"])
            p_r = predict(te, feats, pp, th0, sidx, "pred_mse")
            p_o = predict(te, feats, pp, th, sidx, "pred_obj")
            if p_r.empty or p_o.empty:
                continue
            m = p_r.merge(p_o, on=["symbol", "open_time"], how="inner")
            out.append(m)
            print(f"  fold {i+1}/{len(cuts)-1} {str(c0.date())}  train J: ridge {J0:+.3f} -> obj {J1:+.3f}"
                  f"   ({M.shape[0]} bars, {int((M.sum(0) > 0).sum())} syms)", flush=True)
    P = pd.concat(out, ignore_index=True)
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P = P.merge(PAN[["symbol", "open_time", "alpha_A"]], on=["symbol", "open_time"], how="left").dropna()
    P.to_parquet(CACHE / "sd2_preds.parquet", index=False)

    # ---------------- G3: rank-IC both eras ----------------
    print("\n============ G3 — rank-IC by era ============", flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = P[(P.open_time >= c0) & (P.open_time < c1)]
        for col in ("pred_mse", "pred_obj"):
            ic = d.groupby("open_time").apply(
                lambda g: spearmanr(g[col], g["alpha_A"]).correlation if len(g) >= 10 else np.nan).dropna()
            print(f"  {era:<8}{col:<10} rank-IC {ic.mean():+.4f}", flush=True)

    # ---------------- G1/G2: held-out net ----------------
    print("\n============ G1/G2 — held-out 2025-01..2026-06, top-40 ADV ============", flush=True)
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    Q = P.merge(A, on=["symbol", "date"], how="left")
    ho = topn(Q[(Q.open_time >= HO0) & (Q.open_time < HO1)].dropna(subset=["tadv"]), "tadv", NTOP)
    series = {}
    for col, lbl in (("pred_mse", "MSE (incumbent)"), ("pred_obj", "trading-objective")):
        net, g, tu = cont_book(ho, col, (cost10, cmed))
        series[lbl] = net
        lo, hi = block_ci(net.to_numpy())
        print(f"  continuous book, {lbl:<20} gross {sharpe(g):+.2f}  net@10k {sharpe(net):+.2f} "
              f"[{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn {tu.mean():.3f}", flush=True)
    a, b = series["MSE (incumbent)"], series["trading-objective"]
    idx = a.index.intersection(b.index)
    dd, lo, hi = paired_block_ci(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
    print(f"\n  G1  Δ(objective − MSE) net@10k {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  "
          f"-> {'PASS' if lo > 0 else 'FAIL'}", flush=True)

    v = ho.rename(columns={"pred_mse": "pred"})
    W, Aa = build(v, "band")
    gb = (W * Aa).sum(axis=0); dWb = W.diff(axis=1).abs()
    kv = pd.Series([float(cost10.get(s, cmed)) for s in W.index], index=W.index)
    nb = (gb - 0.25 * dWb.mul(kv, axis=0).sum(axis=0) / 1e4).iloc[1:]
    lo2, hi2 = block_ci(nb.to_numpy())
    print(f"  incumbent BAND construction  net@10k {sharpe(nb):+.2f} [{lo2:+.2f},{hi2:+.2f}] "
          f"{tag_ci(lo2, hi2)}", flush=True)
    idx2 = nb.index.intersection(b.index)
    dd2, lo3, hi3 = paired_block_ci(nb.loc[idx2].to_numpy(), b.loc[idx2].to_numpy())
    print(f"  G2  Δ(objective − band) net@10k {dd2:+.2f} [{lo3:+.2f},{hi3:+.2f}] {tag_ci(lo3, hi3)}  "
          f"-> {'PASS' if lo3 > 0 else 'FAIL'}", flush=True)
    print("\nSDITER2DONE", flush=True)


if __name__ == "__main__":
    main()
