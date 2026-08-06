"""Research cycle 4: PARAMETRIC PORTFOLIO POLICIES (Brandt, Santa-Clara & Valkanov 2009 RFS).
Different paradigm: skip return-prediction; parameterize weights directly as w_i = (1/N) theta' z_i
(z = cross-sectionally standardized characteristics) and choose theta to maximize the book's utility.
With mean-variance utility the book return is theta'f_t (f = characteristic-factor returns), so the optimum
is CLOSED FORM theta* = Sigma_f^{-1} mu_f -- estimated WALK-FORWARD (per cut-window, expanding, 1d embargo,
Ledoit-Wolf shrinkage since our 14 features are highly redundant -> Sigma_f near-singular). PPP jointly
optimizes signal COMBINATION + factor COVARIANCE, which predict-then-sort ignores.

Same 14 V0_LEAN features & same eval windows as the incumbent (per-symbol Ridge -> top-K=3 band -> era-locked
beta-hedge). Compare gross Sharpe, turnover, net-at-cost, and paired block-bootstrap CI on Sharpe diff.
Variants: PPP-MV (Sigma^{-1}mu, LW-shrunk) and PPP-mean (theta=mu, ignore covariance = robust baseline).
Run: python3 -u -m live.build_ppp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover as inc_turnover

PYR = 6 * 365.0
EMB = pd.Timedelta(days=1)
K, M = 3, 8
COST_GRID = [24.0, 12.0, 6.0]
RNG = np.random.default_rng(0)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR)


def block_ci(a, b, block=30, nb=3000):
    """Paired block-bootstrap CI on Sharpe(b) - Sharpe(a), per-bar aligned."""
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        aa, bb = a[idx], b[idx]
        d[i] = bb.mean() / bb.std() - aa.mean() / aa.std()
    d *= np.sqrt(PYR)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def standardize(d, feats):
    g = d.groupby("open_time")
    Z = np.empty((len(d), len(feats)))
    for j, f in enumerate(feats):
        mu = g[f].transform("mean"); sd = g[f].transform("std").replace(0, np.nan)
        Z[:, j] = ((d[f] - mu) / sd).clip(-3, 3).to_numpy()
    return np.nan_to_num(Z, nan=0.0)


def factor_returns(codes, ntimes, Z, r):
    """f_{c,t} = (1/N_t) sum_i z_{c,i,t} r_{i,t}."""
    N = np.bincount(codes, minlength=ntimes).astype(float)
    F = np.column_stack([np.bincount(codes, Z[:, j] * r, ntimes) for j in range(Z.shape[1])])
    return F / np.maximum(N, 1)[:, None]


def moments(Ftr, hl_bars):
    """(mu, Sigma) of training factor returns. hl_bars=None -> equal-weight + Ledoit-Wolf shrinkage;
    else exponential time-decay (matches incumbent HL=60d) + shrink-to-diagonal (LW doesn't take weights)."""
    if hl_bars is None:
        return Ftr.mean(0), LedoitWolf().fit(Ftr).covariance_
    n = len(Ftr); K = Ftr.shape[1]
    w = np.exp(-(n - 1 - np.arange(n)) / hl_bars); w /= w.sum()
    mu = (w[:, None] * Ftr).sum(0)
    Fc = Ftr - mu
    Sig = (Fc.T * w) @ Fc
    tgt = (np.trace(Sig) / K) * np.eye(K)       # Ledoit-Wolf identity target -> guarantees PD
    return mu, 0.7 * Sig + 0.3 * tgt


def ppp_book(d, Z, F, ftimes, cuts, mode, hl_bars=None):
    """Walk-forward theta per window; per-bar dollar-neutral book scaled to gross=2; return per-bar
    (times, book_ret, mkt, turnover). d rows carry symbol/open_time/return_pct; Z aligned to d."""
    otidx = pd.DatetimeIndex(d["open_time"]); r = d["return_pct"].to_numpy(); sym = d["symbol"].to_numpy()
    rows = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        trm = np.asarray(ftimes < fc)
        if trm.sum() < 200:
            continue
        mu, Sig = moments(F[trm], hl_bars)
        theta = mu if mode == "mean" else np.linalg.solve(Sig, mu)
        theta = theta / (np.linalg.norm(theta) + 1e-12)
        em = np.asarray((otidx >= c0) & (otidx < c1))
        if not em.any():
            continue
        a = Z[em] @ theta                                 # raw tilt; sum over names per bar = 0 (z demeaned)
        sub = pd.DataFrame({"open_time": otidx[em], "symbol": sym[em], "a": a, "r": r[em]})
        rows.append(sub)
    if not rows:
        return pd.DatetimeIndex([]), np.array([]), np.array([]), 0.0
    D = pd.concat(rows, ignore_index=True)
    gac = D.groupby("open_time")["a"].transform(lambda s: 0.5 * np.abs(s).sum())
    D["w"] = D["a"] / gac.replace(0, np.nan)              # scale so sum|w| = 2 (1 long / 1 short unit)
    D = D.dropna(subset=["w"])
    per = D.groupby("open_time").apply(lambda g: pd.Series({
        "book": float((g["w"] * g["r"]).sum()),
        "mkt": float(g["r"].mean()), "n": len(g)}), include_groups=False)
    per = per[per["n"] >= 10].sort_index()
    W = D.pivot_table(index="open_time", columns="symbol", values="w", fill_value=0.0).sort_index()
    W = W.loc[per.index]
    turn = 0.25 * float(W.diff().abs().sum(axis=1).mean())   # 0-1 scale, matches incumbent convention
    return per.index, per["book"].to_numpy(), per["mkt"].to_numpy(), turn


def incumbent(PAN, RP, cuts):
    pred = gen_pred(PAN, list(V0), cuts)
    pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
    d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
    d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
    d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
    pos = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                          for _, g in d.groupby("symbol", sort=False)])
    rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(d["open_time"], sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    mkt = sa[ok] / np.maximum(na[ok], 1)
    turn = inc_turnover(d["open_time"].to_numpy("datetime64[ns]"), d["symbol"].to_numpy(), pos)
    return uniq[ok], ls, mkt, turn


def netline(name, times, book, mkt, turn, beta):
    hed = book - beta * mkt
    gm, gsd = hed.mean(), hed.std()
    nets = "  ".join(f"{c:g}:{(gm - turn*c/1e4)/gsd*np.sqrt(PYR):+.2f}" for c in COST_GRID)
    be = gm * 1e4 / max(turn, 1e-9)
    print(f"    {name:<22} grossSh {sh(hed):+.2f} | {gm*1e4:+.2f}bps | turn {turn:.2f} | "
          f"break-even {be:5.1f} | net@cost {nets}", flush=True)
    return pd.Series(hed, index=times)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    d = PAN.merge(RP, on=["symbol", "open_time"], how="inner").dropna(subset=list(V0) + ["return_pct"])
    d = d.sort_values(["open_time", "symbol"]).reset_index(drop=True)
    Z = standardize(d, list(V0))
    codes, uniq = pd.factorize(d["open_time"], sort=True)
    F = factor_returns(codes, len(uniq), Z, d["return_pct"].to_numpy())
    print(f"panel {len(d):,} rows | {d.symbol.nunique()} syms | K={len(V0)} chars | {len(uniq)} bars\n", flush=True)

    HL_TD = 360  # 60 days * 6 bars/day -> matches incumbent HL=60d time-decay
    VARIANTS = [("PPP-MV (equal)", "mv", "mv", None), ("PPP-mean (equal)", "mean", "mn", None),
                ("PPP-MV-td60", "mv", "mvT", HL_TD), ("PPP-mean-td60", "mean", "mnT", HL_TD)]
    store = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        store[era] = {"inc": incumbent(PAN, RP, cuts)}
        for _, mode, key, hl in VARIANTS:
            store[era][key] = ppp_book(d, Z, F, uniq, cuts, mode, hl)
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    keys = ["inc"] + [v[2] for v in VARIANTS]
    beta = {era: {k: np.polyfit(store[era][k][2], store[era][k][1], 1)[0] for k in keys} for era in store}

    print(f"deployed top-K={K} band M={M}; PPP dollar-neutral gross=2; era-locked beta-hedge; "
          f"td60 = HL60d time-decay (matches incumbent)\n", flush=True)
    for era in ("RECENT", "OOS"):
        s = store[era]; b = beta[other[era]]
        print(f"===== {era} =====", flush=True)
        inc_s = netline("incumbent(Ridge+band)", *s["inc"], b["inc"])
        for label, mode, key, hl in VARIANTS:
            ps = netline(label, *s[key], b[key])
            j = pd.concat([inc_s.rename("i"), ps.rename("p")], axis=1).dropna()
            lo, hi = block_ci(j["i"].to_numpy(), j["p"].to_numpy())
            v = "PPP better" if lo > 0 else ("PPP worse" if hi < 0 else "CI spans 0 (tie)")
            print(f"      {label} - incumbent grossSharpe 95% CI [{lo:+.2f}, {hi:+.2f}] -> {v}", flush=True)
        print("", flush=True)
    print("PPPDONE", flush=True)


if __name__ == "__main__":
    main()
