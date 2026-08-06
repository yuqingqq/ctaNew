"""Research cycle 5: TURNOVER-OPTIMIZED construction (targets NET at retail; user: '24bps is mostly long-tail
slippage'). Cycle-4 hint: continuous smooth weights churned less than top-K (turn 0.18 vs 0.28). Test whether
a turnover-controlled book beats the incumbent hysteresis BAND on NET@cost, both eras, on the SAME incumbent
per-symbol-Ridge predictions.

Levers (all vs incumbent band): weight-EWMA smoothing (keep top-K concentration, ramp positions in/out) and
continuous z-weighting (spread across names = cycle-4 texture). Prior from convex-sizing CLOSED: concentration
is what matters, so spreading may cut gross. Unhedged to ISOLATE the turnover/net tradeoff (beta-hedge is a
separable validated overlay). Unified weight-based turnover (0.25*sum|dW|, 0-1) so all schemes compare like-for-
like. Block-bootstrap CI on the NET@24 Sharpe diff (variant - band), both eras.
Run: python3 -u -m live.build_turnover_opt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk

PYR = 6 * 365.0
K, M = 3, 8
COST = [24.0, 12.0, 6.0]
RNG = np.random.default_rng(0)


def norm_sides(W):
    """Per bar (column): long weights sum to +1, short to -1."""
    pos = W.clip(lower=0); neg = W.clip(upper=0)
    ps = pos.sum().replace(0, np.nan); ns = neg.sum().abs().replace(0, np.nan)
    return pos.div(ps, axis=1).fillna(0.0) + neg.div(ns, axis=1).fillna(0.0)


def smooth(W, lam):
    return W if lam <= 0 else W.T.ewm(alpha=1 - lam, adjust=False).mean().T   # EWMA along time (columns)


def build_W(target, mask, lam):
    """target,mask: names x bars. Normalize -> EWMA-smooth -> mask to universe -> renormalize."""
    W = norm_sides(target)
    W = smooth(W, lam) * mask
    return norm_sides(W)


def series(W, R):
    gross = (W * R).sum(axis=0)                        # per-bar book return
    dW = 0.25 * W.diff(axis=1).abs().sum(axis=0)       # per-bar turnover (0-1)
    j = pd.concat([gross.rename("g"), dW.rename("t")], axis=1).iloc[1:].dropna()
    return j["g"].to_numpy(), j["t"].to_numpy(), j.index


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def block_ci(a, b, block=30, nb=3000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) - sh(a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(name, g, t):
    gm = g.mean()
    be = gm * 1e4 / max(t.mean(), 1e-9)
    nets = "  ".join(f"{c:g}:{sh(g - t * c / 1e4):+.2f}" for c in COST)
    print(f"    {name:<20} gross {gm*1e4:+.2f}bps | turn {t.mean():.3f} | break-even {be:5.1f} | "
          f"grossSh {sh(g):+.2f} | net@cost {nets}", flush=True)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"top-K={K} concentration; unhedged; unified turnover; net CI vs incumbent band(M={M})\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
        d["zc"] = d.groupby("open_time")["pred"].transform(lambda s: ((s - s.mean()) / s.std()).clip(-3, 3))
        posband = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                  for _, g in d.groupby("symbol", sort=False)])
        d["posband"] = posband
        d["postopk"] = np.where(d["rhi"] <= K, 1, np.where(d["rlo"] <= K, -1, 0))
        R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
        mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
        Tk = d.pivot_table(index="symbol", columns="open_time", values="postopk", fill_value=0.0).reindex_like(R)
        Bd = d.pivot_table(index="symbol", columns="open_time", values="posband", fill_value=0.0).reindex_like(R)
        Zc = d.pivot_table(index="symbol", columns="open_time", values="zc", fill_value=0.0).reindex_like(R)
        variants = [
            ("band M=8 (incumbent)", Bd, 0.0),
            ("topK full-rebal",      Tk, 0.0),
            ("topK EWMA lam=.7",     Tk, 0.7),
            ("topK EWMA lam=.85",    Tk, 0.85),
            ("continuous z",         Zc, 0.0),
            ("continuous z EWMA=.7", Zc, 0.7),
        ]
        print(f"===== {era} =====", flush=True)
        store = {}
        for name, tgt, lam in variants:
            W = build_W(tgt, mask, lam)
            g, t, idx = series(W, R)
            report(name, g, t)
            store[name] = pd.Series(g - t * 24.0 / 1e4, index=idx)     # NET@24 per-bar series
        base = store["band M=8 (incumbent)"]
        print("  NET@24 Sharpe diff vs incumbent band (95% CI):", flush=True)
        for name in store:
            if name == "band M=8 (incumbent)":
                continue
            j = pd.concat([base.rename("a"), store[name].rename("b")], axis=1).dropna()
            lo, hi = block_ci(j["a"].to_numpy(), j["b"].to_numpy())
            v = "BETTER" if lo > 0 else ("worse" if hi < 0 else "tie")
            print(f"    {name:<22} [{lo:+.2f}, {hi:+.2f}] -> {v}", flush=True)
        print("", flush=True)
    print("TURNOPTDONE", flush=True)


if __name__ == "__main__":
    main()
