"""Decisive validation of the mom_14_3 lead.
A. QUARTERLY orthogonal-IC stability (acid test): residualize mom_14_3 vs edge, IC vs fwd alpha, by quarter.
   Persistent-positive across quarters = real; sign-flips = period artifact (the OB-flow trap).
B. λ-ROBUSTNESS + honest CI: factor net@24 and blend-vs-strategy ΔSh with 7-day-block bootstrap CI, for
   λ in {0.7,0.8,0.85,0.9}, both eras. Does the OOS diversification survive across λ and a conservative CI?
Run: python3 -u -m live.build_mom_validate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk
from live.build_turnover_opt import build_W

PYR = 6 * 365.0
K, M = 3, 8
CTRL = ["return_1d", "ret_3d", "rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]
LAMS = [0.7, 0.8, 0.85, 0.9]
RNG = np.random.default_rng(1)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def blk_ci(a, b, block=42, nb=3000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) / np.sqrt(PYR) - sh(a[idx]) / np.sqrt(PYR)
    return float(np.percentile(d, 2.5) * np.sqrt(PYR)), float(np.percentile(d, 97.5) * np.sqrt(PYR))


def net_series(W, R, c):
    gross = (W * R).sum(axis=0); turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
    j = pd.concat([gross.rename("g"), turn.rename("t")], axis=1).iloc[1:].dropna()
    return (j["g"] - j["t"] * c / 1e4), j["g"], j["t"].mean()


def factor_pos(PAN):
    d = PAN.dropna(subset=["mom_14_3"] + CTRL)
    parts = []
    for t, g in d.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in CTRL])
        b, *_ = np.linalg.lstsq(X, g["mom_14_3"].to_numpy(), rcond=None)
        rk = pd.Series(g["mom_14_3"].to_numpy() - X @ b).rank(pct=True).to_numpy()
        p = np.where(rk >= 0.8, 1, np.where(rk <= 0.2, -1, 0)).astype(np.int8)
        parts.append(pd.DataFrame({"symbol": g["symbol"].to_numpy(), "open_time": t, "pf": p}))
    return pd.concat(parts)


def main():
    PAN = build_panel()
    miss = [c for c in set(["return_pct"] + CTRL) if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    PAN["mom_14_3"] = PAN.groupby("symbol")["return_pct"].transform(lambda s: s.shift(18).rolling(66).sum())

    # ---- PART A: quarterly orthogonal IC ----
    d = PAN.dropna(subset=["mom_14_3"] + CTRL + ["alpha_vs_btc_realized"])
    rows = []
    for t, g in d.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in CTRL])
        b, *_ = np.linalg.lstsq(X, g["mom_14_3"].to_numpy(), rcond=None)
        resid = g["mom_14_3"].to_numpy() - X @ b
        rows.append((t, spearmanr(resid, g["alpha_vs_btc_realized"]).correlation))
    P = pd.DataFrame(rows, columns=["t", "ic"]).dropna().set_index("t")
    P["q"] = P.index.to_period("Q")
    q = P.groupby("q")["ic"].mean()
    print("PART A — quarterly ORTHOGONAL IC of mom_14_3 (persistent + = real; sign-flips = artifact):", flush=True)
    print("  " + "  ".join(f"{str(k)[2:]}:{v:+.3f}" for k, v in q.items()), flush=True)
    pos = (q > 0).sum()
    print(f"  quarters positive: {pos}/{len(q)}  | mean {q.mean():+.4f} | worst {q.min():+.3f}", flush=True)

    # ---- PART B: λ-robustness + honest CI ----
    FP = factor_pos(PAN)
    print("\nPART B — λ-robustness, factor net@24 + blend ΔSh (7d-block CI), both eras:", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        base = pred.merge(PAN[["symbol", "open_time", "return_pct"]], on=["symbol", "open_time"], how="inner") \
            .dropna().sort_values(["symbol", "open_time"])
        base["rhi"] = base.groupby("open_time")["pred"].rank(ascending=False, method="first")
        base["n"] = base.groupby("open_time")["pred"].transform("size"); base["rlo"] = base["n"] + 1 - base["rhi"]
        base["ps"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                     for _, g in base.groupby("symbol", sort=False)])
        d2 = base.merge(FP, on=["symbol", "open_time"], how="left")
        R = d2.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
        mask = d2.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
        Ps = d2.pivot_table(index="symbol", columns="open_time", values="ps", fill_value=0.0).reindex_like(R)
        Pf = d2.pivot_table(index="symbol", columns="open_time", values="pf", fill_value=0.0).reindex_like(R)
        _, gs, _ = net_series(build_W(Ps, mask, 0.0), R, 0.0)
        print(f"  {era}:", flush=True)
        for lam in LAMS:
            _, gf, ftrn = net_series(build_W(Pf, mask, lam), R, 0.0)
            nf, _, _ = net_series(build_W(Pf, mask, lam), R, 24.0)
            j = pd.concat([gs.rename("s"), nf.rename("f")], axis=1).dropna()
            snn = j["s"] / j["s"].std(); bl = 0.5 * snn + 0.5 * (j["f"] / j["f"].std())
            lo, hi = blk_ci(snn.to_numpy(), bl.to_numpy())
            tag = "DIV" if lo > 0 else ("hurt" if hi < 0 else "tie")
            print(f"    λ={lam} turn {ftrn:.3f} | factor net@24 Sh {sh(nf):+.2f} | blend ΔSh@24 [{lo:+.2f},{hi:+.2f}] {tag}",
                  flush=True)
    print("\nMOMVALDONE", flush=True)


if __name__ == "__main__":
    main()
