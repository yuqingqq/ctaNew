"""Confirmatory run: does the cycle-5 EWMA turnover win STACK with the beta-hedge (i.e. survive in the DEPLOYED
book)? Compare the current deployed construction (top-K=3 band + era-locked beta-hedge) vs (top-K=3 EWMA lam=0.7
+ era-locked beta-hedge), on NET@cost, both eras, with a block-bootstrap CI on the NET@24 Sharpe diff.
Expectation: turnover-control (EWMA) and beta-hedge are separable, so EWMA's ~half-turnover holds and net stays
tie-or-better. Run: python3 -u -m live.build_ewma_hedge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk
from live.build_turnover_opt import build_W

PYR = 6 * 365.0
K, M, LAM = 3, 8, 0.7
COST = [24.0, 12.0, 6.0]
RNG = np.random.default_rng(0)


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def book_series(W, R, mkt):
    gross = (W * R).sum(axis=0)
    turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
    df = pd.concat([gross.rename("g"), turn.rename("t"), mkt.rename("m")], axis=1).iloc[1:].dropna()
    return df


def block_ci(a, b, block=30, nb=3000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) - sh(a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"deployed top-K={K}; band M={M} vs EWMA lam={LAM}; era-locked beta-hedge\n", flush=True)
    store = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
        d["posband"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                       for _, g in d.groupby("symbol", sort=False)])
        d["postopk"] = np.where(d["rhi"] <= K, 1, np.where(d["rlo"] <= K, -1, 0))
        R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
        mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
        mkt = (R * mask).sum(axis=0) / mask.sum(axis=0).replace(0, np.nan)
        Bd = d.pivot_table(index="symbol", columns="open_time", values="posband", fill_value=0.0).reindex_like(R)
        Tk = d.pivot_table(index="symbol", columns="open_time", values="postopk", fill_value=0.0).reindex_like(R)
        store[era] = {"band+hedge": book_series(build_W(Bd, mask, 0.0), R, mkt),
                      "EWMA.7+hedge": book_series(build_W(Tk, mask, LAM), R, mkt)}
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    beta = {e: {b: np.polyfit(store[e][b]["m"], store[e][b]["g"], 1)[0] for b in store[e]} for e in store}
    for era in ("RECENT", "OOS"):
        print(f"===== {era} =====", flush=True)
        net24 = {}
        for b in ("band+hedge", "EWMA.7+hedge"):
            df = store[era][b]; bt = beta[other[era]][b]
            hg = df["g"] - bt * df["m"]                       # era-locked beta-hedge
            gm = hg.mean(); tm = df["t"].mean()
            nets = "  ".join(f"{c:g}:{sh(hg - df['t'] * c / 1e4):+.2f}" for c in COST)
            print(f"    {b:<14} gross {gm*1e4:+.2f}bps | turn {tm:.3f} | break-even {gm*1e4/max(tm,1e-9):5.1f} | "
                  f"grossSh {sh(hg):+.2f} | net@cost {nets}", flush=True)
            net24[b] = pd.Series((hg - df["t"] * 24.0 / 1e4).to_numpy(), index=df.index)
        j = pd.concat([net24["band+hedge"].rename("a"), net24["EWMA.7+hedge"].rename("b")], axis=1).dropna()
        lo, hi = block_ci(j["a"].to_numpy(), j["b"].to_numpy())
        v = "EWMA BETTER" if lo > 0 else ("EWMA worse" if hi < 0 else "tie (equal net, ~half turnover)")
        print(f"  NET@24 Sharpe diff (EWMA - band) 95% CI [{lo:+.2f}, {hi:+.2f}] -> {v}\n", flush=True)
    print("EWMAHEDGEDONE", flush=True)


if __name__ == "__main__":
    main()
