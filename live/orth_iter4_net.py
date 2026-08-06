"""ORTHOGONAL-DATA loop iter 4: is tt_pos_chg_1d a NET-VIABLE orthogonal diversifier once we control turnover
(positioning changes slowly -> compressible, cf. cycle-5 EWMA) and/or trade at fee-tier? Build the factor as an
EWMA-smoothed quintile L/S book (oriented long-low-resid / short-high-resid = the stable negative-IC sign), and:
  (a) factor NET@{24,12,6}bps Sharpe for lambda in {0, .85}, both eras;
  (b) does a NET equal-vol BLEND (strategy + factor) beat strategy-alone NET at each cost (block CI), both eras.
Strategy = per-symbol Ridge -> top-K=3 band. Run: python3 -u -m live.orth_iter4_net
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk
from live.build_turnover_opt import build_W
from live.orthogonal_harness import build_panel_with_metrics, CONTROLS

PYR = 6 * 365.0
K, M = 3, 8
FAC = "tt_pos_chg_1d"
COSTS = [24.0, 12.0, 6.0]
LAMS = [0.0, 0.85]
RNG = np.random.default_rng(4)


def sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def block_ci(a, b, block=30, nb=3000):
    n = len(a); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(b[idx]) - sh(a[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def net_series(W, R, c):
    gross = (W * R).sum(axis=0)
    turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
    j = pd.concat([gross.rename("g"), turn.rename("t")], axis=1).iloc[1:].dropna()
    return (j["g"] - j["t"] * c / 1e4), j["t"].mean()


def factor_pos(d):
    ctrl = [c for c in CONTROLS if c in d.columns]
    d = d.dropna(subset=[FAC] + ctrl)
    parts = []
    for t, g in d.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in ctrl])
        b, *_ = np.linalg.lstsq(X, g[FAC].to_numpy(), rcond=None)
        rk = pd.Series(g[FAC].to_numpy() - X @ b).rank(pct=True).to_numpy()
        p = np.where(rk <= 0.2, 1, np.where(rk >= 0.8, -1, 0)).astype(np.int8)   # oriented: long low-resid
        parts.append(pd.DataFrame({"symbol": g["symbol"].to_numpy(), "open_time": t, "pf": p}))
    return pd.concat(parts)


def main():
    PAN = build_panel_with_metrics()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    FP = factor_pos(PAN.merge(RP, on=["symbol", "open_time"], how="inner"))
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
        d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
        d["ps"] = np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                                  for _, g in d.groupby("symbol", sort=False)])
        d = d.merge(FP, on=["symbol", "open_time"], how="left")
        R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
        mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
        Ps = d.pivot_table(index="symbol", columns="open_time", values="ps", fill_value=0.0).reindex_like(R)
        Pf = d.pivot_table(index="symbol", columns="open_time", values="pf", fill_value=0.0).reindex_like(R)
        Ws = build_W(Ps, mask, 0.0)
        print(f"===== {era} =====", flush=True)
        for lam in LAMS:
            Wf = build_W(Pf, mask, lam)
            row = []
            for c in COSTS:
                ns, st = net_series(Ws, R, c)
                nf, ft = net_series(Wf, R, c)
                j = pd.concat([ns.rename("s"), nf.rename("f")], axis=1).dropna()
                sn = j["s"] / j["s"].std(); fn = j["f"] / j["f"].std()
                blend = 0.5 * sn + 0.5 * fn
                lo, hi = block_ci(sn.to_numpy(), blend.to_numpy())
                tag = "DIV" if lo > 0 else ("hurt" if hi < 0 else "tie")
                row.append(f"c{c:g}: facSh {sh(j['f']):+.2f} strat {sh(j['s']):+.2f} blend {sh(blend):+.2f} "
                           f"Δ[{lo:+.2f},{hi:+.2f}]{tag}")
                if c == COSTS[0]:
                    ftrn = ft
            print(f"  factor λ={lam:<4} (turn {ftrn:.3f}):", flush=True)
            for r in row:
                print(f"      {r}", flush=True)
        print("", flush=True)
    print("ORTHNETDONE", flush=True)


if __name__ == "__main__":
    main()
