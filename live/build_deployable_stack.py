"""Capstone: the full deployable stack = BIG-NAMES-ONLY (top-N ADV) × EWMA turnover-control × era-locked beta-hedge.
Net at realistic big-name cost, both eras, with 7d-block-bootstrap CI on the net Sharpe. Nails down the actual
deployable number. Quintile L/S within the tier; net_t = (gross − β·mkt) − turnover·cost. Run: python3 -u -m live.build_deployable_stack
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_turnover_opt import build_W

PYR = 6 * 365.0
TIERS = [20, 40, 999]
LAMS = [0.0, 0.85]
RNG = np.random.default_rng(30)


def adv_cost():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    adv = pd.Series(out)
    return adv, 6 + 36 * (1 - adv.rank(pct=True))


def series(W, R, mask):
    gross = (W * R).sum(axis=0); turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
    mkt = (R * mask).sum(axis=0) / mask.sum(axis=0).replace(0, np.nan)
    return pd.concat([gross.rename("g"), turn.rename("t"), mkt.rename("m")], axis=1).iloc[1:].dropna()


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if len(x) and x.std() > 0 else np.nan


def ci(net, block=42, nb=3000):
    x = net.to_numpy(); n = len(x); nblk = int(np.ceil(n / block)); d = np.empty(nb)
    for i in range(nb):
        st = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        d[i] = sh(x[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    adv, cost = adv_cost()
    ranked = adv.sort_values(ascending=False).index.tolist()
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    store = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d0 = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        store[era] = {}
        for N in TIERS:
            uni = set(ranked[:N]); sub = d0[d0["symbol"].isin(uni)].sort_values(["symbol", "open_time"]).copy()
            sub["rk"] = sub.groupby("open_time")["pred"].rank(pct=True)
            sub["pos"] = np.where(sub["rk"] >= 0.8, 1.0, np.where(sub["rk"] <= 0.2, -1.0, 0.0))
            R = sub.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
            mask = sub.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
            P = sub.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(R)
            tc = float(cost[[s for s in uni if s in cost.index]].mean())
            for lam in LAMS:
                store[era][(N, lam)] = (series(build_W(P, mask, lam), R, mask), tc)
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    print("full deployable stack: big-names × EWMA × era-locked beta-hedge; net Sharpe [7d-block CI]\n", flush=True)
    for era in ("RECENT", "OOS"):
        print(f"===== {era} =====", flush=True)
        print(f"  {'tier':<7}{'λ':<6}{'turn':<7}{'cost':<6}{'grossSh':<9}{'net@model [CI]':<26}{'net@6 [CI]':<24}", flush=True)
        for N in TIERS:
            for lam in LAMS:
                j, tc = store[era][(N, lam)]
                jo, _ = store[other[era]][(N, lam)]
                beta = np.polyfit(jo["m"], jo["g"], 1)[0]          # era-locked hedge beta
                hg = j["g"] - beta * j["m"]
                nm = hg - j["t"] * tc / 1e4; n6 = hg - j["t"] * 6 / 1e4
                lom, him = ci(nm); lo6, hi6 = ci(n6)
                lab = f"top{N}" if N < 999 else "all"
                print(f"  {lab:<7}{lam:<6}{j['t'].mean():<7.2f}{tc:<6.0f}{sh(hg):<+9.2f}"
                      f"{f'{sh(nm):+.2f} [{lom:+.2f},{him:+.2f}]':<26}{f'{sh(n6):+.2f} [{lo6:+.2f},{hi6:+.2f}]':<24}",
                      flush=True)
        print("", flush=True)
    print("STACKDONE", flush=True)


if __name__ == "__main__":
    main()
