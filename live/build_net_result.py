"""THE net-of-cost result for the validated PIT top-40 book. Beta-neutral (in-era) alpha, per-bar turnover,
net Sharpe at a cost grid with day-clustered CI, both eras. Raw quintile + EWMA λ=0.85 turnover-control.
Realistic majors taker cost ~6-10bps (model); 24 shown only as the full-universe-retail reference.
Run: python3 -u -m live.build_net_result
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_turnover_opt import build_W

PYR = 6 * 365.0
N = 40
COSTS = [24, 12, 8, 6]
RNG = np.random.default_rng(11)


def trailing_adv():
    frames = []
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            if not isinstance(d.index, pd.DatetimeIndex):
                continue
            dv = (d["total_volume"] * d["vwap"]).sort_index()
            tadv = dv.resample("1D").sum().rolling(30, min_periods=10).mean().shift(1)
            frames.append(pd.DataFrame({"symbol": sym, "date": tadv.index, "tadv": tadv.values}))
        except Exception:
            pass
    A = pd.concat(frames, ignore_index=True); A["date"] = pd.to_datetime(A["date"], utc=True)
    return A.dropna(subset=["tadv"])


def day_ci(vals, times, stat, nb=3000):
    dd = pd.DatetimeIndex(times).floor("1D")
    g = [x.to_numpy() for _, x in pd.Series(np.arange(len(times))).groupby(dd.values)]
    out = np.empty(nb)
    for i in range(nb):
        idx = np.concatenate([g[k] for k in RNG.integers(0, len(g), len(g))]); s = vals[idx]
        out[i] = (s.mean() * 1e4) if stat == "mean" else (s.mean() / s.std() * np.sqrt(PYR) if s.std() > 0 else 0)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if x.std() > 0 else np.nan


def main():
    tadv = trailing_adv()
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print(f"PIT top-{N} book, beta-neutral (in-era); net Sharpe [day-CI] at cost; gross for reference\n", flush=True)
    for era, cuts in (("OOS", OOS_CUTS), ("RECENT", RECENT_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts); pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        d["date"] = d["open_time"].dt.floor("1D")
        d = d.merge(tadv, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
        d["advrank"] = d.groupby("open_time")["tadv"].rank(ascending=False, method="first")
        d = d[d["advrank"] <= N].copy()
        d["rk"] = d.groupby("open_time")["pred"].rank(pct=True)
        d["pos"] = np.where(d["rk"] >= 0.8, 1.0, np.where(d["rk"] <= 0.2, -1.0, 0.0))
        R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").fillna(0.0)
        mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
        P = d.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(R)
        print(f"===== {era} =====", flush=True)
        for lam in (0.0, 0.85):
            W = build_W(P, mask, lam)
            gross = (W * R).sum(axis=0); turn = 0.25 * W.diff(axis=1).abs().sum(axis=0)
            mkt = (R * mask).sum(axis=0) / mask.sum(axis=0).replace(0, np.nan)
            j = pd.concat([gross.rename("g"), turn.rename("t"), mkt.rename("m")], axis=1).iloc[1:].dropna()
            beta = np.polyfit(j["m"], j["g"], 1)[0]
            al = (j["g"] - beta * j["m"])
            t = j.index.to_numpy()
            glo, ghi = day_ci(al.to_numpy(), t, "sharpe")
            cells = []
            for c in COSTS:
                net = (al - j["t"] * c / 1e4).to_numpy()
                lo, hi = day_ci(net, t, "sharpe")
                tag = "" if lo > 0 else ("(spans0)" if hi > 0 else "(neg)")
                cells.append(f"c{c}: {sh(net):+.2f}[{lo:+.2f},{hi:+.2f}]{tag}")
            print(f"  λ={lam:<4} turn {j['t'].mean():.2f} | gross Sh {sh(al):+.2f}[{glo:+.2f},{ghi:+.2f}] | "
                  + "  ".join(cells), flush=True)
        print("", flush=True)
    print("NETRESULTDONE", flush=True)


if __name__ == "__main__":
    main()
