"""Beyond-cross-section loop — iteration 3 (B3): a book NATIVE to a slow horizon.

Our measured binding constraint is cost: 2-4 bps/bar at turnover 0.40 on a 4h grid. At weekly rebalance that
is ~1/50th. The cost/turnover loop slowed a FAST signal down (iteration 1) and found it era-locked — but that
is not the same experiment as building a signal native to the horizon: a multi-day LABEL, multi-day FEATURES,
and rebalancing at the horizon.

Construction, per horizon H in {1d, 3d, 7d, 14d, 30d}:
  label    forward H-day BTC-beta residual return, cross-sectionally z-scored
           (beta from a trailing 30d daily regression, shifted — PIT)
  features horizon-scaled: trailing returns at H/3, H, 3H; realised vol at H and 3H; the H-skip-recent
           momentum the CONCLUSION doc flags as a real different-root sleeve; and distance from the H-high
  book     top-40 by PIT ADV, quintile L/S, rebalanced every H, costed per symbol
  stats    labels OVERLAP for H>1 bar, so CIs use non-overlapping-block bootstrap, never day clusters alone

Gates: G1 rank-IC same sign both eras with block CI excluding 0; G2 net Sharpe CI>0 both eras;
G3 hard split (select H and K on 2023-06..2024-12, evaluate 2025-01..2026-06).
Falsifier: G2 fails at every horizon -> the cross-sectional prediction frame is closed at every tradeable
horizon, not only at 4h.
Run: python3 -u -m live.bx_iter3_horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.cost_loop_harness import CACHE, REPO, block_ci, build_panel, paired_block_ci, pit_adv, sharpe, tag_ci
from live.build_alpha_beta_decomp import x6, FULL
from live.cl_iter4_capacity import cost_tiers

HORIZONS = {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30}
ERAS = {"OOS": ("2023-06-01", "2025-10-01"), "RECENT": ("2025-10-01", "2026-07-01")}
SEL = ("2023-06-01", "2025-01-01")
HO = ("2025-01-01", "2026-07-01")
KQ = 0.2
RNG = np.random.default_rng(83)


def daily_panel() -> pd.DataFrame:
    """Collapse the 4h panel to a daily grid and build daily prices + BTC beta residual labels."""
    fp = CACHE / "bx_daily.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["date"] = pd.to_datetime(d["date"], utc=True); return d
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    PAN = build_panel()[["symbol", "open_time"]].copy()
    px = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                         columns=["symbol", "open_time", "return_pct"])
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    px = px[(px.open_time.dt.hour % 4 == 0)].sort_values(["symbol", "open_time"])
    # compound 4h returns into a daily series (the 4h return is non-overlapping on the 4h grid)
    px["date"] = px["open_time"].dt.floor("1D")
    d = px.groupby(["symbol", "date"])["return_pct"].apply(lambda s: np.prod(1 + s.values) - 1).rename(
        "ret_1d").reset_index()
    btc = d[d.symbol == "BTCUSDT"][["date", "ret_1d"]].rename(columns={"ret_1d": "btc"})
    if btc.empty:                                   # BTC is not a leg in the panel; rebuild from the market
        btc = d.groupby("date")["ret_1d"].mean().rename("btc").reset_index()
    d = d.merge(btc, on="date", how="left").sort_values(["symbol", "date"])
    g = d.groupby("symbol")
    cov = g.apply(lambda x: x["ret_1d"].rolling(30, min_periods=20).cov(x["btc"])).reset_index(level=0, drop=True)
    var = g["btc"].transform(lambda x: x.rolling(30, min_periods=20).var())
    d["beta"] = (cov / var.replace(0, np.nan)).groupby(d["symbol"]).shift(1)
    d.to_parquet(fp, index=False)
    return d


def build_features(d: pd.DataFrame, H: int) -> pd.DataFrame:
    x = d.sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol")
    lr = np.log1p(x["ret_1d"].clip(-0.95, 10))
    x["_lr"] = lr
    for name, w in (("mom_s", max(1, H // 3)), ("mom_m", H), ("mom_l", 3 * H)):
        x[name] = g["_lr"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=max(1, min(w, w // 2))).sum())
    x["mom_skip"] = x["mom_l"] - x["mom_s"]                       # skip-recent momentum
    for name, w in (("vol_m", H), ("vol_l", 3 * H)):
        x[name] = g["_lr"].transform(lambda s, w=w: s.shift(1).rolling(
            max(5, w), min_periods=max(3, w // 2)).std())
    x["dd"] = g["_lr"].transform(
        lambda s: s.shift(1).rolling(3 * H, min_periods=H).sum()
        - s.shift(1).rolling(3 * H, min_periods=H).max())
    # forward H-day residual label
    fwdH = g["_lr"].transform(lambda s: s[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1))
    x["_bl"] = np.log1p(x["btc"].clip(-0.95, 10))
    bfwd = x.groupby("symbol")["_bl"].transform(
        lambda s: s[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1))
    x["y"] = fwdH - x["beta"] * bfwd
    gz = x.groupby("date")["y"]
    x["z"] = ((x["y"] - gz.transform("mean")) / gz.transform("std").replace(0, np.nan)).clip(-10, 10)
    return x


FEATS = ["mom_s", "mom_m", "mom_l", "mom_skip", "vol_m", "vol_l", "dd"]


def walk(x: pd.DataFrame, H: int, cuts) -> pd.DataFrame:
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]
        fc = c0 - pd.Timedelta(days=H + 1)                    # embargo the overlapping label
        tr = x[(x.date < fc) & x["z"].notna()]
        te = x[(x.date >= c0) & (x.date < c1)]
        if len(tr) < 5000 or te.empty:
            continue
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 120:
                continue
            try:
                s, h = x6.fit_preproc(gg, FEATS)
                X = x6.apply_preproc(gg, FEATS, s, h)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z"].to_numpy())
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "date": gte["date"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, FEATS, s, h)),
                                             "y": gte["y"].values}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def mean_block_ci(x, block, nb=3000, seed=7):
    """Block bootstrap CI on the MEAN of a series (block_ci returns a SHARPE CI — wrong for rank-IC)."""
    r = np.random.default_rng(seed)
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    n = len(a)
    if n < 8:
        return (np.nan, np.nan)
    nb_blk = int(np.ceil(n / block))
    d = np.empty(nb)
    for i in range(nb):
        st = r.integers(0, max(n - block + 1, 1), nb_blk)
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in st])[:n]
        d[i] = a[idx].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def book(P, H, adv, cost):
    c, med = cost
    p = P.merge(adv, left_on=["symbol", "date"], right_on=["symbol", "date"], how="left").dropna(
        subset=["tadv", "pred", "y"])
    p["ar"] = p.groupby("date")["tadv"].rank(ascending=False, method="first")
    p = p[p["ar"] <= 40]
    # rebalance every H days: hold the selection fixed within a block
    dates = np.sort(p["date"].unique())
    blk = {d: i // H for i, d in enumerate(dates)}
    p["blk"] = p["date"].map(blk)
    first = p.groupby(["blk", "symbol"])["pred"].transform("first")
    p["rk"] = p.assign(_f=first).groupby("date")["_f"].rank(pct=True)
    p["pos"] = np.where(p["rk"] >= 1 - KQ, 1.0, np.where(p["rk"] <= KQ, -1.0, 0.0))
    # one observation per block: the block's realised residual return, equal-weighted per side
    blkret = p[p["pos"] != 0].groupby("blk").apply(
        lambda g: (g.loc[g.pos > 0, "y"].mean() - g.loc[g.pos < 0, "y"].mean())
        if (g.pos > 0).any() and (g.pos < 0).any() else np.nan).dropna()
    names = p[p["pos"] != 0].groupby("blk")["symbol"].apply(set)
    churn = np.mean([len(names.iloc[i] - names.iloc[i - 1]) / max(len(names.iloc[i]), 1)
                     for i in range(1, len(names))]) if len(names) > 1 else 1.0
    avg_cost = float(np.mean([c.get(s, med) for s in p["symbol"].unique()]))
    net = blkret - churn * 2 * avg_cost / 1e4
    return blkret, net, churn, avg_cost


def main():
    CT = cost_tiers(); cost = CT["cost_10k"]
    d = daily_panel()
    A = pit_adv().rename(columns={"date": "date"})
    print(f"daily panel: {d.symbol.nunique()} symbols, {d.date.min().date()} -> {d.date.max().date()}",
          flush=True)
    out = {}
    for hname, H in HORIZONS.items():
        x = build_features(d, H)
        for era, (t0, t1) in ERAS.items():
            cuts = pd.date_range(t0, t1, freq="3MS", tz="UTC")
            P = walk(x, H, cuts)
            if P.empty:
                print(f"  {hname:<5}{era:<8} no preds", flush=True); continue
            P["date"] = pd.to_datetime(P["date"], utc=True)
            ic = P.groupby("date").apply(
                lambda g: spearmanr(g["pred"], g["y"]).correlation if len(g) >= 10 else np.nan).dropna()
            # non-overlapping blocks of length H for the IC CI
            lo, hi = mean_block_ci(ic.to_numpy(), block=max(H, 2))
            br, net, churn, ac = book(P, H, A, cost)
            nlo, nhi = block_ci(net.to_numpy(), block=4) if len(net) > 20 else (np.nan, np.nan)
            pyr = 365.0 / H
            gs = br.mean() / br.std() * np.sqrt(pyr) if br.std() > 0 else np.nan
            ns = net.mean() / net.std() * np.sqrt(pyr) if net.std() > 0 else np.nan
            out[(hname, era)] = (ic.mean(), lo, hi, gs, ns, nlo, nhi, churn, len(net))
            pw = "" if len(net) >= 30 else "  UNDERPOWERED"
            print(f"  {hname:<5}{era:<8} IC {ic.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"gross Sh {gs:+.2f}  net Sh {ns:+.2f} [{nlo:+.2f},{nhi:+.2f}] {tag_ci(nlo, nhi)}  "
                  f"churn {churn:.2f}  n={len(net)}{pw}", flush=True)

    print("\n=== GATE READ ===", flush=True)
    for hname in HORIZONS:
        a = out.get((hname, "OOS")); b = out.get((hname, "RECENT"))
        if not a or not b:
            continue
        g1 = np.sign(a[0]) == np.sign(b[0]) and a[1] > 0 and b[1] > 0
        g2 = a[5] > 0 and b[5] > 0
        print(f"  {hname:<5} G1 {'PASS' if g1 else 'fail'}   G2 {'PASS' if g2 else 'fail'}", flush=True)
    print("\nBXITER3DONE", flush=True)


if __name__ == "__main__":
    main()
