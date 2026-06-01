"""
scope_trend_regime_overlay.py

Regime-switched strategy test:
  - BULL  -> TREND-FOLLOW (long top-3 by per-symbol mom_30d, short bottom-3)
  - ELSE  -> MEAN-REVERT (long top-3 by V0 pred, short bottom-3)

PIT trend-regime detector built from BTC 4h closes (ret_7d/30d/90d, dist-200dMA).
Two detectors compared:
  (a) simple threshold on BTC ret_30d
  (b) KMeans(k=3) on BTC trend features (the trend-axis fix for the failed vol-axis HMM)

Baselines (same windows):
  - mean-rev everywhere
  - mean-rev + FLAT in bull (current best)
  - regime-switched (trend-follow in bull)

Cost 4.5 bps/leg, full rotation per cycle (conservative). Gross + net.
Annualized Sharpe = sqrt(6*365) * mean/std of per-cycle PnL.
Read-only on existing files.
"""
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans

PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_3yr_v0.parquet"
PREDS = "/home/yuqing/ctaNew/research/convexity_portable_2026-05-20/results/_cache/x70_v0_3yr_preds.parquet"
BTC_GLOB = "/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"

COST_BPS_LEG = 4.5
K = 3
ANN = np.sqrt(6 * 365)


def sharpe(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return ANN * x.mean() / x.std()


# ---------------------------------------------------------------- BTC trend
def btc_trend_features():
    fs = sorted(glob.glob(BTC_GLOB))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in fs])
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    # resample to 4h close (last 5m close in each 4h bin)
    c4 = btc["close"].resample("4h").last().dropna()
    # bars per period at 4h: 7d=42, 30d=180, 90d=540, 200d MA=1200
    feat = pd.DataFrame(index=c4.index)
    feat["close"] = c4
    feat["ret_7d"] = c4.pct_change(42)
    feat["ret_30d"] = c4.pct_change(180)
    feat["ret_90d"] = c4.pct_change(540)
    ma200 = c4.rolling(1200, min_periods=600).mean()
    feat["dist_ma200"] = c4 / ma200 - 1.0
    return feat.dropna()


# ---------------------------------------------------------------- load signals
def load_signals():
    preds = pd.read_parquet(PREDS, columns=["symbol", "open_time", "pred", "return_pct", "alpha_A"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    m = (preds["open_time"].dt.hour % 4 == 0) & (preds["open_time"].dt.minute == 0)
    preds = preds[m].copy()

    # per-symbol mom_30d: trailing 30d (180 bars of 4h) return, PIT, from panel close-proxy.
    # panel has return_pct (4h-fwd raw). Build trailing cumulative return per symbol from it.
    # Use panel return_pct shifted so mom_30d at time t = product of past 180 4h returns (PIT, no fwd leak).
    panel = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    mp = (panel["open_time"].dt.hour % 4 == 0) & (panel["open_time"].dt.minute == 0)
    panel = panel[mp].sort_values(["symbol", "open_time"]).copy()
    # return_pct is the FORWARD 4h return realized over [t, t+4h]. The trailing 30d
    # momentum ending AT t (PIT, known at t) = product of forward returns over the prior
    # 180 bars, i.e. returns realized in (t-30d, t]. Those are return_pct shifted by +1 bar
    # (the return realized over the previous bar) cumulated. Use log-sum then shift(1).
    g = panel.groupby("symbol", group_keys=False)
    lr = np.log1p(panel["return_pct"].clip(-0.99, None))
    panel["mom_30d"] = g.apply(lambda d: np.expm1(
        np.log1p(d["return_pct"].clip(-0.99, None)).rolling(180, min_periods=120).sum().shift(1)
    )).values
    mom = panel[["symbol", "open_time", "mom_30d"]]

    df = preds.merge(mom, on=["symbol", "open_time"], how="left")
    return df


# ---------------------------------------------------------------- backtest
def basket_pnl(sub, sig_col, cost):
    """long top-K by sig, short bottom-K. returns raw, alpha net+gross."""
    s = sub.dropna(subset=[sig_col])
    if len(s) < 2 * K:
        return None
    s = s.sort_values(sig_col)
    short = s.head(K)
    long = s.tail(K)
    raw_g = long["return_pct"].mean() - short["return_pct"].mean()
    alp_g = long["alpha_A"].mean() - short["alpha_A"].mean()
    c = cost  # already in return units, applied per leg, full rotation
    return raw_g, raw_g - c, alp_g, alp_g - c


def run(df, btc, label, start=None):
    if start is not None:
        df = df[df["open_time"] >= pd.Timestamp(start, tz="UTC")]
    cycles = sorted(df["open_time"].unique())

    # map regime per cycle from BTC features (PIT: features known at cycle open)
    btc_at = btc.reindex(btc.index.union(cycles)).sort_index().ffill().reindex(cycles)

    # detector (a) simple threshold
    reg_a = pd.Series(index=cycles, dtype=object)
    reg_a[:] = "sideways"
    reg_a[btc_at["ret_30d"] > 0.10] = "bull"
    reg_a[btc_at["ret_30d"] < -0.10] = "bear"

    # detector (b) KMeans k=3 on trend features (fit on this window's cycles, PIT-ish:
    # clustering is unsupervised on standardized trend features; label by mean ret_30d)
    X = btc_at[["ret_7d", "ret_30d", "ret_90d", "dist_ma200"]].copy()
    Xz = (X - X.mean()) / X.std()
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xz.values)
    lab = pd.Series(km.labels_, index=cycles)
    # rank clusters by mean ret_30d -> bear/sideways/bull
    order = btc_at.groupby(lab)["ret_30d"].mean().sort_values().index.tolist()
    names = {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}
    reg_b = lab.map(names)

    # cost in return units
    cost = 2 * K * COST_BPS_LEG / 1e4 / K  # mean over K legs each side: per-leg cost
    # mean(long)-mean(short) each is avg of K legs; full rotation = each leg pays cost.
    # net = gross - (cost per leg)*(legs)/normalization. Since pnl = mean(long)-mean(short),
    # cost per side = mean of K legs each costing COST_BPS_LEG -> COST_BPS_LEG/1e4 per side.
    # two sides => 2*COST_BPS_LEG/1e4.
    cost = 2 * COST_BPS_LEG / 1e4

    by_cyc = {c: g for c, g in df.groupby("open_time")}

    def strat(regimes, bull_mode):
        """bull_mode: 'trend' | 'flat' | None(meanrev everywhere)."""
        out = []
        for c in cycles:
            sub = by_cyc[c]
            r = regimes[c] if regimes is not None else "sideways"
            if bull_mode is not None and r == "bull":
                if bull_mode == "flat":
                    out.append((c, 0.0, 0.0, 0.0, 0.0, r))
                    continue
                res = basket_pnl(sub, "mom_30d", cost)  # trend-follow
            else:
                res = basket_pnl(sub, "pred", cost)  # mean-rev
            if res is None:
                continue
            out.append((c, res[0], res[1], res[2], res[3], r))
        o = pd.DataFrame(out, columns=["cyc", "raw_g", "raw_n", "alp_g", "alp_n", "reg"])
        return o

    results = {}
    # baselines
    results["meanrev_everywhere"] = strat(None, None)
    results["meanrev+flat_bull(a)"] = strat(reg_a, "flat")
    results["meanrev+flat_bull(b)"] = strat(reg_b, "flat")
    results["switched_trend(a_simple)"] = strat(reg_a, "trend")
    results["switched_trend(b_kmeans)"] = strat(reg_b, "trend")

    print(f"\n{'='*78}\nWINDOW: {label}  (n cycles={len(cycles)})\n{'='*78}")
    # regime agreement
    agree = (reg_a.values == reg_b.values).mean()
    ca = pd.Series(reg_a).value_counts().to_dict()
    cb = pd.Series(reg_b).value_counts().to_dict()
    print(f"Detector regime counts  (a simple): {ca}")
    print(f"Detector regime counts  (b kmeans): {cb}")
    print(f"Regime agreement (a vs b): {agree:.1%}")
    nb_a = (reg_a == "bull").sum()
    nb_b = (reg_b == "bull").sum()
    print(f"Bull cycles: a={nb_a} ({nb_a/len(cycles):.0%}), b={nb_b} ({nb_b/len(cycles):.0%})")

    print(f"\n{'strategy':<28} {'rawNetSh':>9} {'alpNetSh':>9} {'rawGrSh':>8} {'alpGrSh':>8} {'totRawNet':>10}")
    for k, o in results.items():
        print(f"{k:<28} {sharpe(o.raw_n):>9.2f} {sharpe(o.alp_n):>9.2f} "
              f"{sharpe(o.raw_g):>8.2f} {sharpe(o.alp_g):>8.2f} {o.raw_n.sum()*1e4:>10.0f}")

    # bull-only sub-analysis: what does the overlay earn vs flat in bull cycles?
    sw = results["switched_trend(a_simple)"]
    bull_pnl = sw[sw.reg == "bull"]["raw_n"]
    print(f"\n[detector a] bull-cycle trend-follow: n={len(bull_pnl)} netRawSh={sharpe(bull_pnl):.2f} "
          f"meanbps={bull_pnl.mean()*1e4:.1f} totbps={bull_pnl.sum()*1e4:.0f}")
    swb = results["switched_trend(b_kmeans)"]
    bull_pnl_b = swb[swb.reg == "bull"]["raw_n"]
    print(f"[detector b] bull-cycle trend-follow: n={len(bull_pnl_b)} netRawSh={sharpe(bull_pnl_b):.2f} "
          f"meanbps={bull_pnl_b.mean()*1e4:.1f} totbps={bull_pnl_b.sum()*1e4:.0f}")
    return results


if __name__ == "__main__":
    print("loading BTC trend features...")
    btc = btc_trend_features()
    print("loading signals (preds + per-symbol mom_30d)...")
    df = load_signals()
    print(f"signals: {len(df)} rows, {df.symbol.nunique()} symbols, "
          f"mom_30d non-null {df.mom_30d.notna().mean():.1%}")

    run(df, btc, "RECENT 12mo (>=2025-05-01)", start="2025-05-01")
    run(df, btc, "FULL 3yr", start=None)
