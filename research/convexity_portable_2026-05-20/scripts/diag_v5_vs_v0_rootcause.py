"""
Root-cause: V5 has HIGHER full-cross-section IC than V0 but WORSE portfolio Sharpe.
Both: Ridge per-symbol, predict 4h-fwd BTC-beta-removed alpha-residual.
Trade: every 4h, long top-K=3 by pred, short bottom-K=3.

Investigate: per-cycle IC dist, tail-pick quality, decile location of IC,
concentration, turnover, net-beta. Read-only.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path("/home/yuqing/ctaNew/research/convexity_portable_2026-05-20")
V0P = ROOT / "results/_cache/x70_v0_3yr_preds.parquet"
V5P = ROOT / "results/_cache/x78_v5_single_preds.parquet"
K = 3
ANN = np.sqrt(365 * 6)  # 4h cycles per year (6 per day)


def load(p):
    df = pd.read_parquet(p, columns=["symbol", "open_time", "alpha_A", "return_pct", "pred", "fold"])
    ot = pd.to_datetime(df["open_time"], utc=True)
    df = df[(ot.dt.hour % 4 == 0) & (ot.dt.minute == 0)].copy()
    df["open_time"] = ot[df.index]
    return df.dropna(subset=["pred", "alpha_A"])


def per_cycle_ic(df):
    ics = []
    for _, g in df.groupby("open_time"):
        if len(g) < 8:
            continue
        r = spearmanr(g["pred"], g["alpha_A"]).correlation
        if np.isfinite(r):
            ics.append(r)
    ics = np.array(ics)
    return ics


def tail_spread(df, use="alpha_A"):
    """Per cycle: mean(top-K alpha) - mean(bot-K alpha) by pred rank. Also long/short legs."""
    rows = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * K + 2:
            continue
        gs = g.sort_values("pred")
        bot = gs.head(K)
        top = gs.tail(K)
        rows.append({
            "open_time": t,
            "long_leg": top[use].mean(),
            "short_leg": bot[use].mean(),
            "spread": top[use].mean() - bot[use].mean(),
            "top_syms": tuple(top["symbol"].values),
            "bot_syms": tuple(bot["symbol"].values),
        })
    return pd.DataFrame(rows)


def decile_profile(df):
    """Within each cycle assign pred-decile (0..9), avg realized alpha_A per decile."""
    parts = []
    for t, g in df.groupby("open_time"):
        if len(g) < 20:
            continue
        gg = g.copy()
        gg["dec"] = pd.qcut(gg["pred"].rank(method="first"), 10, labels=False)
        parts.append(gg[["dec", "alpha_A"]])
    allp = pd.concat(parts)
    return allp.groupby("dec")["alpha_A"].mean()


def sharpe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.std(ddof=1) == 0 or len(x) < 2:
        return np.nan
    return x.mean() / x.std(ddof=1) * ANN


def basket_turnover(ts):
    """ts: DataFrame with top_syms/bot_syms sorted by time. Jaccard churn per side."""
    ts = ts.sort_values("open_time").reset_index(drop=True)
    churn = []
    for i in range(1, len(ts)):
        prev = set(ts.loc[i - 1, "top_syms"]) | set(ts.loc[i - 1, "bot_syms"])
        cur = set(ts.loc[i, "top_syms"]) | set(ts.loc[i, "bot_syms"])
        # fraction of 2K names that changed
        changed = len(cur - prev) / (2 * K)
        churn.append(changed)
    return np.mean(churn)


def sel_freq(ts):
    from collections import Counter
    c = Counter()
    for _, r in ts.iterrows():
        for s in r["top_syms"]:
            c[s] += 1
        for s in r["bot_syms"]:
            c[s] += 1
    return c


def main():
    v0 = load(V0P)
    v5 = load(V5P)
    print(f"V0 cycles={v0['open_time'].nunique()} rows={len(v0)}  V5 cycles={v5['open_time'].nunique()} rows={len(v5)}")
    print(f"avg names/cycle: V0={len(v0)/v0['open_time'].nunique():.1f} V5={len(v5)/v5['open_time'].nunique():.1f}\n")

    # ---- 1. per-cycle IC ----
    print("=== 1. PER-CYCLE RANK-IC ===")
    for n, df in [("V0", v0), ("V5", v5)]:
        ic = per_cycle_ic(df)
        full = spearmanr(df["pred"], df["alpha_A"]).correlation
        print(f"{n}: full-XS IC={full:+.4f} | per-cycle mean={ic.mean():+.4f} std={ic.std():.4f} "
              f"%pos={100*(ic>0).mean():.1f}% IC-Sharpe={ic.mean()/ic.std():+.3f}")
    print()

    # ---- 2. tail-pick quality (THE KEY) ----
    print("=== 2. TAIL K=3 SPREAD (only extremes trade) ===")
    ts0 = tail_spread(v0)
    ts5 = tail_spread(v5)
    for n, ts in [("V0", ts0), ("V5", ts5)]:
        print(f"{n}: spread mean={ts['spread'].mean()*1e4:+.2f}bps Sharpe={sharpe(ts['spread']):+.3f} "
              f"| long_leg mean={ts['long_leg'].mean()*1e4:+.2f}bps Sh={sharpe(ts['long_leg']):+.3f} "
              f"| short_leg mean={ts['short_leg'].mean()*1e4:+.2f}bps Sh={sharpe(-ts['short_leg']):+.3f}")
    print()

    # ---- 3. decile profile ----
    print("=== 3. REALIZED alpha_A BY PRED-DECILE (bps) ===")
    d0 = decile_profile(v0) * 1e4
    d5 = decile_profile(v5) * 1e4
    prof = pd.DataFrame({"V0_bps": d0, "V5_bps": d5})
    prof["V5-V0"] = prof["V5_bps"] - prof["V0_bps"]
    print(prof.round(2).to_string())
    print(f"\nTraded tails: dec0(short)+dec9(long) realized L-S spread:")
    print(f"  V0: {d0[9]-d0[0]:+.2f}bps   V5: {d5[9]-d5[0]:+.2f}bps")
    print(f"Middle deciles (1-8) mean |alpha|: V0={d0.iloc[1:9].abs().mean():.2f} V5={d5.iloc[1:9].abs().mean():.2f}")
    print()

    # ---- 4. concentration ----
    print("=== 4. SELECTION CONCENTRATION (top symbols by select-freq) ===")
    for n, ts in [("V0", ts0), ("V5", ts5)]:
        c = sel_freq(ts)
        tot = sum(c.values())
        top10 = c.most_common(10)
        share = sum(v for _, v in top10) / tot
        print(f"{n}: top-10 syms = {100*share:.1f}% of all picks | {[(s, v) for s, v in top10]}")
    # per-symbol pred-std (volatility of predictions)
    print("\n  per-symbol pred-std (mean across symbols) & is selection corr w/ pred-std?")
    for n, df, ts in [("V0", v0, ts0), ("V5", v5, ts5)]:
        ps = df.groupby("symbol")["pred"].std()
        c = sel_freq(ts)
        sf = pd.Series(c).reindex(ps.index).fillna(0)
        rho = spearmanr(ps, sf).correlation
        print(f"  {n}: mean pred-std={ps.mean():.4f} | rho(pred-std, sel-freq)={rho:+.3f}")
    print()

    # ---- 5. turnover ----
    print("=== 5. BASKET TURNOVER (frac of 2K=6 names changed per cycle) ===")
    print(f"  V0={basket_turnover(ts0):.3f}  V5={basket_turnover(ts5):.3f}")
    print()

    # ---- 6. net-beta of basket ----
    print("=== 6. NET-BETA PROXY (trailing 60-cycle beta of alpha_A... use raw return vs BTC) ===")
    btc = load_btc()
    if btc is not None:
        for n, ts, df in [("V0", ts0, v0), ("V5", ts5, v5)]:
            nb = net_beta(ts, df, btc)
            print(f"  {n}: mean |net-beta| basket={nb:.4f}")
    else:
        print("  BTC klines unavailable, skipping.")
    print()

    # ---- summary: cost-net per-cycle PnL Sharpe (THE portfolio metric) ----
    print("=== PORTFOLIO PnL: per-cycle equal-wt K=3 L-S, NET of turnover cost ===")
    print("    cost = (#legs changed this cycle) * cost_bps / (2K legs)  [per-name avg]")
    for cost_bps in [4.5, 9.0]:
        print(f"  -- cost {cost_bps} bps/leg entry --")
        for n, ts in [("V0", ts0), ("V5", ts5)]:
            net = cost_net_pnl(ts, cost_bps)
            gross_sh = sharpe(ts["spread"])
            net_sh = sharpe(net)
            print(f"    {n}: gross Sh={gross_sh:+.3f} gross mean={ts['spread'].mean()*1e4:+.2f}bps "
                  f"| NET Sh={net_sh:+.3f} net mean={np.mean(net)*1e4:+.2f}bps")
    print()


def cost_net_pnl(ts, cost_bps):
    """Per-cycle realized spread minus cost of legs that changed vs prior cycle.
    Spread is per-name average; cost charged per changed leg averaged over 2K legs."""
    ts = ts.sort_values("open_time").reset_index(drop=True)
    out = []
    prev_top, prev_bot = set(), set()
    for i in range(len(ts)):
        top = set(ts.loc[i, "top_syms"])
        bot = set(ts.loc[i, "bot_syms"])
        n_changed = len(top - prev_top) + len(bot - prev_bot)
        cost = n_changed * (cost_bps / 1e4) / (2 * K)
        out.append(ts.loc[i, "spread"] - cost)
        prev_top, prev_bot = top, bot
    return np.array(out)


def load_btc():
    import glob
    files = sorted(glob.glob("/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"))
    if not files:
        return None
    parts = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            parts.append(d)
        except Exception:
            pass
    if not parts:
        return None
    b = pd.concat(parts)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b = b.dropna(subset=["open_time"]).set_index("open_time").sort_index()
    b4 = b["close"].resample("4h").last()
    return b4.pct_change()


def net_beta(ts, df, btc_ret):
    """Approx: long-leg uses raw return_pct; estimate basket beta via cycle-level OLS on BTC 4h ret.
    Net-beta = beta(long basket raw ret) - beta(short basket raw ret)."""
    rows = []
    g = df.groupby("open_time")
    for _, r in ts.iterrows():
        t = r["open_time"]
        try:
            sub = g.get_group(t)
        except KeyError:
            continue
        long_r = sub[sub["symbol"].isin(r["top_syms"])]["return_pct"].mean()
        short_r = sub[sub["symbol"].isin(r["bot_syms"])]["return_pct"].mean()
        br = btc_ret.get(t, np.nan)
        rows.append((long_r - short_r, br))
    arr = pd.DataFrame(rows, columns=["ls", "btc"]).dropna()
    if len(arr) < 50 or arr["btc"].std() == 0:
        return np.nan
    beta = np.polyfit(arr["btc"], arr["ls"], 1)[0]
    return beta


if __name__ == "__main__":
    main()
