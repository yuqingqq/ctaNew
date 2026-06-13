"""Free orthogonal signals correlation test (Glassnode pre-flight).

Before committing to a paid Glassnode subscription, test what's freely available
to see if any signal can match the top cohort predictor (btc_rvol_7d Sharpe
spread q4-q0 = +15.77) or the second-best (btc_ret_3d +11.32).

Free orthogonal signals tested:
  - ETH/BTC ratio (24h change, 7d change) — cross-asset momentum
  - Universe-mean funding rate at entry
  - Universe funding dispersion (cross-symbol std of funding rates)
  - Cross-symbol return dispersion at entry (xs std of 24h returns)
  - Hours-to-next-funding-tick (Binance funds every 8h at 00/08/16 UTC)
  - Day-of-month (proxy for monthly options expiry)
  - Day-of-week

Decision rule:
  If any free signal Sharpe spread > 11 → Glassnode subscription is justified
  If all free signals < 5 → on-chain data unlikely to help 4h CS strategy
  In between → mixed evidence; consider correlated alternatives
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_free_signals"
OUT.mkdir(parents=True, exist_ok=True)
CYCLES_PER_YEAR = (288 * 365) / 48
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
FUNDING_DIR = REPO / "data/ml/cache"


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def load_sym_close(sym):
    sym_dir = KLINES_DIR / sym / "5m"
    if not sym_dir.exists(): return None
    files = sorted(sym_dir.glob("*.parquet"))
    if not files: return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
    return df["close"]


def load_all_funding(symbols):
    """Load funding for each symbol; aggregate to 5m grid by ffill from 8h ticks."""
    out = {}
    for sym in symbols:
        path = FUNDING_DIR / f"funding_{sym}.parquet"
        if not path.exists(): continue
        try:
            df = pd.read_parquet(path)
            tcol = "fundingTime" if "fundingTime" in df.columns else "open_time"
            rcol = "fundingRate" if "fundingRate" in df.columns else "funding_rate"
            if tcol not in df.columns or rcol not in df.columns: continue
            df[tcol] = pd.to_datetime(df[tcol], utc=True).astype("datetime64[ns, UTC]")
            s = df.set_index(tcol)[rcol].astype(float).sort_index()
            out[sym] = s
        except Exception as e:
            continue
    return out


def main():
    print("=== Free orthogonal signals correlation test ===\n", flush=True)

    # Load cohort PnLs from iter 2
    coh = pd.read_csv(REPO / "outputs/vBTC_iter_loop/iter2_cohort_pnl.csv")
    coh["time"] = pd.to_datetime(coh["time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  Loaded {len(coh)} cohorts from iter 2", flush=True)
    print(f"  Cohort PnL: mean={coh['cohort_pnl_bps_24h'].mean():+.2f}  "
          f"Sharpe={_sharpe(coh['cohort_pnl_bps_24h']):+.2f}\n", flush=True)

    # ------ Build free orthogonal features at each cohort time ------
    times = coh["time"].sort_values().unique()
    feat_rows = []

    print(f"  loading BTC + ETH klines...", flush=True)
    t0 = time.time()
    btc_close = load_sym_close("BTCUSDT")
    eth_close = load_sym_close("ETHUSDT")
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # ETH/BTC ratio over time
    btc_re = btc_close.reindex(btc_close.index)
    eth_re = eth_close.reindex(btc_close.index, method="nearest", tolerance=pd.Timedelta("5min"))
    ethbtc = (eth_re / btc_re).dropna()

    # Load all 51 sym closes for cross-symbol dispersion
    print(f"  loading 51-panel closes for cross-symbol dispersion...", flush=True)
    t0 = time.time()
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    closes = {}
    for sym in all_syms:
        c = load_sym_close(sym)
        if c is not None:
            closes[sym] = c
    print(f"  loaded {len(closes)} symbols ({time.time()-t0:.0f}s)", flush=True)

    # Build a return dispersion series (cross-sectional std of 24h returns)
    print(f"  computing cross-symbol 24h return dispersion...", flush=True)
    t0 = time.time()
    # Compute 24h pct change for each symbol on a daily grid (anchor on 4h)
    # For speed, only at cohort times
    xs_disp = {}
    for ts in times:
        rets_at = []
        try:
            for sym, c in closes.items():
                idx = c.index.get_indexer([ts], method="nearest")[0]
                idx_lag = idx - 288
                if idx_lag < 0: continue
                p_now = c.iloc[idx]; p_lag = c.iloc[idx_lag]
                if p_lag > 0 and not pd.isna(p_now) and not pd.isna(p_lag):
                    rets_at.append((p_now - p_lag) / p_lag)
            if len(rets_at) >= 10:
                xs_disp[ts] = float(np.std(rets_at))
        except Exception:
            pass
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Load funding for all symbols
    print(f"  loading funding...", flush=True)
    t0 = time.time()
    funding = load_all_funding(all_syms)
    print(f"  loaded funding for {len(funding)} symbols ({time.time()-t0:.0f}s)",
          flush=True)

    # For each cohort time, compute features
    print(f"  building feature panel...", flush=True)
    t0 = time.time()
    for ts in times:
        row = {"time": ts}
        # 1. ETH/BTC ratio change 24h
        try:
            idx = ethbtc.index.get_indexer([ts], method="nearest")[0]
            idx_24h = idx - 288
            if idx_24h >= 0:
                row["ethbtc_change_24h"] = (ethbtc.iloc[idx] - ethbtc.iloc[idx_24h]) / ethbtc.iloc[idx_24h]
            idx_7d = idx - 288 * 7
            if idx_7d >= 0:
                row["ethbtc_change_7d"] = (ethbtc.iloc[idx] - ethbtc.iloc[idx_7d]) / ethbtc.iloc[idx_7d]
        except Exception:
            pass

        # 2. Universe-mean funding rate at most-recent tick
        fr_now = []
        for sym, fs in funding.items():
            try:
                idx_funding = fs.index.get_indexer([ts], method="ffill")[0]
                if idx_funding >= 0 and idx_funding < len(fs):
                    fr_now.append(fs.iloc[idx_funding])
            except Exception:
                pass
        if len(fr_now) >= 10:
            row["univ_funding_mean"] = float(np.mean(fr_now))
            row["univ_funding_std"] = float(np.std(fr_now))

        # 3. Cross-symbol return dispersion
        row["xs_ret_disp_24h"] = xs_disp.get(ts, np.nan)

        # 4. Hours to next funding tick (next 00/08/16 UTC)
        hour = pd.Timestamp(ts).hour
        if hour < 8: row["hours_to_funding"] = 8 - hour
        elif hour < 16: row["hours_to_funding"] = 16 - hour
        else: row["hours_to_funding"] = 24 - hour + 0

        # 5. Day-of-month
        row["day_of_month"] = pd.Timestamp(ts).day

        # 6. Day-of-week
        row["dow"] = pd.Timestamp(ts).dayofweek

        feat_rows.append(row)
    feats = pd.DataFrame(feat_rows)
    print(f"  built {len(feats)} feature rows ({time.time()-t0:.0f}s)\n", flush=True)

    # Merge with cohort PnL
    df = coh.merge(feats, on="time", how="inner")
    df.to_csv(OUT / "free_signals_features.csv", index=False)
    print(f"  merged: {len(df)} cohorts with features", flush=True)
    feature_cols = ["ethbtc_change_24h", "ethbtc_change_7d", "univ_funding_mean",
                     "univ_funding_std", "xs_ret_disp_24h", "hours_to_funding",
                     "day_of_month", "dow"]
    print(f"  feature coverage:", flush=True)
    for f in feature_cols:
        if f in df.columns:
            print(f"    {f:>22}: {df[f].notna().sum()} / {len(df)}",
                  flush=True)

    # ------ Bucket analysis ------
    print(f"\n=== Per-feature cohort PnL Sharpe spread (q4 - q0) ===\n", flush=True)
    rows = []
    for f in feature_cols:
        if f not in df.columns: continue
        sub = df[df[f].notna()].copy()
        if len(sub) < 25: continue
        try:
            sub["b"] = pd.qcut(sub[f], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        if sub["b"].nunique() < 5: continue
        means = sub.groupby("b")["cohort_pnl_bps_24h"].mean()
        sharpes = sub.groupby("b")["cohort_pnl_bps_24h"].apply(_sharpe)
        spread_mean = means.iloc[-1] - means.iloc[0]
        spread_sh = sharpes.iloc[-1] - sharpes.iloc[0]
        rows.append({
            "feature": f,
            "n_total": len(sub),
            "mean_spread_bps": spread_mean,
            "sharpe_spread": spread_sh,
            "best_bucket_mean": means.max(),
            "worst_bucket_mean": means.min(),
            "best_bucket_sharpe": sharpes.max(),
            "worst_bucket_sharpe": sharpes.min(),
        })
    ranks = pd.DataFrame(rows).sort_values("sharpe_spread", key=lambda x: x.abs(),
                                                ascending=False)
    print(ranks.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    ranks.to_csv(OUT / "free_signals_ranking.csv", index=False)

    # ------ Decision ------
    print(f"\n=== Decision ===", flush=True)
    print(f"  Reference: btc_rvol_7d Sharpe spread = +15.77 (top from iter 2)",
          flush=True)
    print(f"  Reference: btc_ret_3d  Sharpe spread = +11.32 (2nd)", flush=True)
    print()
    best_free = ranks.iloc[0] if len(ranks) > 0 else None
    if best_free is not None:
        abs_spread = abs(best_free["sharpe_spread"])
        print(f"  Best free signal: {best_free['feature']}  "
              f"Sharpe spread = {best_free['sharpe_spread']:+.2f}", flush=True)
        if abs_spread > 11:
            print(f"\n  → STRONG free signal exists (spread > 11). Glassnode "
                  f"subscription likely justified; orthogonal info IS available "
                  f"in the cross-sectional structure.", flush=True)
        elif abs_spread > 5:
            print(f"\n  → MODERATE free signal (5 < spread < 11). On-chain "
                  f"metrics MIGHT add value. Worth testing 1-2 Glassnode "
                  f"metrics on cheapest tier as proof-of-concept.", flush=True)
        else:
            print(f"\n  → WEAK free signals (all < 5). Orthogonal information "
                  f"doesn't seem to live in the data we have. Glassnode "
                  f"unlikely to help. Pivot to operational deployment.",
                  flush=True)


if __name__ == "__main__":
    main()
